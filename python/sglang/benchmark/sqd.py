"""Experimental, fixed-batch SQD generation with real SGLang model runners.

Pool 0 performs prefill and retains MLA KV. Pool 1 receives only recurrent
state, then runs KDA/FFNs with remote MLA calls. Compare against ordinary
generation on pool 0 before each split run. See test/manual/sqd/README.md.
"""

import argparse
import dataclasses
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

import torch

from sglang.benchmark.one_batch import (
    TreeCacheNamespace,
    load_model,
    prepare_synthetic_inputs_for_latency_test,
)
from sglang.benchmark.sqd_contract import DecodePlan
from sglang.benchmark.sqd_transport import (
    MLAService,
    PairedTransport,
    RemoteMLAAttention,
    transfer_recurrent_state,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from sglang.srt.distributed.utils import StatelessProcessGroup
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import SpawnRanks, get_schedule, publish
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import configure_logger


def prepare_batch(runner, token_ids):
    runner.req_to_token_pool.clear()
    runner.token_to_kv_pool_allocator.clear()
    reqs = prepare_synthetic_inputs_for_latency_test(
        len(token_ids), 0, custom_inputs=token_ids
    )
    tree = TreeCacheNamespace(
        page_size=get_schedule().page_size,
        device=runner.device,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=tree,
        model_config=runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device)
        batch.prefill_input_ids_cpu = None
    return batch


def forward(runner, batch):
    fb = ForwardBatch.init_new(batch, runner, return_hidden_states_before_norm=False)
    return runner.forward(fb).logits_output.next_token_logits


def advance(batch, token_ids):
    batch.input_ids = token_ids.to(torch.int64)
    batch.prepare_for_decode()


def generate_baseline(runner, token_ids, output_tokens):
    batch = prepare_batch(runner, token_ids)
    outputs, scores = [], []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for step in range(output_tokens):
        logits = forward(runner, batch)
        next_ids = logits.argmax(dim=-1)
        outputs.append(next_ids.cpu())
        scores.append(logits.float().cpu())
        if step + 1 < output_tokens:
            advance(batch, next_ids)
    torch.cuda.synchronize()
    return {
        "ids": torch.stack(outputs),
        "logits": torch.stack(scores),
        "seconds": time.perf_counter() - start,
    }


@torch.no_grad()
def worker(role, tp_rank, config, server_args, port_args):
    gpu_id = role * config.tp_size + tp_rank
    torch.cuda.set_device(gpu_id)
    publish(
        server_args,
        role="scheduler",
        ranks=SpawnRanks(world_rank=tp_rank, gpu_id=gpu_id),
    )
    configure_logger(server_args, prefix=f" SQD{role} TP{tp_rank}")
    wrapped, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)
    runner = wrapped.torch_runner
    model = runner.model
    if type(model).__name__ != "KimiLinearForCausalLM":
        raise ValueError("This prototype currently supports Kimi-Linear only")
    layer_ids = tuple(
        i for i in range(len(model.model.layers)) if not model.config.is_kda_layer(i)
    )
    group = StatelessProcessGroup.create(
        "127.0.0.1", config.pair_port + tp_rank, role, 2
    )
    transport = PairedTransport(group, torch.device(f"cuda:{gpu_id}"))
    dtype = next(model.parameters()).dtype
    hidden_size = model.config.hidden_size
    service = None
    if role == 0:
        service = MLAService(model, layer_ids, transport, hidden_size, dtype)
    else:
        for i in layer_ids:
            model.model.layers[i].self_attn = RemoteMLAAttention(transport)
        torch.cuda.empty_cache()

    for case_index, prompts in enumerate(config.cases):
        token_ids = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            for prompt in prompts
        ]
        if any(
            len(ids) + config.output_tokens > config.context_length for ids in token_ids
        ):
            raise ValueError("Prompt plus output exceeds --context-length")
        plan = DecodePlan(
            request_ids=tuple(str(i) for i in range(len(prompts))),
            prompt_lengths=tuple(map(len, token_ids)),
            mla_layers=layer_ids,
            hidden_size=hidden_size,
            dtype=str(dtype),
            tp_size=config.tp_size,
            dcp_size=config.dcp_size,
        )
        for peer in group.all_gather_obj(plan):
            plan.check_peer(peer)
        # Validate token contents too; equal lengths alone do not identify a prompt.
        peer_tokens = group.all_gather_obj(token_ids)
        if peer_tokens[0] != peer_tokens[1]:
            raise ValueError("SQD peers tokenized different inputs")

        if role == 0:
            runner.model = model
            if tp_rank == 0:
                print(f"SQD_BASELINE case={case_index}", flush=True)
            baseline = generate_baseline(runner, token_ids, config.output_tokens)
            if tp_rank == 0:
                torch.save(
                    baseline, config.output_dir / f"case-{case_index}-baseline.pt"
                )
        group.barrier()
        if tp_rank == 0:
            print(f"SQD_HANDOFF case={case_index} role={role}", flush=True)
        batch = prepare_batch(runner, token_ids)
        begin_bytes = transport.sent_bytes
        if role == 0:
            logits = forward(runner, batch)
            first_ids = logits.argmax(dim=-1)
            transport.send(first_ids.contiguous())
            transport.send(logits.contiguous())
        else:
            first_ids = transport.recv(
                torch.empty(len(prompts), dtype=torch.int64, device=gpu_id)
            )
            logits = transport.recv(
                torch.empty(
                    (len(prompts), model.config.vocab_size),
                    dtype=torch.float32,
                    device=gpu_id,
                )
            )
        transfer_recurrent_state(runner, batch, transport, sender=role == 0)
        handoff_bytes = transport.sent_bytes - begin_bytes
        if role == 0:
            runner.model = service
        generated, scores = [first_ids.cpu()], [logits.float().cpu()]
        next_ids = first_ids
        torch.cuda.synchronize()
        start = time.perf_counter()
        calls_before = (
            service.calls
            if role == 0
            else sum(model.model.layers[i].self_attn.calls for i in layer_ids)
        )
        for _ in range(config.output_tokens - 1):
            if role == 1:
                transport.send(next_ids.contiguous())
            else:
                next_ids = transport.recv(torch.empty_like(next_ids))
            advance(batch, next_ids)
            result = forward(runner, batch)
            if role == 1:
                next_ids = result.argmax(dim=-1)
                generated.append(next_ids.cpu())
                scores.append(result.float().cpu())
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        calls_after = (
            service.calls
            if role == 0
            else sum(model.model.layers[i].self_attn.calls for i in layer_ids)
        )
        expected_calls = len(layer_ids) * (config.output_tokens - 1)
        if calls_after - calls_before != expected_calls:
            raise AssertionError("Not every MLA layer executed remotely")
        record = {
            "role": "prefill_mla" if role == 0 else "kda_ffn",
            "tp_rank": tp_rank,
            "plan": dataclasses.asdict(plan),
            "remote_mla_calls": calls_after - calls_before,
            "handoff_sent_bytes": handoff_bytes,
            "decode_sent_bytes": transport.sent_bytes - begin_bytes - handoff_bytes,
            "decode_seconds": seconds,
            "mla_kv_transferred_bytes": 0,
        }
        (
            config.output_dir / f"case-{case_index}-role-{role}-rank-{tp_rank}.json"
        ).write_text(json.dumps(record, indent=2) + "\n")
        if role == 1 and tp_rank == 0:
            ids = torch.stack(generated)
            torch.save(
                {"ids": ids, "logits": torch.stack(scores)},
                config.output_dir / f"case-{case_index}-split.pt",
            )
            (config.output_dir / f"case-{case_index}-texts.json").write_text(
                json.dumps(
                    [
                        {
                            "prompt": prompt,
                            "output": tokenizer.decode(ids[:, i].tolist()),
                        }
                        for i, prompt in enumerate(prompts)
                    ],
                    indent=2,
                )
                + "\n"
            )
        group.barrier()
    print(f"SQD_WORKER_DONE role={role} rank={tp_rank}", flush=True)
    transport.close()
    destroy_model_parallel()
    destroy_distributed_environment()


def compare_results(config):
    comparisons = []
    for i in range(len(config.cases)):
        reference = torch.load(
            config.output_dir / f"case-{i}-baseline.pt", weights_only=True
        )
        actual = torch.load(config.output_dir / f"case-{i}-split.pt", weights_only=True)
        finite = bool(
            torch.isfinite(actual["logits"]).all()
            and torch.isfinite(reference["logits"]).all()
        )
        error = (
            float((actual["logits"] - reference["logits"]).abs().max())
            if finite
            else None
        )
        ids_equal = torch.equal(reference["ids"], actual["ids"])
        comparisons.append(
            {
                "case": i,
                "finite_logits": finite,
                "exact_token_ids": ids_equal,
                "max_logit_error": error,
                "logit_atol": config.logit_atol,
                "passed": finite and ids_equal and error <= config.logit_atol,
            }
        )
    (config.output_dir / "comparison.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    print(json.dumps(comparisons, indent=2), flush=True)
    if not all(item["passed"] for item in comparisons):
        raise AssertionError("SQD output differs from full-model baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", help="Pin a Hugging Face model revision")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dcp-size", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--cases", type=Path, help="JSON list of prompt batches")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-port", type=int, default=29650)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--logit-atol", type=float, default=0.05)
    config = parser.parse_args()
    if config.tp_size < 1 or config.dcp_size < 1 or config.tp_size % config.dcp_size:
        parser.error("DCP must divide TP")
    if config.output_tokens < 2:
        parser.error("At least two output tokens are required to exercise split decode")
    if config.context_length <= config.output_tokens or config.timeout <= 0:
        parser.error("Context must exceed output length, and timeout must be positive")
    if not math.isfinite(config.logit_atol) or config.logit_atol < 0:
        parser.error("Logit tolerance must be finite and nonnegative")
    if torch.cuda.device_count() < 2 * config.tp_size:
        parser.error("Requires two disjoint GPU groups, each of --tp-size GPUs")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    config.cases = (
        json.loads(config.cases.read_text())
        if config.cases
        else [
            ["What is the capital of France? Answer briefly."],
            [
                "Calculate 17 plus 25.",
                "Explain in one sentence why the sky looks blue.",
            ],
        ]
    )
    if not config.cases or any(
        not isinstance(batch, list)
        or not batch
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in batch)
        for batch in config.cases
    ):
        parser.error("Cases must be a nonempty list of nonempty prompt batches")
    max_batch = max(map(len, config.cases))
    args = ServerArgs(
        model_path=config.model_path,
        revision=config.revision,
        trust_remote_code=True,
        tp_size=config.tp_size,
        dcp_size=config.dcp_size,
        context_length=config.context_length,
        max_running_requests=max_batch,
        max_total_tokens=max_batch * config.context_length,
        max_mamba_cache_size=max(16, max_batch * 4),
        mem_fraction_static=0.7,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        attention_backend="flashinfer",
        linear_attn_backend="triton",
        random_seed=42,
    )
    args.resolve_once()
    _set_envs_and_config(args)
    ports = [PortArgs.init_new(args), PortArgs.init_new(args)]
    (config.output_dir / "launch.json").write_text(
        json.dumps(
            {
                "model": config.model_path,
                "revision": config.revision,
                "tp_per_pool": config.tp_size,
                "dcp_per_pool": config.dcp_size,
                "output_tokens": config.output_tokens,
                "cases": config.cases,
                "pid": os.getpid(),
                "torch": torch.__version__,
                "scope": "single-node fixed-batch eager Kimi-Linear; full pools allocated",
            },
            indent=2,
        )
        + "\n"
    )
    context = multiprocessing.get_context("spawn")
    children = [
        context.Process(target=worker, args=(role, rank, config, args, ports[role]))
        for role in range(2)
        for rank in range(config.tp_size)
    ]
    try:
        for child in children:
            child.start()
        deadline = time.monotonic() + config.timeout
        while any(child.is_alive() for child in children):
            if time.monotonic() > deadline:
                raise TimeoutError("SQD worker deadline exceeded")
            if any(child.exitcode not in (None, 0) for child in children):
                raise RuntimeError("SQD worker failed; inspect the worker traceback")
            for child in children:
                child.join(timeout=0.1)
        if any(child.exitcode != 0 for child in children):
            raise RuntimeError("SQD worker failed")
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
        for child in children:
            if child.pid is not None:
                child.join()
    compare_results(config)


if __name__ == "__main__":
    main()
