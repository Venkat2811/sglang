"""Warm fixed-batch SQD timing with matched full-model controls and GPU energy.

This measures model execution, not an HTTP serving system. Both SQD and
native controls can require CUDA graph replay. Run the separate
correctness runner first. See README.md for comparison boundaries.
"""

import argparse
import json
import multiprocessing
import time
from contextlib import contextmanager
from pathlib import Path

import pynvml
import torch

from sglang.benchmark.one_batch import load_model
from sglang.benchmark.sqd import advance, forward, prepare_batch
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
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import SpawnRanks, publish
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger


@contextmanager
def phase(barrier):
    # The parent reads energy before releasing every worker, then waits for
    # every CUDA stream to finish. There is no per-token timing or CPU copy.
    torch.cuda.synchronize()
    barrier.wait()
    barrier.wait()
    yield
    torch.cuda.synchronize()
    barrier.wait()


def graph_forward(runner, batch):
    fb = ForwardBatch.init_new(batch, runner, return_hidden_states_before_norm=False)
    output = runner.forward(fb)
    if fb.forward_mode.is_decode() and not output.can_run_graph:
        raise AssertionError("Required CUDA graph replay fell back to eager decode")
    if fb.forward_mode.is_decode():
        runner.sqd_graph_replays += 1
    return output.logits_output.next_token_logits


@torch.no_grad()
def run_worker(role, rank, config, args, ports, barrier):
    gpu = role * config.tp_size + rank
    torch.cuda.set_device(gpu)
    publish(args, role="scheduler", ranks=SpawnRanks(world_rank=rank, gpu_id=gpu))
    configure_logger(args, prefix=f" PERF{role} TP{rank}")
    transport = service = model = group = None
    layer_ids = ()

    def prepare_split(runner):
        nonlocal transport, service, model, group, layer_ids
        model = runner.model
        if type(model).__name__ != "KimiLinearForCausalLM":
            raise ValueError("This benchmark supports Kimi-Linear only")
        layer_ids = tuple(
            i
            for i in range(len(model.model.layers))
            if not model.config.is_kda_layer(i)
        )
        group = StatelessProcessGroup.create(
            "127.0.0.1", config.pair_port + rank, role, 2
        )
        transport = PairedTransport(group, torch.device(f"cuda:{gpu}"))
        if role == 0:
            service = MLAService(
                model,
                layer_ids,
                transport,
                model.config.hidden_size,
                next(model.parameters()).dtype,
            )
            runner.model = service
        else:
            for i in layer_ids:
                model.model.layers[i].self_attn = RemoteMLAAttention(transport)
            torch.cuda.empty_cache()

    # Both peers capture the same batch sizes and NCCL send/receive sequence.
    # Restore the full model for each prefill; its decode graph keeps the service.
    wrapped, tokenizer = load_model(
        args,
        ports,
        gpu,
        rank,
        before_capture=prepare_split if config.mode == "split" else None,
    )
    runner = wrapped.torch_runner
    runner.sqd_graph_replays = 0
    if model is None:
        model = runner.model
    if type(model).__name__ != "KimiLinearForCausalLM":
        raise ValueError("This benchmark supports Kimi-Linear only")
    model_forward = graph_forward if config.decode_graphs else forward

    for case, prompts in enumerate(config.cases):
        indices = list(range(len(prompts)))
        if config.mode == "replicas":
            indices = indices[role::2]
        tokens = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompts[i]}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            for i in indices
        ]
        if any(
            len(ids) + config.output_tokens > config.context_length for ids in tokens
        ):
            raise ValueError("Prompt plus output exceeds context length")
        if transport and group.all_gather_obj(tokens) != [tokens, tokens]:
            raise ValueError("SQD peers tokenized different inputs")
        for trial in range(config.warmup + config.repeat):
            runner.model = model
            batch = prepare_batch(runner, tokens)
            generated = []
            with phase(barrier):
                if config.mode != "split" or role == 0:
                    logits = model_forward(runner, batch)
                    next_ids = logits.argmax(dim=-1)
            if transport:
                with phase(barrier):
                    if role == 0:
                        transport.send(next_ids.contiguous())
                    else:
                        next_ids = transport.recv(
                            torch.empty(len(tokens), dtype=torch.int64, device=gpu)
                        )
                    transfer_recurrent_state(runner, batch, transport, sender=role == 0)
                if role == 0:
                    runner.model = service
            generated.append(next_ids)
            calls_before = service.calls if service else 0
            bytes_before = transport.sent_bytes if transport else 0
            with phase(barrier):
                for _ in range(config.output_tokens - 1):
                    if transport:
                        if role == 1:
                            transport.send(next_ids.contiguous())
                        else:
                            next_ids = transport.recv(torch.empty_like(next_ids))
                    advance(batch, next_ids)
                    logits = model_forward(runner, batch)
                    if not service:
                        next_ids = logits.argmax(dim=-1)
                        generated.append(next_ids)
            if (
                service
                and not config.decode_graphs
                and service.calls - calls_before
                != len(layer_ids) * (config.output_tokens - 1)
            ):
                raise AssertionError("Not every MLA layer ran remotely")
            if rank == 0 and not service:
                record = {
                    "case": case,
                    "trial": trial,
                    "role": role,
                    "request_indices": indices,
                    "prompt_lengths": list(map(len, tokens)),
                    "ids": torch.stack(generated).cpu().tolist(),
                    "finite_final_logits": bool(torch.isfinite(logits).all()),
                    # Python counters do not execute during graph replay.
                    "decode_sent_bytes": (
                        None
                        if transport and config.decode_graphs
                        else transport.sent_bytes - bytes_before
                        if transport
                        else 0
                    ),
                }
                stem = f"case-{case}-trial-{trial}-role-{role}"
                (config.output_dir / f"{stem}.json").write_text(
                    json.dumps(record) + "\n"
                )
                if trial == config.warmup + config.repeat - 1:
                    torch.save(
                        logits.cpu(),
                        config.output_dir / f"case-{case}-role-{role}-final.pt",
                    )
                if not record["finite_final_logits"]:
                    raise AssertionError("Nonfinite logits")
    (config.output_dir / f"rank-{role}-{rank}.json").write_text(
        json.dumps({"decode_graph_replays": runner.sqd_graph_replays}) + "\n"
    )
    if config.decode_graphs:
        # Captured NCCL operations retain the communicator until graph release.
        torch.cuda.synchronize()
        runner.decode_cuda_graph_runner.backend.cleanup()
    if transport:
        transport.close()
    destroy_model_parallel()
    destroy_distributed_environment()


def worker(*args):
    try:
        run_worker(*args)
    except BaseException:
        args[-1].abort()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--mode", choices=["full", "split", "replicas"], required=True)
    parser.add_argument(
        "--decode-graphs",
        action="store_true",
        help="Use and require CUDA graph replay on every decode step and pool",
    )
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--dcp-size", type=int, required=True)
    parser.add_argument(
        "--budget-gpus",
        type=int,
        required=True,
        help="Energy includes these first visible GPUs, even when idle",
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--pair-port", type=int, default=29750)
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Maximum seconds waiting for any phase, including model load",
    )
    config = parser.parse_args()
    pools = 1 if config.mode == "full" else 2
    if config.tp_size < 1 or config.dcp_size < 1 or config.tp_size % config.dcp_size:
        parser.error("DCP must divide TP")
    if not pools * config.tp_size <= config.budget_gpus <= torch.cuda.device_count():
        parser.error("GPU budget must cover all workers and fit visible GPUs")
    if (
        config.warmup < 1
        or config.repeat < 1
        or config.output_tokens < 2
        or config.timeout <= 0
    ):
        parser.error(
            "Positive warmup, repeat and timeout, and at least two tokens required"
        )
    if config.context_length <= config.output_tokens:
        parser.error("Context must exceed output length")
    config.cases = json.loads(config.cases.read_text())
    if (
        not isinstance(config.cases, list)
        or not config.cases
        or any(
            not isinstance(batch, list)
            or not batch
            or any(not isinstance(p, str) or not p.strip() for p in batch)
            for batch in config.cases
        )
    ):
        parser.error("Cases must be a nonempty list of nonempty prompt batches")
    if config.mode == "replicas" and any(len(batch) < 2 for batch in config.cases):
        parser.error("Two replicas require at least two requests per batch")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    max_batch = max(map(len, config.cases))
    if config.mode == "replicas":
        max_batch = (max_batch + 1) // 2
    capture_bs = sorted(
        {
            len(batch[role::pools]) if config.mode == "replicas" else len(batch)
            for batch in config.cases
            for role in range(pools)
        }
    )
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
        disable_cuda_graph=not config.decode_graphs,
        cuda_graph_config={
            "prefill": {"backend": "disabled"},
            "decode": {
                "backend": "full",
                "max_bs": max_batch,
                "bs": capture_bs,
            },
        }
        if config.decode_graphs
        else None,
        attention_backend="flashinfer",
        linear_attn_backend="triton",
        random_seed=42,
    )
    args.resolve_once()
    _set_envs_and_config(args)
    ports = [PortArgs.init_new(args) for _ in range(pools)]
    pynvml.nvmlInit()
    handles = [
        pynvml.nvmlDeviceGetHandleByUUID(str(torch.cuda.get_device_properties(i).uuid))
        for i in range(config.budget_gpus)
    ]
    metadata = vars(config).copy()
    metadata.update(
        output_dir=str(config.output_dir),
        torch=torch.__version__,
        gpus=[
            {
                "uuid": pynvml.nvmlDeviceGetUUID(h),
                "name": pynvml.nvmlDeviceGetName(h),
                "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000,
            }
            for h in handles
        ],
        energy="Whole-GPU NVML counter deltas; no idle subtraction; excludes host CPU",
    )
    (config.output_dir / "launch.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(pools * config.tp_size + 1, timeout=config.timeout)
    children = [
        context.Process(
            target=worker, args=(role, rank, config, args, ports[role], barrier)
        )
        for role in range(pools)
        for rank in range(config.tp_size)
    ]
    records = []
    try:
        for child in children:
            child.start()
        for case, prompts in enumerate(config.cases):
            for trial in range(config.warmup + config.repeat):
                for stage in (
                    ["prefill", "handoff", "decode"]
                    if config.mode == "split"
                    else ["prefill", "decode"]
                ):
                    barrier.wait()
                    energy0 = [
                        pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in handles
                    ]
                    start = time.perf_counter()
                    barrier.wait()
                    barrier.wait()
                    seconds = time.perf_counter() - start
                    energy1 = [
                        pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in handles
                    ]
                    joules = [(b - a) / 1000 for a, b in zip(energy0, energy1)]
                    if any(value < 0 for value in joules):
                        raise RuntimeError("GPU energy counter reset")
                    record = {
                        "case": case,
                        "trial": trial,
                        "warmup": trial < config.warmup,
                        "stage": stage,
                        "seconds": seconds,
                        "gpu_joules": joules,
                    }
                    if stage == "decode":
                        count = len(prompts) * (config.output_tokens - 1)
                        record.update(
                            tokens=count,
                            tpot_ms=seconds * 1000 / (config.output_tokens - 1),
                            tokens_per_second=count / seconds,
                            tokens_per_joule=count / sum(joules),
                        )
                    records.append(record)
                    (config.output_dir / "measurements.json").write_text(
                        json.dumps(records, indent=2) + "\n"
                    )
                    print(json.dumps(record), flush=True)
        deadline = time.monotonic() + 60
        for child in children:
            child.join(timeout=max(0, deadline - time.monotonic()))
        if any(child.exitcode != 0 for child in children):
            raise RuntimeError("Benchmark worker failed or did not exit")
    finally:
        barrier.abort()
        for child in children:
            if child.is_alive():
                child.kill()
        for child in children:
            if child.pid is not None:
                child.join()
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
