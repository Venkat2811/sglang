"""Bounded Kimi Linear DCP4 baseline before hybrid L3 enablement.

KIMI_DCP_OUTPUT_DIR must name a new directory. KIMI_DCP_MODEL_PATH may point
to a local snapshot of REVISION. This verifies cold/warm model continuation
with HiCache enabled, not L2 eviction, L3 restore, or live P/D transfer.
"""

import hashlib
import json
import math
import os
import unittest
from contextlib import contextmanager
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.kl_multiturn_utils import make_mamba_prefill_assert
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
REVISION = "e1df551a447157d4658b573f9a695d57658590e9"
BOUNDARIES = (63, 64, 65, 255, 256, 257, 1023, 1024, 1025)
NEW_TOKENS = 16
# Existing Kimi P/D fixture uses 0.20; the DCP HiCache fixture uses KL < 0.01.
# Fix both limits before running, and additionally require exact greedy IDs.
LOGPROB_ATOL = 0.20
KL_THRESHOLD = 0.01


class TestKimiLinearDcpBaseline(CustomTestCase):
    def test_cold_warm_continuation(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 4)
        output = Path(os.environ["KIMI_DCP_OUTPUT_DIR"])
        output.mkdir(parents=True, exist_ok=False)
        model = os.environ.get("KIMI_DCP_MODEL_PATH", MODEL)
        tokenizer = AutoTokenizer.from_pretrained(
            model, revision=REVISION, trust_remote_code=True
        )
        source = Path(__file__).read_bytes()
        (output / "runner.py").write_bytes(source)
        report = {
            "status": "running",
            "model": MODEL,
            "model_path": model,
            "revision": REVISION,
            "tp": 4,
            "dcp": 4,
            "scope": "cold and warm continuation; no forced tier restore or P/D",
            "runner_sha256": hashlib.sha256(source).hexdigest(),
            "logprob_atol": LOGPROB_ATOL,
            "kl_threshold": KL_THRESHOLD,
            "checks": [],
            "requests": 0,
        }

        def save():
            (output / "summary.json").write_text(json.dumps(report, indent=2))

        prefix = tokenizer.encode("Read the repeated facts carefully.\n")
        filler = tokenizer.encode(
            "The sky is blue and the grass is green. ", add_special_tokens=False
        )
        suffix = tokenizer.encode(
            "\nAnswer with one word. The capital of France is",
            add_special_tokens=False,
        )
        cases = []
        for size in BOUNDARIES:
            n = size - len(prefix) - len(suffix)
            self.assertGreaterEqual(n, 0)
            ids = prefix + (filler * (n // len(filler) + 1))[:n] + suffix
            self.assertEqual(len(ids), size)
            cases.append({"ids": ids, "answer": "Paris"})
        for depth in (0.1, 0.9):
            facts = ["The library keeps records of books."] * 100
            facts.insert(int(len(facts) * depth), "The secret access code is 739391.")
            prompt = "\n".join(facts) + (
                "\nWhat is the secret access code? Reply with only the six digits."
            )
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            ids = ids if isinstance(ids, list) else ids["input_ids"]
            self.assertLess(len(ids) + 2 * NEW_TOKENS + 64, 4096)
            cases.append({"ids": ids, "answer": "739391"})
        (output / "cases.json").write_text(json.dumps(cases))
        url = f"http://127.0.0.1:{find_available_port(31000)}"
        # Follow test_unified_radix_cache_kl_dcp.py. In particular, do not cap
        # max-total-tokens: that fixture documents invalid DCP KV writes.
        common = [
            "--revision",
            REVISION,
            "--trust-remote-code",
            "--tp-size",
            "4",
            "--dcp-size",
            "4",
            "--page-size",
            "64",
            "--attention-backend",
            "cutedsl_mla",
            "--dcp-comm-backend",
            "a2a",
            "--dcp-replicate-q-proj",
            "--dtype",
            "bfloat16",
            "--random-seed",
            "0",
            "--context-length",
            "4096",
            "--chunked-prefill-size",
            "1024",
            "--cuda-graph-max-bs-decode",
            "8",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--mem-fraction-static",
            "0.80",
            "--max-running-requests",
            "8",
            "--max-mamba-cache-size",
            "64",
            "--enable-metrics",
        ]

        @contextmanager
        def server(name, extra):
            (output / f"{name}-args.json").write_text(json.dumps(common + extra))
            with (output / f"{name}-server.log").open("w") as log:
                process = None
                try:
                    process = popen_launch_server(
                        model,
                        url,
                        timeout=1200,
                        other_args=common + extra,
                        env={"SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1"},
                        return_stdout_stderr=(log, log),
                    )
                    r = requests.get(url + "/get_server_info", timeout=30)
                    r.raise_for_status()
                    (output / f"{name}-server-info.json").write_text(r.text)
                    yield
                finally:
                    if process is not None:
                        terminate_and_kill_process_tree(process, wait_timeout=60)

        def flush():
            r = requests.post(url + "/flush_cache", params={"timeout": 30}, timeout=40)
            r.raise_for_status()

        def generate(name, ids, new_tokens=NEW_TOKENS, start=-1):
            payload = {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": new_tokens,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": start,
            }
            record = {"request": payload}
            try:
                r = requests.post(url + "/generate", json=payload, timeout=180)
                record.update(status_code=r.status_code, response=r.json())
                r.raise_for_status()
                report["requests"] += len(ids) if isinstance(ids[0], list) else 1
                save()
                return record["response"]
            finally:
                (output / f"{name}.json").write_text(json.dumps(record, indent=2))

        def check_result(result, answer=None, count=NEW_TOKENS):
            self.assertEqual(len(result["output_ids"]), count)
            lps = result["meta_info"]["output_token_logprobs"]
            self.assertEqual(len(lps), count)
            self.assertEqual([x[1] for x in lps], result["output_ids"])
            self.assertTrue(all(math.isfinite(x[0]) for x in lps))
            if answer is not None:
                self.assertIn(answer, result["text"])

        def parity(name, ref, result, count=NEW_TOKENS):
            check_result(ref, count=count)
            check_result(result, count=count)
            self.assertEqual(result["output_ids"], ref["output_ids"])
            a = result["meta_info"]["output_token_logprobs"]
            b = ref["meta_info"]["output_token_logprobs"]
            delta = max(abs(x[0] - y[0]) for x, y in zip(a, b))
            self.assertLessEqual(delta, LOGPROB_ATOL)
            report["checks"].append({"name": name, "max_logprob_delta": delta})
            save()

        save()
        try:
            refs, warm_refs = [], []
            with server("reference", []):
                for i, case in enumerate(cases):
                    flush()
                    result = generate(f"reference-{i}", case["ids"])
                    check_result(result, case["answer"])
                    self.assertEqual(result["meta_info"]["cached_tokens"], 0)
                    refs.append(result)
                    # Match the warm request's exact input and output lengths.
                    # Moving one token from decode into prefill is a separate
                    # numerical comparison, governed by the existing Kimi KL
                    # criterion rather than the cache-parity absolute bound.
                    flush()
                    extended = generate(
                        f"reference-extension-{i}",
                        case["ids"] + result["output_ids"][:1],
                        new_tokens=NEW_TOKENS - 1,
                    )
                    check_result(extended, count=NEW_TOKENS - 1)
                    self.assertEqual(extended["meta_info"]["cached_tokens"], 0)
                    self.assertEqual(extended["output_ids"], result["output_ids"][1:])
                    diffs = [
                        a[0] - b[0]
                        for a, b in zip(
                            extended["meta_info"]["output_token_logprobs"],
                            result["meta_info"]["output_token_logprobs"][1:],
                        )
                    ]
                    kl = sum(math.expm1(x) - x for x in diffs) / (NEW_TOKENS - 1)
                    self.assertLess(kl, KL_THRESHOLD)
                    report["checks"].append(
                        {"name": f"prefill-decode-reference-{i}", "kl": kl}
                    )
                    warm_refs.append(extended)
                    save()
                # Match checkpoint-producing work for the multi-turn control,
                # too. A warm recurrent-state path has a different numerical
                # schedule from full cold prefill even with HiCache disabled.
                flush()
                generate("reference-seed", cases[-1]["ids"])
                generate(
                    "reference-seed-extension",
                    cases[-1]["ids"] + refs[-1]["output_ids"][:1],
                    new_tokens=NEW_TOKENS - 1,
                )
                followup = (
                    cases[-1]["ids"]
                    + refs[-1]["output_ids"]
                    + tokenizer.encode(
                        "\nRepeat the access code:", add_special_tokens=False
                    )
                )
                continuation_ref = generate("reference-continuation", followup)
                check_result(continuation_ref)
                self.assertGreater(continuation_ref["meta_info"]["cached_tokens"], 0)
                flush()
                batch_refs = generate(
                    "reference-batch", [case["ids"] for case in cases[6:9]]
                )
                self.assertEqual(len(batch_refs), 3)
                for i, result in enumerate(batch_refs):
                    check_result(result)
                    self.assertEqual(result["output_ids"], refs[6 + i]["output_ids"])
            with server(
                "hicache",
                [
                    "--enable-hierarchical-cache",
                    "--hicache-size",
                    "10",
                    "--hicache-write-policy",
                    "write_through",
                ],
            ):
                for i, (case, ref) in enumerate(zip(cases, refs)):
                    flush()
                    cold = generate(f"hicache-cold-{i}", case["ids"])
                    self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
                    parity(f"hicache-cold-{i}", ref, cold)
                    # Matching leaves one input token for computing logits.
                    # An exactly page-aligned prompt can have only its terminal
                    # KDA checkpoint, which an identical prompt cannot reuse.
                    # Append the reference's first token so that checkpoint is
                    # eligible, then compare the remaining continuation exactly.
                    warm = generate(
                        f"hicache-warm-{i}",
                        case["ids"] + ref["output_ids"][:1],
                        new_tokens=NEW_TOKENS - 1,
                    )
                    parity(
                        f"hicache-warm-{i}", warm_refs[i], warm, count=NEW_TOKENS - 1
                    )
                    make_mamba_prefill_assert(256)(warm, len(case["ids"]), str(i))
                    if len(case["ids"]) >= 512:
                        self.assertGreater(warm["meta_info"]["cached_tokens"], 0)
                # A continuation crosses both KDA checkpoint and MLA boundaries.
                continued = generate("continuation-warm", followup)
                check_result(continued)
                self.assertGreater(continued["meta_info"]["cached_tokens"], 0)
                parity("continuation-vs-reference", continuation_ref, continued)
                flush()
                cold = generate("continuation-cold", followup)
                check_result(cold)
                self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
                self.assertEqual(cold["output_ids"], continued["output_ids"])
                diffs = [
                    a[0] - b[0]
                    for a, b in zip(
                        cold["meta_info"]["output_token_logprobs"],
                        continued["meta_info"]["output_token_logprobs"],
                    )
                ]
                kl = sum(math.expm1(x) - x for x in diffs) / NEW_TOKENS
                self.assertLess(kl, KL_THRESHOLD)
                report["checks"].append(
                    {
                        "name": "warm-cold-continuation",
                        "kl": kl,
                        "max_logprob_delta": max(abs(x) for x in diffs),
                    }
                )
                save()
                # Teacher-forced prefill of the same output catches decode/
                # prefill disagreement, using the existing Kimi KL contract.
                flush()
                replay = generate(
                    "continuation-replay",
                    followup + continued["output_ids"],
                    new_tokens=0,
                    start=len(followup) - 1,
                )
                replay_lps = replay["meta_info"]["input_token_logprobs"][-NEW_TOKENS:]
                self.assertEqual([x[1] for x in replay_lps], continued["output_ids"])
                out_lps = continued["meta_info"]["output_token_logprobs"]
                diffs = [x[0] - y[0] for x, y in zip(replay_lps, out_lps)]
                self.assertTrue(all(math.isfinite(x) for x in diffs))
                kl = sum(math.expm1(x) - x for x in diffs) / NEW_TOKENS
                self.assertLess(kl, KL_THRESHOLD)
                report["checks"].append({"name": "continuation-replay", "kl": kl})
                flush()
                batched = generate("batch", [case["ids"] for case in cases[6:9]])
                self.assertEqual(len(batched), 3)
                for i, result in enumerate(batched):
                    parity(f"batch-{i}", batch_refs[i], result)
                r = requests.get(url + "/metrics", timeout=30)
                r.raise_for_status()
                (output / "hicache-metrics.prom").write_text(r.text)
            report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            save()


if __name__ == "__main__":
    unittest.main()
