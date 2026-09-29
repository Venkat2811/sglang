"""Bounded MLA inference witness with a fresh Mooncake reader.

MOONCAKE_SMOKE_OUTPUT_DIR must name a new artifact directory. Run with one GPU:
  python -m pytest test/manual/hicache/test_mooncake_mla_inference.py -q -s

Set MOONCAKE_SMOKE_TP=8 MOONCAKE_SMOKE_DCP=8 for eight-GPU qualification.
Distributed runs also remove one private shard and require a common shorter
prefix on every rank. This does not qualify hybrid models or disaggregation.
The private TCP donor survives both engines; neither engine owns store capacity.
No shared service or cache is modified. Model download/JIT are startup costs.
Set MOONCAKE_SMOKE_MODE=cold_repeat to diagnose uncached numerical variation
before interpreting a restore parity failure.
DCP1 defaults to deterministic Triton; DCP>1 defaults to FlashInfer, matching
the file-backed DCP inference gate. MOONCAKE_SMOKE_PROFILE overrides this choice.
FlashInfer can be batch-sensitive: a failed cold control invalidates the oracle
and is not evidence of a storage regression.
"""

import hashlib
import json
import math
import os
import re
import socket
import subprocess
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import requests
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
REVISION = "85864749cd611b4353ce1decdb286193298f64c7"
LENGTHS = (63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513)
PAGE = 64
LOGPROB_ATOL = 0.05


class TestMooncakeMlaInference(CustomTestCase):
    """Restored model continuation must agree with uncached token/logprob output.

    A wrong-page restore can preserve byte counts and still corrupt inference.
    Distinct early prefixes, exact boundary lengths and independent cold outputs
    catch that silent failure; a fresh process plus required storage attribution
    prevents L1/L2 hits from masquerading as an L3 pass.
    """

    def test_fresh_reader(self):
        from mooncake.store import MooncakeDistributedStore

        output = Path(os.environ["MOONCAKE_SMOKE_OUTPUT_DIR"])
        output.mkdir(parents=True, exist_ok=False)
        (output / "runner.py").write_bytes(Path(__file__).read_bytes())
        tp = int(os.environ.get("MOONCAKE_SMOKE_TP", "1"))
        dcp = int(os.environ.get("MOONCAKE_SMOKE_DCP", "1"))
        profile = os.environ.get(
            "MOONCAKE_SMOKE_PROFILE", "deterministic" if dcp == 1 else "flashinfer"
        )
        self.assertIn(profile, ("deterministic", "flashinfer"))
        self.assertGreaterEqual(dcp, 1)
        self.assertGreaterEqual(tp, dcp)
        self.assertEqual(tp % dcp, 0)
        self.assertGreaterEqual(torch.cuda.device_count(), tp)
        logical_page = PAGE * dcp
        chunk_size = 4 * logical_page
        # FlashInfer's C4 cold control can diverge in the free continuation
        # after the six-token answer. Qualify the complete answer instead,
        # retaining exact IDs and the same predeclared logprob tolerance.
        new_tokens = 8 if dcp == 1 else 6
        lengths = (
            LENGTHS
            if dcp == 1
            else tuple(
                boundary + delta
                for boundary in (PAGE, logical_page, 2 * logical_page, chunk_size)
                for delta in (-1, 0, 1)
            )
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=True
        )
        cases = []
        for concurrency in (1, 4):
            for index, length in enumerate(lengths):
                prefix = tokenizer.encode(
                    f"Record {concurrency}-{index}: The access code is {7300 + index}. "
                )
                suffix = tokenizer.encode(
                    "\nThe access code is", add_special_tokens=False
                )
                filler = tokenizer.encode(
                    "The library keeps records of books. ", add_special_tokens=False
                )
                remaining = length - len(prefix) - len(suffix)
                self.assertGreaterEqual(remaining, 0)
                ids = prefix + (filler * (remaining // len(filler) + 1))[:remaining]
                ids += suffix
                self.assertEqual(len(ids), length)
                cases.append(
                    {
                        "name": f"c{concurrency}-n{length}",
                        "c": concurrency,
                        "ids": ids,
                        "answer": f"{7300 + index}.",
                    }
                )
        # No two prompts share a complete first page, even across concurrency arms.
        self.assertEqual(len({tuple(c["ids"][:PAGE]) for c in cases}), len(cases))
        serialized = json.dumps(cases, sort_keys=True)
        (output / "cases.json").write_text(serialized)
        report = {
            "model": MODEL,
            "revision": REVISION,
            "cases_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "tp": tp,
            "dcp": dcp,
            "page_size": PAGE,
            "logical_page_size": logical_page,
            "chunked_prefill_size": chunk_size,
            "new_tokens": new_tokens,
            "logprob_atol": LOGPROB_ATOL,
            "phases": {},
            "status": "running",
            "profile": profile,
            "mode": os.environ.get("MOONCAKE_SMOKE_MODE", "restore"),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self.assertIn(report["mode"], ("restore", "cold_repeat"))
        report_path = output / "summary.json"

        def save():
            report_path.write_text(json.dumps(report, indent=2))

        # Pin tokenizer fixups to the same revision, even in an empty HF cache.
        tokenizer_path = snapshot_download(
            MODEL, revision=REVISION, allow_patterns=["*.json", "*.py", "*.model"]
        )
        port = find_available_port(31000)
        url = f"http://127.0.0.1:{port}"
        common = [
            "--revision",
            REVISION,
            "--tokenizer-path",
            tokenizer_path,
            "--trust-remote-code",
            "--host",
            "127.0.0.1",
            "--tp-size",
            str(tp),
            "--dcp-size",
            str(dcp),
            "--dcp-comm-backend",
            "ag_rs",
            "--attention-backend",
            "triton" if profile == "deterministic" else "flashinfer",
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "auto",
            "--page-size",
            str(PAGE),
            "--context-length",
            str(max(4096, 2 * chunk_size)),
            "--chunked-prefill-size",
            str(chunk_size),
            "--max-total-tokens",
            str(max(8192, 8 * chunk_size)),
            "--max-running-requests",
            "4",
            "--mem-fraction-static",
            "0.4",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--random-seed",
            "0",
            "--enable-cache-report",
            "--enable-metrics",
            "--log-level",
            "debug",
        ]
        if profile == "deterministic":
            common.append("--enable-deterministic-inference")
        env_overrides = (
            {"SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE": str(PAGE)}
            if profile == "deterministic"
            else {}
        )
        # The deterministic default alignment is 4096. With a smaller chunk
        # budget, radix admission rounds every partial prefill down to zero.
        # Keep alignment compatible with this deliberately tiny boundary sweep.
        report["environment_overrides"] = env_overrides

        @contextmanager
        def server(phase, extra):
            args = common + extra
            (output / f"{phase}-args.json").write_text(json.dumps(args, indent=2))
            with (output / f"{phase}-server.log").open("w") as log:
                process = None
                try:
                    process = popen_launch_server(
                        MODEL,
                        url,
                        timeout=900,
                        other_args=args,
                        env={**os.environ, **env_overrides},
                        return_stdout_stderr=(log, log),
                    )
                    info = requests.get(url + "/get_server_info", timeout=30)
                    info.raise_for_status()
                    (output / f"{phase}-server-info.json").write_text(info.text)
                    yield
                finally:
                    if process is not None:
                        terminate_and_kill_process_tree(process)

        def snapshot(phase, boundary):
            response = requests.get(url + "/metrics", timeout=30)
            response.raise_for_status()
            (output / f"{phase}-{boundary}.prom").write_text(response.text)

        def run_phase(phase, phase_cases=cases):
            snapshot(phase, "before")

            def generate(case):
                payload = {
                    "input_ids": case["ids"],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": new_tokens,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                    "logprob_start_len": len(case["ids"]) - 1,
                }
                record = {"case": case["name"], "request": payload}
                path = output / f"{phase}-{case['name']}.json"
                try:
                    response = requests.post(
                        url + "/generate", json=payload, timeout=120
                    )
                    record.update(
                        status_code=response.status_code, response=response.json()
                    )
                    response.raise_for_status()
                    if dcp > 1:
                        self.assertEqual(
                            record["response"]["text"].strip(),
                            case["answer"],
                            case["name"],
                        )
                    return record["response"]
                except Exception as error:
                    record["error"] = repr(error)
                    raise
                finally:
                    path.write_text(json.dumps(record, indent=2))

            responses = []
            for concurrency in (1, 4):
                cohort = [c for c in phase_cases if c["c"] == concurrency]
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    # Submit one bounded wave at a time. A failed request must
                    # not leave the rest of the sweep queued behind a timeout.
                    for start in range(0, len(cohort), concurrency):
                        responses.extend(
                            executor.map(generate, cohort[start : start + concurrency])
                        )
            snapshot(phase, "after")
            report["phases"][phase] = responses
            save()
            return responses

        def assert_parity(reference, actual, phase_cases=cases):
            self.assertEqual(len(actual), len(phase_cases))
            self.assertEqual(len(reference), len(phase_cases))
            for case, expected, result in zip(phase_cases, reference, actual):
                self.assertEqual(
                    result["output_ids"], expected["output_ids"], case["name"]
                )
                self.assertEqual(len(result["output_ids"]), new_tokens, case["name"])
                a = result["meta_info"]["output_token_logprobs"]
                b = expected["meta_info"]["output_token_logprobs"]
                self.assertEqual(len(a), new_tokens)
                self.assertEqual(len(b), new_tokens)
                for got, want in zip(a, b):
                    self.assertEqual(got[1], want[1])
                    self.assertTrue(math.isfinite(got[0]) and math.isfinite(want[0]))
                    self.assertLessEqual(
                        abs(got[0] - want[0]), LOGPROB_ATOL, case["name"]
                    )

        def assert_rank_prefixes(phase, expected):
            if dcp == 1:
                return
            matches = re.findall(
                r"DCP L3 prefetch: tp_rank=(\d+) tokens=(\d+)",
                (output / f"{phase}-server.log").read_text(),
            )
            by_rank = {
                rank: Counter(int(n) for r, n in matches if int(r) == rank and int(n))
                for rank in range(tp)
            }
            report.setdefault("rank_prefixes", {})[phase] = by_rank
            save()
            for rank, counts in by_rank.items():
                self.assertEqual(counts, Counter(n for n in expected if n), rank)

        master = donor = None
        try:
            save()
            with server("reference", ["--disable-radix-cache"]):
                reference = run_phase("reference")
                for result in reference:
                    self.assertEqual(result["meta_info"]["cached_tokens"], 0)

            # A cold control must pass before treating this backend as a strict
            # numerical oracle for cache restores. Never loosen tolerances in
            # response to a restored-output failure.
            with server("cold-repeat", ["--disable-radix-cache"]):
                repeated = run_phase("cold-repeat")
                for result in repeated:
                    self.assertEqual(result["meta_info"]["cached_tokens"], 0)
                assert_parity(reference, repeated)
            if report["mode"] == "cold_repeat":
                report["status"] = "passed"
                return

            master_port = find_available_port(50051)
            address = f"127.0.0.1:{master_port}"
            with (output / "master.log").open("w") as log:
                master = subprocess.Popen(
                    [
                        "mooncake_master",
                        f"--rpc_port={master_port}",
                        "--enable_metric_reporting=false",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            deadline = time.monotonic() + 30
            while True:
                self.assertIsNone(master.poll(), "Mooncake master exited")
                self.assertLess(
                    time.monotonic(), deadline, "Mooncake master startup timed out"
                )
                try:
                    with socket.create_connection(
                        ("127.0.0.1", master_port), timeout=1
                    ):
                        break
                except OSError:
                    time.sleep(0.1)
            donor = MooncakeDistributedStore()
            self.assertEqual(
                donor.setup(
                    "127.0.0.1", "P2PHANDSHAKE", 2 << 30, 16 << 20, "tcp", "", address
                ),
                0,
            )
            tag = "mla-inference-smoke"
            storage_config = {
                "master_server_address": address,
                "metadata_server": "P2PHANDSHAKE",
                "local_hostname": "127.0.0.1",
                "protocol": "tcp",
                "device_name": "",
                "global_segment_size": 0,
                "extra_backend_tag": tag,
                "prefetch_threshold": 1,
            }
            extra = [
                "--enable-hierarchical-cache",
                "--hicache-size",
                "1",
                "--hicache-write-policy",
                "write_through",
                "--hicache-mem-layout",
                "page_first",
                "--hicache-io-backend",
                "kernel",
                "--hicache-storage-backend",
                "mooncake",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
                "--hicache-storage-backend-extra-config",
                json.dumps(storage_config),
            ]
            key_prefix = f"{tag}_{MODEL.replace('/', '-')}"
            if dcp > 1:
                key_prefix += (
                    f"_dcp_v1_tp{tp}_ds{dcp}_pp1_cp0of1"
                    f"_page{logical_page}_bfloat16_page_first_pp0"
                )

            def page_keys(case):
                return [
                    [
                        f"{key_prefix}_{key}_{'dcp' + str(rank) if dcp > 1 else ''}_k"
                        for rank in range(dcp)
                    ]
                    for key in get_storage_hash_str(
                        case["ids"][
                            : (len(case["ids"]) - 1) // logical_page * logical_page
                        ],
                        None,
                        page_size=logical_page,
                    )
                ]

            keys = sorted(
                {key for case in cases for page in page_keys(case) for key in page}
            )
            with server("writer", extra):
                writer = run_phase("writer")
                for result in writer:
                    self.assertEqual(result["meta_info"]["cached_tokens"], 0)
                assert_parity(reference, writer)
                deadline = time.monotonic() + 60
                while not all(donor.is_exist(key) == 1 for key in keys):
                    self.assertLess(
                        time.monotonic(), deadline, "Prompt objects were not published"
                    )
                    time.sleep(0.1)
                expected_bytes = (
                    PAGE
                    * config.num_hidden_layers
                    * (config.kv_lora_rank + config.qk_rope_head_dim)
                    * 2
                )
                sizes = {key: len(donor.get(key)) for key in keys}
                self.assertEqual(set(sizes.values()), {expected_bytes})
                (output / "published-objects.json").write_text(
                    json.dumps(sizes, indent=2)
                )
            # Confirm that storage survives writer shutdown before starting reader.
            self.assertTrue(all(donor.is_exist(key) == 1 for key in keys))
            with server("reader", extra):
                reader = run_phase("reader")
                assert_parity(reference, reader)
                expected_prefixes = []
                for case, result in zip(cases, reader):
                    expected = (len(case["ids"]) - 1) // logical_page * logical_page
                    expected_prefixes.append(expected)
                    details = result["meta_info"]["cached_tokens_details"]
                    storage = details["storage"] if details is not None else 0
                    self.assertEqual(storage, expected, case["name"])
            assert_rank_prefixes("reader", expected_prefixes)
            if dcp > 1:
                case_index = next(
                    i
                    for i, case in enumerate(cases)
                    if case["c"] == 1 and len(case["ids"]) == chunk_size + 1
                )
                fault_case = cases[case_index]
                # A middle-page hole on one rank must stop the global prefix,
                # even when every later object and every other rank is intact.
                removed_key = page_keys(fault_case)[1][-1]
                self.assertEqual(donor.remove(removed_key, force=True), 0)
                self.assertEqual(donor.is_exist(removed_key), 0)
                self.assertTrue(
                    all(donor.is_exist(key) == 1 for key in keys if key != removed_key)
                )
                report["missing_shard"] = {
                    "key": removed_key,
                    "case": fault_case["name"],
                    "expected_prefix": logical_page,
                }
                save()
                with server("missing-shard-reader", extra):
                    missing = run_phase("missing-shard-reader", [fault_case])
                    assert_parity([reference[case_index]], missing, [fault_case])
                    self.assertEqual(
                        missing[0]["meta_info"]["cached_tokens_details"]["storage"],
                        logical_page,
                    )
                assert_rank_prefixes("missing-shard-reader", [logical_page])
            report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            save()
            if donor is not None:
                donor.close()
            if master is not None:
                master.terminate()
                try:
                    master.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    master.kill()
                    master.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
