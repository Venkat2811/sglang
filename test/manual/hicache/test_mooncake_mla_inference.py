"""Bounded one-GPU MLA inference witness with a fresh Mooncake reader.

MOONCAKE_SMOKE_OUTPUT_DIR must name a new artifact directory. Run with one GPU:
  python -m pytest test/manual/hicache/test_mooncake_mla_inference.py -q -s

This is DCP1 characterization, not distributed DCP or hybrid qualification.
The private TCP donor survives both engines; neither engine owns store capacity.
No shared service or cache is modified. Model download/JIT are startup costs.
Set MOONCAKE_SMOKE_MODE=cold_repeat to diagnose uncached numerical variation
before interpreting a restore parity failure.
The default uses SGLang's deterministic Triton MLA path. Set
MOONCAKE_SMOKE_PROFILE=flashinfer for the ordinary, potentially batch-sensitive
backend; exact parity failures in its cold control are not storage regressions.
"""

import hashlib
import json
import math
import os
import socket
import subprocess
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import requests
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
        profile = os.environ.get("MOONCAKE_SMOKE_PROFILE", "deterministic")
        self.assertIn(profile, ("deterministic", "flashinfer"))
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(
            MODEL, revision=REVISION, trust_remote_code=True
        )
        cases = []
        for concurrency in (1, 4):
            for index, length in enumerate(LENGTHS):
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
                    {"name": f"c{concurrency}-n{length}", "c": concurrency, "ids": ids}
                )
        # No two prompts share a complete first page, even across concurrency arms.
        self.assertEqual(len({tuple(c["ids"][:PAGE]) for c in cases}), len(cases))
        serialized = json.dumps(cases, sort_keys=True)
        (output / "cases.json").write_text(serialized)
        report = {
            "model": MODEL,
            "revision": REVISION,
            "cases_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "tp": 1,
            "dcp": 1,
            "page_size": PAGE,
            "chunked_prefill_size": 256,
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

        port = find_available_port(31000)
        url = f"http://127.0.0.1:{port}"
        common = [
            "--revision",
            REVISION,
            "--trust-remote-code",
            "--host",
            "127.0.0.1",
            "--tp-size",
            "1",
            "--dcp-size",
            "1",
            "--attention-backend",
            "triton" if profile == "deterministic" else "flashinfer",
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "auto",
            "--page-size",
            str(PAGE),
            "--context-length",
            "4096",
            "--chunked-prefill-size",
            "256",
            "--max-total-tokens",
            "8192",
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

        def run_phase(phase):
            snapshot(phase, "before")

            def generate(case):
                payload = {
                    "input_ids": case["ids"],
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 8,
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
                    return record["response"]
                except Exception as error:
                    record["error"] = repr(error)
                    raise
                finally:
                    path.write_text(json.dumps(record, indent=2))

            responses = []
            for concurrency in (1, 4):
                cohort = [c for c in cases if c["c"] == concurrency]
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

        def assert_parity(reference, actual):
            for case, expected, result in zip(cases, reference, actual):
                self.assertEqual(
                    result["output_ids"], expected["output_ids"], case["name"]
                )
                self.assertEqual(len(result["output_ids"]), 8, case["name"])
                a = result["meta_info"]["output_token_logprobs"]
                b = expected["meta_info"]["output_token_logprobs"]
                self.assertEqual(len(a), 8)
                self.assertEqual(len(b), 8)
                for got, want in zip(a, b):
                    self.assertEqual(got[1], want[1])
                    self.assertTrue(math.isfinite(got[0]) and math.isfinite(want[0]))
                    self.assertLessEqual(
                        abs(got[0] - want[0]), LOGPROB_ATOL, case["name"]
                    )

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
            keys = sorted(
                {
                    f"{tag}_{MODEL.replace('/', '-')}_{key}__k"
                    for case in cases
                    for key in get_storage_hash_str(
                        case["ids"][: (len(case["ids"]) - 1) // PAGE * PAGE],
                        None,
                        page_size=PAGE,
                    )
                }
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
                for case, result in zip(cases, reader):
                    expected = (len(case["ids"]) - 1) // PAGE * PAGE
                    details = result["meta_info"]["cached_tokens_details"]
                    storage = details["storage"] if details is not None else 0
                    self.assertEqual(storage, expected, case["name"])
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
