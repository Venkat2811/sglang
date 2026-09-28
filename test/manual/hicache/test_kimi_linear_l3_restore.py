"""Bounded Kimi DCP4 L3 continuation, with matched ordinary-radix controls.

KIMI_L3_OUTPUT_DIR and KIMI_L3_STORAGE_DIR must be new directories.
KIMI_L3_BACKEND=file|mooncake; Mooncake uses an externally owned persistent
store described by KIMI_L3_EXTRA_CONFIG. The reader starts after the writer
exits, and flushes L1/L2 before every restore. This does not prove SSD residency.
"""

import hashlib
import json
import math
import os
import re
import unittest
from contextlib import contextmanager
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
REVISION = "e1df551a447157d4658b573f9a695d57658590e9"


class TestKimiLinearL3Restore(CustomTestCase):
    def test_fresh_reader_continuation(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 4)
        backend = os.environ.get("KIMI_L3_BACKEND", "file")
        self.assertIn(backend, ("file", "mooncake"))
        output = Path(os.environ["KIMI_L3_OUTPUT_DIR"])
        storage = Path(os.environ["KIMI_L3_STORAGE_DIR"])
        output.mkdir(parents=True, exist_ok=False)
        storage.mkdir(parents=True, exist_ok=False)
        source = Path(__file__).read_bytes()
        (output / "runner.py").write_bytes(source)
        report = dict(
            status="running",
            backend=backend,
            model=MODEL,
            revision=REVISION,
            tp=4,
            dcp=4,
            requests=0,
            checks=[],
            logprob_atol=0.20,
            runner_sha256=hashlib.sha256(source).hexdigest(),
        )

        def save():
            (output / "summary.json").write_text(json.dumps(report, indent=2))

        model = os.environ.get("KIMI_L3_MODEL_PATH", MODEL)
        tokenizer = AutoTokenizer.from_pretrained(
            model, revision=REVISION, trust_remote_code=True
        )
        prefix = tokenizer.encode("Read the repeated facts carefully.\n")
        filler = tokenizer.encode(
            "The sky is blue and the grass is green. ", add_special_tokens=False
        )
        suffix = tokenizer.encode(
            "\nAnswer with one word. The capital of France is", add_special_tokens=False
        )
        cases = []
        for size in (257, 513, 1025):
            n = size - len(prefix) - len(suffix)
            cases.append(prefix + (filler * (n // len(filler) + 1))[:n] + suffix)
        (output / "cases.json").write_text(json.dumps(cases))
        url = f"http://127.0.0.1:{find_available_port(31000)}"
        common = [
            "--log-level",
            "debug",
            "--served-model-name",
            MODEL,
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
            "--enable-cache-report",
        ]
        config = {
            "prefetch_threshold": 1,
            **json.loads(os.environ.get("KIMI_L3_EXTRA_CONFIG", "{}")),
        }
        cache_args = [
            "--enable-hierarchical-cache",
            "--hicache-size",
            "10",
            "--hicache-write-policy",
            "write_through",
            "--hicache-mem-layout",
            "page_first",
            "--hicache-io-backend",
            "kernel",
            "--hicache-storage-backend",
            backend,
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend-extra-config",
            json.dumps(config),
        ]
        env = {
            "SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1",
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(storage),
        }

        @contextmanager
        def server(name, cached):
            args = common + (cache_args if cached else [])
            (output / f"{name}-args.json").write_text(json.dumps(args))
            process = None
            with (output / f"{name}-server.log").open("w") as log:
                try:
                    process = popen_launch_server(
                        model,
                        url,
                        timeout=600,
                        other_args=args,
                        env=env,
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

        def generate(name, ids):
            payload = {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            }
            r = requests.post(url + "/generate", json=payload, timeout=180)
            (output / f"{name}.json").write_text(
                json.dumps(
                    {
                        "request": payload,
                        "status_code": r.status_code,
                        "response": r.json(),
                    },
                    indent=2,
                )
            )
            r.raise_for_status()
            result = r.json()
            self.assertEqual(len(result["output_ids"]), 16)
            self.assertIn("Paris", result["text"])
            lps = result["meta_info"]["output_token_logprobs"]
            self.assertEqual([x[1] for x in lps], result["output_ids"])
            self.assertTrue(all(math.isfinite(x[0]) for x in lps))
            report["requests"] += 1
            save()
            return result

        def parity(name, ref, actual):
            self.assertEqual(actual["output_ids"], ref["output_ids"], name)
            delta = max(
                abs(a[0] - b[0])
                for a, b in zip(
                    actual["meta_info"]["output_token_logprobs"],
                    ref["meta_info"]["output_token_logprobs"],
                )
            )
            self.assertLessEqual(delta, 0.20, name)
            report["checks"].append({"name": name, "max_logprob_delta": delta})
            save()

        save()
        try:
            with server("reference", False):
                refs, cold = [], []
                for i, ids in enumerate(cases):
                    flush()
                    cold.append(generate(f"reference-cold-{i}", ids))
                    self.assertEqual(cold[-1]["meta_info"]["cached_tokens"], 0)
                    refs.append(generate(f"reference-warm-{i}", ids))
                    self.assertEqual(
                        refs[-1]["meta_info"]["cached_tokens"], len(ids) - 1
                    )
            with server("writer", True):
                for i, ids in enumerate(cases):
                    flush()
                    result = generate(f"writer-{i}", ids)
                    parity(f"writer-{i}", cold[i], result)
                # Idle flush waits for in-flight write-through/storage acknowledgments.
                flush()
            files = [
                {
                    "path": str(p.relative_to(storage)),
                    "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in storage.rglob("*.bin")
            ]
            (output / "stored-files.json").write_text(json.dumps(files, indent=2))
            if backend == "file":
                self.assertTrue(any("mamba_tp" in p["path"] for p in files))
            with server("fresh-reader", True):
                for i, ids in enumerate(cases):
                    flush()
                    result = generate(f"restore-{i}", ids)
                    parity(f"restore-{i}", refs[i], result)
                    details = result["meta_info"]["cached_tokens_details"]
                    self.assertIsNotNone(details, f"No L3 restore for case {i}")
                    self.assertEqual(details["storage"], len(ids) - 1)
                    self.assertEqual(details["host"], 0)
                    self.assertEqual(details["device"], 0)
                    report["checks"][-1]["cache_details"] = details
                    save()
                r = requests.get(url + "/metrics", timeout=30)
                r.raise_for_status()
                (output / "fresh-reader-metrics.prom").write_text(r.text)
            matches = re.findall(
                r"DCP L3 prefetch: tp_rank=(\d+) tokens=(\d+)",
                (output / "fresh-reader-server.log").read_text(),
            )
            for rank in range(4):
                observed = [int(n) for r, n in matches if int(r) == rank and int(n)]
                self.assertEqual(observed, [256, 512, 1024])
            report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            save()


if __name__ == "__main__":
    unittest.main()
