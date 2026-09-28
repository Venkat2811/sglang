"""Small live P/D baseline, without L3: PD_DCP_MODEL=mla|hybrid.

Requires PD_DCP_OUTPUT_DIR (new directory). PD_DCP_MODEL_PATH optionally
selects a local snapshot of the pinned revision. MLA uses P2 + D2/DCP2;
hybrid uses the existing Kimi P4/EP4 + D4/DCP4 fixture's kernels and KV dtype.
This exercises actual transfer and multiple decode tokens, not role-local L3.
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_pd_server,
    popen_launch_server,
    terminate_and_kill_process_tree,
)
from sglang.utils import wait_for_http_ready


class TestPdDcpBaseline(CustomTestCase):
    def test_transfer_continuation(self):
        kind = os.environ["PD_DCP_MODEL"]
        self.assertIn(kind, ("mla", "hybrid"))
        hybrid = kind == "hybrid"
        model_id, revision = (
            (
                "moonshotai/Kimi-Linear-48B-A3B-Instruct",
                "e1df551a447157d4658b573f9a695d57658590e9",
            )
            if hybrid
            else (
                "deepseek-ai/DeepSeek-V2-Lite-Chat",
                "85864749cd611b4353ce1decdb286193298f64c7",
            )
        )
        tp = 4 if hybrid else 2
        self.assertGreaterEqual(torch.cuda.device_count(), 2 * tp)
        model = os.environ.get("PD_DCP_MODEL_PATH", model_id)
        atol = 0.20 if hybrid else 0.05
        output = Path(os.environ["PD_DCP_OUTPUT_DIR"])
        output.mkdir(parents=True, exist_ok=False)
        source = Path(__file__).read_bytes()
        (output / "runner.py").write_bytes(source)
        report = {
            "status": "running",
            "model": model_id,
            "revision": revision,
            "prefill_tp": tp,
            "prefill_dcp": 1,
            "decode_tp": tp,
            "decode_dcp": tp,
            "logprob_atol": atol,
            "runner_sha256": hashlib.sha256(source).hexdigest(),
            "scope": "live P/D only; no HiCache or L3",
            "requests": 0,
            "checks": [],
        }

        def save():
            (output / "summary.json").write_text(json.dumps(report, indent=2))

        tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, trust_remote_code=True
        )
        logical = 64 * tp
        lengths = sorted(
            set((63, 64, 65, logical - 1, logical, logical + 1, 1023, 1024, 1025))
        )
        cases = []
        for i, size in enumerate(lengths):
            prefix = tokenizer.encode(
                "Read the repeated facts carefully.\n"
                if hybrid
                else f"Record {i}: The access code is {7300 + i}. "
            )
            filler = tokenizer.encode(
                "The library keeps records of books. ", add_special_tokens=False
            )
            suffix = tokenizer.encode(
                "\nAnswer with one word. The capital of France is"
                if hybrid
                else "\nThe access code is",
                add_special_tokens=False,
            )
            n = size - len(prefix) - len(suffix)
            self.assertGreaterEqual(n, 0)
            ids = prefix + (filler * (n // len(filler) + 1))[:n] + suffix
            self.assertEqual(len(ids), size)
            cases.append(
                {
                    "ids": ids,
                    "new_tokens": 1 if hybrid else 6,
                    "answer": None if hybrid else f"{7300 + i}.",
                }
            )
        if hybrid:
            for depth in (0.1, 0.9):
                facts = ["The library keeps records of books."] * 100
                facts.insert(
                    int(len(facts) * depth), "The secret access code is 739391."
                )
                prompt = (
                    "\n".join(facts)
                    + "\nWhat is the secret access code? Reply with only the six digits."
                )
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                ids = ids if isinstance(ids, list) else ids["input_ids"]
                cases.append(
                    {
                        "ids": ids,
                        "new_tokens": 16,
                        "answer": "739391",
                        "ignore_eos": False,
                    }
                )
        self.assertTrue(all(len(c["ids"]) + c["new_tokens"] < 4096 for c in cases))
        (output / "cases.json").write_text(json.dumps(cases))
        ports = []
        for _ in range(6):
            ports.append(find_available_port(max(ports, default=30999) + 1))
        ref_url, p_url, d_url, router_url = [f"http://127.0.0.1:{p}" for p in ports[:4]]
        bootstrap, nccl = ports[4:]
        env = {
            "SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1",
            "MC_TCP_ENABLE_CONNECTION_POOL": "true",
            "MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER": "65535",
            "MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER": "65535",
        }
        report["transfer_environment"] = {
            **env,
            "MOONCAKE_PROTOCOL": os.environ.get("MOONCAKE_PROTOCOL", "rdma"),
            "MC_FORCE_TCP": os.environ.get("MC_FORCE_TCP", "0"),
            "WITH_NVIDIA_PEERMEM": os.environ.get("WITH_NVIDIA_PEERMEM", "unset"),
        }
        common = [
            "--revision",
            revision,
            "--trust-remote-code",
            "--tp-size",
            str(tp),
            "--attention-backend",
            "tokenspeed_mla" if hybrid else "flashinfer",
            "--kv-cache-dtype",
            "fp8_e4m3" if hybrid else "auto",
            "--dtype",
            "bfloat16",
            "--page-size",
            "64",
            "--random-seed",
            "0",
            "--context-length",
            "4096",
            "--chunked-prefill-size",
            "1024",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-max-bs-decode",
            "8",
            "--mem-fraction-static",
            "0.70",
            "--max-running-requests",
            "8",
            "--enable-metrics",
        ]
        if hybrid:
            common += ["--max-mamba-cache-size", "64"]
        p_extra = ["--ep-size", str(tp)] if hybrid else []
        d_extra = [
            "--dcp-size",
            str(tp),
            "--dcp-comm-backend",
            "a2a" if hybrid else "ag_rs",
        ]
        if hybrid:
            d_extra += ["--dcp-replicate-q-proj"]

        @contextmanager
        def server(name, url, args, pd=False):
            (output / f"{name}-args.json").write_text(json.dumps(common + args))
            with (output / f"{name}-server.log").open("w") as log:
                process = None
                try:
                    launcher = popen_launch_pd_server if pd else popen_launch_server
                    process = launcher(
                        model,
                        url,
                        timeout=1200,
                        other_args=common + args,
                        env=env,
                        return_stdout_stderr=(log, log),
                    )
                    if pd:
                        wait_for_http_ready(
                            url + "/health", timeout=1200, process=process
                        )
                    r = requests.get(url + "/get_server_info", timeout=30)
                    r.raise_for_status()
                    (output / f"{name}-server-info.json").write_text(r.text)
                    yield process
                finally:
                    if process is not None:
                        terminate_and_kill_process_tree(process, wait_timeout=60)

        def flush(url):
            r = requests.post(url + "/flush_cache", params={"timeout": 30}, timeout=40)
            r.raise_for_status()

        def generate(name, url, selected):
            many = isinstance(selected, list)
            group = selected if many else [selected]
            counts = {c["new_tokens"] for c in group}
            self.assertEqual(len(counts), 1)
            eos_modes = {c.get("ignore_eos", True) for c in group}
            self.assertEqual(len(eos_modes), 1)
            payload = {
                "input_ids": [c["ids"] for c in group] if many else group[0]["ids"],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": counts.pop(),
                    "ignore_eos": eos_modes.pop(),
                },
                "return_logprob": True,
            }
            record = {"request": payload}
            try:
                r = requests.post(url + "/generate", json=payload, timeout=240)
                record.update(status_code=r.status_code, response=r.json())
                r.raise_for_status()
                results = record["response"] if many else [record["response"]]
                self.assertEqual(len(results), len(group))
                for case, result in zip(group, results):
                    if case.get("ignore_eos", True):
                        self.assertEqual(len(result["output_ids"]), case["new_tokens"])
                    else:
                        # Match the registered Kimi needle fixture: stop at EOS,
                        # but require multiple tokens to exercise decode transfer.
                        self.assertGreater(len(result["output_ids"]), 1)
                        self.assertLessEqual(
                            len(result["output_ids"]), case["new_tokens"]
                        )
                    lps = result["meta_info"]["output_token_logprobs"]
                    self.assertEqual([x[1] for x in lps], result["output_ids"])
                    self.assertTrue(all(math.isfinite(x[0]) for x in lps))
                    if case["answer"] is not None:
                        self.assertIn(case["answer"], result["text"])
                report["requests"] += len(group)
                save()
                return record["response"]
            finally:
                (output / f"{name}.json").write_text(json.dumps(record, indent=2))

        def parity(name, ref, actual):
            self.assertEqual(actual["output_ids"], ref["output_ids"], name)
            delta = max(
                abs(x[0] - y[0])
                for x, y in zip(
                    actual["meta_info"]["output_token_logprobs"],
                    ref["meta_info"]["output_token_logprobs"],
                )
            )
            self.assertLessEqual(delta, atol, name)
            report["checks"].append({"name": name, "max_logprob_delta": delta})
            save()

        save()
        try:
            with server("reference", ref_url, p_extra):
                refs = []
                for i, case in enumerate(cases):
                    flush(ref_url)
                    result = generate(f"reference-{i}", ref_url, case)
                    self.assertEqual(result["meta_info"]["cached_tokens"], 0)
                    refs.append(result)
                flush(ref_url)
                batch_refs = generate("reference-batch", ref_url, cases[6:9])
            with ExitStack() as stack:
                stack.enter_context(
                    server(
                        "prefill",
                        p_url,
                        p_extra
                        + [
                            "--disaggregation-mode",
                            "prefill",
                            "--disaggregation-bootstrap-port",
                            str(bootstrap),
                            "--nccl-port",
                            str(nccl),
                            "--disaggregation-transfer-backend",
                            "mooncake",
                        ],
                        pd=True,
                    )
                )
                stack.enter_context(
                    server(
                        "decode",
                        d_url,
                        d_extra
                        + [
                            "--disaggregation-mode",
                            "decode",
                            "--disaggregation-bootstrap-port",
                            str(bootstrap),
                            "--base-gpu-id",
                            str(tp),
                            "--nccl-port",
                            str(find_available_port(nccl + 1)),
                            "--disaggregation-transfer-backend",
                            "mooncake",
                        ],
                        pd=True,
                    )
                )
                log = stack.enter_context((output / "router.log").open("w"))
                command = [
                    sys.executable,
                    "-m",
                    "sglang_router.launch_router",
                    "--pd-disaggregation",
                    "--mini-lb",
                    "--prefill",
                    p_url,
                    "--decode",
                    d_url,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(ports[3]),
                ]
                (output / "router-args.json").write_text(json.dumps(command))
                router = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                stack.callback(terminate_and_kill_process_tree, router, wait_timeout=60)
                wait_for_http_ready(router_url + "/health", timeout=120, process=router)
                for i, (case, ref) in enumerate(zip(cases, refs)):
                    flush(p_url)
                    flush(d_url)
                    result = generate(f"pd-cold-{i}", router_url, case)
                    parity(f"pd-cold-{i}", ref, result)
                    warm = generate(f"pd-warm-{i}", router_url, case)
                    parity(f"pd-warm-{i}", ref, warm)
                flush(p_url)
                flush(d_url)
                batch = generate("pd-batch", router_url, cases[6:9])
                for i, result in enumerate(batch):
                    parity(f"pd-batch-{i}", batch_refs[i], result)
                for name, url in (("prefill", p_url), ("decode", d_url)):
                    r = requests.get(url + "/metrics", timeout=30)
                    r.raise_for_status()
                    (output / f"{name}-metrics.prom").write_text(r.text)
            report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            save()


if __name__ == "__main__":
    unittest.main()
