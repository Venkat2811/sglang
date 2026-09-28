"""Bounded P4 -> D4/DCP4 Kimi continuation with role-local file L3.

PD_L3_ROLES=prefill|decode|both, PD_L3_OUTPUT_DIR and PD_L3_STORAGE_DIR
must select new owned directories. PD_L3_MODEL_PATH may select the pinned
local snapshot. A cache-off live P/D pair supplies matched controls; the
writer pair exits before the fresh-reader pair starts. Prompts end one token
before a DCP page so actual decode kernels publish the next checkpoint.
"""

import hashlib
import json
import math
import os
import re
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
    terminate_and_kill_process_tree,
)
from sglang.utils import wait_for_http_ready

MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
REVISION = "e1df551a447157d4658b573f9a695d57658590e9"


class TestPdHybridL3(CustomTestCase):
    def test_role_local_continuation(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        selection = os.environ["PD_L3_ROLES"]
        self.assertIn(selection, ("prefill", "decode", "both"))
        roles = {"prefill", "decode"} if selection == "both" else {selection}
        output = Path(os.environ["PD_L3_OUTPUT_DIR"])
        storage = Path(os.environ["PD_L3_STORAGE_DIR"])
        output.mkdir(parents=True, exist_ok=False)
        storage.mkdir(parents=True, exist_ok=False)
        model = os.environ.get("PD_L3_MODEL_PATH", MODEL)
        source = Path(__file__).read_bytes()
        (output / "runner.py").write_bytes(source)
        report = dict(
            status="running",
            model=MODEL,
            revision=REVISION,
            roles=sorted(roles),
            requests=0,
            checks=[],
            logprob_atol=0.20,
            prefill_tp=4,
            decode_tp=4,
            decode_dcp=4,
            runner_sha256=hashlib.sha256(source).hexdigest(),
            transfer_environment={
                key: os.environ.get(key, "unset")
                for key in ("MOONCAKE_PROTOCOL", "MC_FORCE_TCP", "WITH_NVIDIA_PEERMEM")
            },
        )

        def save():
            (output / "summary.json").write_text(json.dumps(report, indent=2))

        tokenizer = AutoTokenizer.from_pretrained(
            model, revision=REVISION, trust_remote_code=True
        )
        # Preserve the chat framing and known-answer instruction while varying
        # only filler tokens. The marker never enters a request.
        template = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Remember the secret access code: 739391.\nINSERT_FILLER_HERE\nWhat is the secret access code? Reply with only the six digits.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        before, after = template.split("INSERT_FILLER_HERE")
        suffix = tokenizer.encode(after, add_special_tokens=False)
        filler = tokenizer.encode(
            "The library keeps records of books. ", add_special_tokens=False
        )
        primes = []
        for size in (255, 511, 1023):
            # Distinct first pages keep another case's L3 checkpoint from
            # warming this case's writer-control comparison.
            prefix = tokenizer.encode(
                before.replace("Remember", f"Record {size}. Remember"),
                add_special_tokens=False,
            )
            n = size - len(prefix) - len(suffix)
            self.assertGreater(n, 0)
            primes.append(prefix + (filler * (n // len(filler) + 1))[:n] + suffix)
        (output / "prime-inputs.json").write_text(json.dumps(primes))
        ports = []
        for _ in range(6):
            ports.append(find_available_port(max(ports, default=30999) + 1))
        p_url, d_url, router_url = [f"http://127.0.0.1:{p}" for p in ports[:3]]
        common = [
            "--revision",
            REVISION,
            "--served-model-name",
            MODEL,
            "--trust-remote-code",
            "--tp-size",
            "4",
            "--attention-backend",
            "tokenspeed_mla",
            "--kv-cache-dtype",
            "fp8_e4m3",
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
            "--max-mamba-cache-size",
            "64",
            "--enable-metrics",
            "--enable-cache-report",
            "--log-level",
            "debug",
            "--disaggregation-transfer-backend",
            "mooncake",
            "--disaggregation-bootstrap-port",
            str(ports[3]),
        ]
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
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend-extra-config",
            json.dumps({"prefetch_threshold": 1}),
        ]

        @contextmanager
        def pair(phase, cached):
            with ExitStack() as stack:
                for role, url in (("prefill", p_url), ("decode", d_url)):
                    args = common + [
                        "--disaggregation-mode",
                        role,
                        "--nccl-port",
                        str(ports[4 if role == "prefill" else 5]),
                    ]
                    if role == "prefill":
                        args += ["--ep-size", "4"]
                    else:
                        args += [
                            "--base-gpu-id",
                            "4",
                            "--dcp-size",
                            "4",
                            "--dcp-comm-backend",
                            "a2a",
                            "--dcp-replicate-q-proj",
                        ]
                        if cached and role in roles:
                            args += ["--disaggregation-decode-enable-radix-cache"]
                    if cached and role in roles:
                        args += cache_args
                    env = {
                        "SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1",
                        "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(storage / role),
                    }
                    (output / f"{phase}-{role}-args.json").write_text(json.dumps(args))
                    log = stack.enter_context(
                        (output / f"{phase}-{role}.log").open("w")
                    )
                    process = popen_launch_pd_server(
                        model,
                        url,
                        timeout=900,
                        other_args=args,
                        env=env,
                        return_stdout_stderr=(log, log),
                    )
                    stack.callback(
                        terminate_and_kill_process_tree, process, wait_timeout=60
                    )
                    wait_for_http_ready(url + "/health", timeout=900, process=process)
                    r = requests.get(url + "/get_server_info", timeout=30)
                    r.raise_for_status()
                    (output / f"{phase}-{role}-server-info.json").write_text(r.text)
                    self.assertEqual(
                        r.json()["enable_hierarchical_cache"], cached and role in roles
                    )
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
                    str(ports[2]),
                ]
                log = stack.enter_context((output / f"{phase}-router.log").open("w"))
                router = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                stack.callback(terminate_and_kill_process_tree, router, wait_timeout=60)
                wait_for_http_ready(router_url + "/health", timeout=120, process=router)
                yield

        def flush():
            for url in (p_url, d_url):
                r = requests.post(
                    url + "/flush_cache", params={"timeout": 30}, timeout=40
                )
                r.raise_for_status()

        def generate(name, ids, count):
            payload = dict(
                input_ids=ids,
                sampling_params=dict(
                    temperature=0, max_new_tokens=count, ignore_eos=True
                ),
                return_logprob=True,
            )
            r = requests.post(router_url + "/generate", json=payload, timeout=240)
            record = dict(request=payload, status_code=r.status_code, response=r.json())
            (output / f"{name}.json").write_text(json.dumps(record, indent=2))
            r.raise_for_status()
            result = r.json()
            self.assertEqual(len(result["output_ids"]), count)
            lps = result["meta_info"]["output_token_logprobs"]
            self.assertEqual([x[1] for x in lps], result["output_ids"])
            self.assertTrue(all(math.isfinite(x[0]) for x in lps))
            report["requests"] += 1
            save()
            return result

        def parity(name, expected, actual):
            self.assertEqual(actual["output_ids"], expected["output_ids"], name)
            delta = max(
                abs(a[0] - b[0])
                for a, b in zip(
                    actual["meta_info"]["output_token_logprobs"],
                    expected["meta_info"]["output_token_logprobs"],
                )
            )
            self.assertLessEqual(delta, 0.20, name)
            report["checks"].append(
                dict(
                    name=name,
                    max_logprob_delta=delta,
                    cache_details=actual["meta_info"].get("cached_tokens_details"),
                )
            )
            save()

        save()
        try:
            with pair("reference", False):
                prime_refs, follow_refs, follow_ids = [], [], []
                for i, ids in enumerate(primes):
                    flush()
                    ref = generate(f"reference-prime-{i}", ids, 16)
                    self.assertIn("739391", ref["text"])
                    prime_refs.append(ref)
                    follow_ids.append(ids + ref["output_ids"][:2])
                    flush()
                    follow_refs.append(
                        generate(f"reference-follow-{i}", follow_ids[-1], 14)
                    )
            with pair("writer", True):
                for i, ids in enumerate(primes):
                    flush()
                    result = generate(f"writer-prime-{i}", ids, 16)
                    parity(f"writer-prime-{i}", prime_refs[i], result)
                flush()
            inventory = [
                dict(
                    path=str(p.relative_to(storage)),
                    bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                )
                for p in storage.rglob("*.bin")
            ]
            (output / "stored-files.json").write_text(json.dumps(inventory, indent=2))
            for role in roles:
                self.assertTrue(
                    any(
                        x["path"].startswith(role + "/") and "mamba_tp" in x["path"]
                        for x in inventory
                    ),
                    f"No recurrent state stored by {role}",
                )
            with pair("fresh", True):
                for i, ids in enumerate(follow_ids):
                    flush()
                    result = generate(f"fresh-follow-{i}", ids, 14)
                    parity(f"fresh-follow-{i}", follow_refs[i], result)
                    if "prefill" in roles:
                        details = result["meta_info"].get("cached_tokens_details")
                        self.assertIsNotNone(details)
                        self.assertGreater(details["storage"], 0)
                        self.assertEqual(details["host"], 0)
                        if roles == {"prefill"}:
                            self.assertEqual(details["device"], 0)
                for role, url in (("prefill", p_url), ("decode", d_url)):
                    r = requests.get(url + "/metrics", timeout=30)
                    r.raise_for_status()
                    (output / f"fresh-{role}-metrics.prom").write_text(r.text)
            if "decode" in roles:
                matches = re.findall(
                    r"DCP L3 prefetch: tp_rank=(\d+) tokens=(\d+)",
                    (output / "fresh-decode.log").read_text(),
                )
                for rank in range(4):
                    self.assertEqual(
                        [int(n) for r, n in matches if int(r) == rank and int(n)],
                        [256, 512, 1024],
                    )
            report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            save()


if __name__ == "__main__":
    unittest.main()
