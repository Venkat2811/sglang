"""Bounded P4 -> D4/DCP4 Kimi continuation with role-local file/Mooncake L3.

PD_L3_ROLES=prefill|decode|both, PD_L3_OUTPUT_DIR and PD_L3_STORAGE_DIR
must select new owned directories. PD_L3_MODEL_PATH may select the pinned
local snapshot. A cache-off live P/D pair supplies matched controls; the
writer pair exits before the fresh-reader pair starts. Prompts end one token
before a DCP page so actual decode kernels publish the next checkpoint.
Ordinary role-local radix reuse supplies the matched continuation control:
decode-produced FP8 KV can differ from recomputed prefill KV even without L3.
PD_L3_BACKEND defaults to file; mooncake starts an independent native donor.
PD_L3_LIFECYCLE=1 adds three concurrent continuations, observed in-flight
restore cancellation, and post-promise state-file failure (file/both only).
"""

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

import requests
import torch
from native_mooncake_test_store import native_test_store
from pd_l3_lifecycle import LiveRestoreLifecycle, plugin_environment
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
        self._configure()
        self._save()
        try:
            self._build_cases()
            self._configure_ports()
            service = (
                native_test_store(output=self.output)
                if self.backend == "mooncake"
                else nullcontext(None)
            )
            with service as self.native:
                self._run_references()
                self._run_writer()
                self._run_fresh()
            self.report["status"] = "passed"
        except Exception as error:
            self.report.update(status="failed", error=repr(error))
            raise
        finally:
            self._save()

    def _configure(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 8)
        selection = os.environ["PD_L3_ROLES"]
        self.assertIn(selection, ("prefill", "decode", "both"))
        self.roles = {"prefill", "decode"} if selection == "both" else {selection}
        self.lifecycle = os.environ.get("PD_L3_LIFECYCLE", "0") == "1"
        self.backend = os.environ.get("PD_L3_BACKEND", "file")
        self.assertIn(self.backend, ("file", "mooncake"))
        if self.lifecycle:
            self.assertEqual((self.backend, selection), ("file", "both"))
        self.report_lock = threading.RLock()
        self.output = Path(os.environ["PD_L3_OUTPUT_DIR"])
        self.storage = Path(os.environ["PD_L3_STORAGE_DIR"])
        self.output.mkdir(parents=True, exist_ok=False)
        self.storage.mkdir(parents=True, exist_ok=False)
        self.model = os.environ.get("PD_L3_MODEL_PATH", MODEL)
        source = Path(__file__).read_bytes()
        (self.output / "runner.py").write_bytes(source)
        if self.lifecycle:
            for name in ("pd_l3_lifecycle.py", "l3_lifecycle_hooks.py"):
                (self.output / name).write_bytes(
                    Path(__file__).with_name(name).read_bytes()
                )
        self.report = dict(
            status="running",
            model=MODEL,
            revision=REVISION,
            roles=sorted(self.roles),
            backend=self.backend,
            lifecycle_enabled=self.lifecycle,
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

    def _build_cases(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model, revision=REVISION, trust_remote_code=True
        )
        # Preserve the chat framing and known-answer instruction while varying
        # only filler tokens. The marker never enters a request.
        template = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Remember the secret access code: 739391.\nINSERT_FILLER_HERE\nReply with the secret access code four times separated by single spaces, and nothing else.",
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
        self.primes = []
        for size in (255, 511, 1023):
            # Distinct first pages keep another case's L3 checkpoint from
            # warming this case's writer-control comparison.
            prefix = tokenizer.encode(
                before.replace("Remember", f"Record {size}. Remember"),
                add_special_tokens=False,
            )
            n = size - len(prefix) - len(suffix)
            self.assertGreater(n, 0)
            self.primes.append(prefix + (filler * (n // len(filler) + 1))[:n] + suffix)
        (self.output / "prime-inputs.json").write_text(json.dumps(self.primes))

    def _configure_ports(self):
        self.ports = []
        for _ in range(6):
            self.ports.append(find_available_port(max(self.ports, default=30999) + 1))
        self.p_url, self.d_url, self.router_url = [
            f"http://127.0.0.1:{p}" for p in self.ports[:3]
        ]
        self.common = [
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
            str(self.ports[3]),
        ]
        self.cache_args = [
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
            self.backend,
            "--hicache-storage-prefetch-policy",
            "wait_complete",
        ]

    def _storage_args(self, role):
        extra = dict(prefetch_threshold=1, enable_metadata_cache=False)
        if self.native is not None:
            extra.update(self.native.config, extra_backend_tag=f"pd-hybrid-{role}")
        return self.cache_args + [
            "--hicache-storage-backend-extra-config",
            json.dumps(extra),
        ]

    def _save(self):
        with self.report_lock:
            (self.output / "summary.json").write_text(json.dumps(self.report, indent=2))

    @contextmanager
    def _pair(self, phase, cached):
        with ExitStack() as stack:
            for role, url in (("prefill", self.p_url), ("decode", self.d_url)):
                args = self.common + [
                    "--disaggregation-mode",
                    role,
                    "--nccl-port",
                    str(self.ports[4 if role == "prefill" else 5]),
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
                    if role in self.roles:
                        args += ["--disaggregation-decode-enable-radix-cache"]
                if cached and role in self.roles:
                    args += self._storage_args(role)
                env = {
                    "SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1",
                    "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(self.storage / role),
                }
                if self.lifecycle and phase == "fresh" and role == "decode":
                    env.update(plugin_environment(self.output))
                (self.output / f"{phase}-{role}-args.json").write_text(json.dumps(args))
                log = stack.enter_context(
                    (self.output / f"{phase}-{role}.log").open("w")
                )
                process = popen_launch_pd_server(
                    self.model,
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
                (self.output / f"{phase}-{role}-server-info.json").write_text(r.text)
                self.assertEqual(
                    r.json()["enable_hierarchical_cache"], cached and role in self.roles
                )
            command = [
                sys.executable,
                "-m",
                "sglang_router.launch_router",
                "--pd-disaggregation",
                "--mini-lb",
                "--prefill",
                self.p_url,
                "--decode",
                self.d_url,
                "--host",
                "127.0.0.1",
                "--port",
                str(self.ports[2]),
            ]
            log = stack.enter_context((self.output / f"{phase}-router.log").open("w"))
            router = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            stack.callback(terminate_and_kill_process_tree, router, wait_timeout=60)
            wait_for_http_ready(
                self.router_url + "/health", timeout=120, process=router
            )
            yield

    def _flush(self, selected=("prefill", "decode")):
        for role, url in (("prefill", self.p_url), ("decode", self.d_url)):
            if role not in selected:
                continue
            r = requests.post(url + "/flush_cache", params={"timeout": 30}, timeout=40)
            r.raise_for_status()

    def _generate(self, name, ids, count):
        payload = dict(
            input_ids=ids,
            sampling_params=dict(temperature=0, max_new_tokens=count, ignore_eos=True),
            return_logprob=True,
        )
        started = time.monotonic()
        r = requests.post(self.router_url + "/generate", json=payload, timeout=240)
        record = dict(
            request=payload,
            status_code=r.status_code,
            response=r.json(),
            elapsed_seconds=time.monotonic() - started,
        )
        (self.output / f"{name}.json").write_text(json.dumps(record, indent=2))
        r.raise_for_status()
        result = r.json()
        self.assertEqual(len(result["output_ids"]), count)
        if "follow" in name:
            self.assertIn("739391", result["text"])
        lps = result["meta_info"]["output_token_logprobs"]
        self.assertEqual([x[1] for x in lps], result["output_ids"])
        self.assertTrue(all(math.isfinite(x[0]) for x in lps))
        with self.report_lock:
            self.report["requests"] += 1
            self._save()
        return result

    def _parity(self, name, expected, actual):
        self.assertEqual(actual["output_ids"], expected["output_ids"], name)
        delta = max(
            abs(a[0] - b[0])
            for a, b in zip(
                actual["meta_info"]["output_token_logprobs"],
                expected["meta_info"]["output_token_logprobs"],
            )
        )
        self.assertLessEqual(delta, 0.20, name)
        self.report["checks"].append(
            dict(
                name=name,
                max_logprob_delta=delta,
                cache_details=actual["meta_info"].get("cached_tokens_details"),
            )
        )
        self._save()

    def _run_references(self):
        with self._pair("reference", False):
            (
                self.prime_refs,
                self.follow_refs,
                self.cold_follow_refs,
                self.follow_ids,
            ) = [], [], [], []
            for i, ids in enumerate(self.primes):
                self._flush()
                ref = self._generate(f"reference-prime-{i}", ids, 16)
                self.assertEqual(ref["text"].strip(), " ".join(["739391"] * 4))
                self.prime_refs.append(ref)
                self.follow_ids.append(ids + ref["output_ids"][:2])
                # Retain ordinary radix only on the roles whose persisted
                # cache is under test. This preserves the same producer
                # kernels and prefix boundary as the future L3 reader.
                self._flush({"prefill", "decode"} - self.roles)
                self.follow_refs.append(
                    self._generate(
                        f"reference-warm-follow-{i}", self.follow_ids[-1], 14
                    )
                )
                if "decode" in self.roles:
                    self.assertEqual(
                        self.follow_refs[-1]["meta_info"]["cached_tokens"],
                        (256, 512, 1024)[i],
                    )
                self._flush()
                self.cold_follow_refs.append(
                    self._generate(f"reference-follow-{i}", self.follow_ids[-1], 14)
                )

    def _run_writer(self):
        with self._pair("writer", True):
            for i, ids in enumerate(self.primes):
                self._flush()
                result = self._generate(f"writer-prime-{i}", ids, 16)
                self._parity(f"writer-prime-{i}", self.prime_refs[i], result)
            self._flush()
        inventory = self._inventory()
        (self.output / "stored-objects.json").write_text(
            json.dumps(inventory, indent=2)
        )
        for role in self.roles:
            role_prefix = role + "/" if self.backend == "file" else f"pd-hybrid-{role}_"
            self.assertTrue(
                any(
                    x["path"].startswith(role_prefix) and self._is_state(x["path"])
                    for x in inventory
                ),
                f"No recurrent state stored by {role}",
            )

    def _inventory(self):
        if self.native is not None:
            return self.native.inventory()
        return [
            dict(
                path=str(p.relative_to(self.storage)),
                bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
            for p in self.storage.rglob("*.bin")
        ]

    def _is_state(self, key, rank=None):
        if self.backend == "file":
            return ("mamba_tp" if rank is None else f"mamba_tp{rank}_") in key
        rank_pattern = r"\d+" if rank is None else str(rank)
        return (
            re.search(
                rf"_{rank_pattern}_(?:mamba_v1_[a-f0-9]+_)?(?:temporal|conv_\d+)$",
                key,
            )
            is not None
        )

    def _remove_state_shard(self):
        victims = [x for x in self._inventory() if self._is_state(x["path"], rank=3)]
        self.assertTrue(victims)
        self.report["removed_state_objects"] = victims
        self._save()
        for record in victims:
            if self.native is None:
                (self.storage / record["path"]).unlink()
            else:
                self.native.remove(record["path"])

    def _run_fresh(self):
        with self._pair("fresh", True):
            for i, ids in enumerate(self.follow_ids):
                self._flush()
                result = self._generate(f"fresh-follow-{i}", ids, 14)
                self._parity(f"fresh-follow-{i}", self.follow_refs[i], result)
                self.report["checks"][-1]["cold_prefill_logprob_delta"] = max(
                    abs(a[0] - b[0])
                    for a, b in zip(
                        result["meta_info"]["output_token_logprobs"],
                        self.cold_follow_refs[i]["meta_info"]["output_token_logprobs"],
                    )
                )
                self._save()
                if "prefill" in self.roles:
                    details = result["meta_info"].get("cached_tokens_details")
                    self.assertIsNotNone(details)
                    self.assertGreater(details["storage"], 0)
                    self.assertEqual(details["host"], 0)
                    if self.roles == {"prefill"}:
                        self.assertEqual(details["device"], 0)
            self._check_fresh_rank_logs()
            if self.lifecycle:
                LiveRestoreLifecycle(self).run()
            else:
                self._flush()
                # One TP shard's state hole must force a common miss even
                # while all MLA KV shards remain present.
                self._remove_state_shard()
                fallback = self._generate(
                    "missing-state-follow-1", self.follow_ids[1], 14
                )
                self._parity(
                    "missing-state-follow-1", self.cold_follow_refs[1], fallback
                )
                self.assertEqual(fallback["meta_info"]["cached_tokens"], 0)
            for role, url in (("prefill", self.p_url), ("decode", self.d_url)):
                r = requests.get(url + "/metrics", timeout=30)
                r.raise_for_status()
                (self.output / f"fresh-{role}-metrics.prom").write_text(r.text)

    def _check_fresh_rank_logs(self):
        if "decode" in self.roles:
            matches = re.findall(
                r"DCP L3 prefetch: tp_rank=(\d+) tokens=(\d+)",
                (self.output / "fresh-decode.log").read_text(),
            )
            for rank in range(4):
                self.assertEqual(
                    [int(n) for r, n in matches if int(r) == rank and int(n)],
                    [256, 512, 1024],
                )


if __name__ == "__main__":
    unittest.main()
