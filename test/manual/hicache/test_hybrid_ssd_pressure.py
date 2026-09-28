"""Bounded Kimi MLA/KDA SSD correctness through file or native Mooncake.

HYBRID_SSD_BACKEND, HYBRID_SSD_OUTPUT_DIR, HYBRID_SSD_STORAGE_DIR and
HYBRID_SSD_BLOCK_DEVICE are required. Independently verify the storage mount
and cgroup io.stat device first. HYBRID_SSD_MODEL_PATH may use the pinned local
snapshot. Only owned files are fsynced/advised; no global page-cache drop.
"""

import hashlib
import json
import os
import re
import subprocess
import time
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path

import requests
import torch
from hybrid_l3_workload import MODEL, REVISION, generate, known_code_cases, parity
from native_mooncake_test_store import native_test_store

from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

PREFIX = 1536


class TestHybridSsdPressure(CustomTestCase):
    def test_disk_continuation(self):
        self._configure()
        try:
            self._references()
            service = (
                native_test_store(
                    output=self.output,
                    dram_bytes=128 << 20,
                    ssd_storage=self.storage,
                )
                if self.backend == "mooncake"
                else nullcontext(None)
            )
            with service as self.native:
                self._writer()
                self._fresh_reader()
            self.report["status"] = "passed"
        except Exception as error:
            self.report.update(status="failed", error=repr(error))
            raise
        finally:
            self._save()

    def _configure(self):
        self.tp = int(os.environ.get("HYBRID_SSD_TP", "4"))
        self.assertIn(self.tp, (4, 8))
        self.assertGreaterEqual(torch.cuda.device_count(), self.tp)
        self.logical_page = 64 * self.tp
        logical_cap = max(2560, PREFIX + 3 * self.logical_page)
        self.physical_cap = logical_cap // self.tp
        self.backend = os.environ["HYBRID_SSD_BACKEND"]
        self.assertIn(self.backend, ("file", "mooncake"))
        self.output = Path(os.environ["HYBRID_SSD_OUTPUT_DIR"])
        self.storage = Path(os.environ["HYBRID_SSD_STORAGE_DIR"])
        self.device = os.environ["HYBRID_SSD_BLOCK_DEVICE"]
        self.model = os.environ.get("HYBRID_SSD_MODEL_PATH", MODEL)
        self.output.mkdir(parents=True, exist_ok=False)
        self.storage.mkdir(parents=True, exist_ok=False)
        for name in (
            Path(__file__).name,
            "hybrid_l3_workload.py",
            "native_mooncake_test_store.py",
        ):
            (self.output / name).write_bytes(
                Path(__file__).with_name(name).read_bytes()
            )
        filesystem = subprocess.check_output(
            ["findmnt", "-T", str(self.storage), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
            text=True,
        )
        self.assertNotIn("tmpfs", filesystem)
        self.report = dict(
            status="running",
            backend=self.backend,
            model=MODEL,
            revision=REVISION,
            tp=self.tp,
            dcp=self.tp,
            physical_device_token_cap=self.physical_cap,
            logical_device_token_cap=logical_cap,
            host_to_device_ratio=1.5,
            store_dram_bytes=128 << 20,
            filesystem=filesystem,
            block_device=self.device,
            requests=0,
            checks=[],
            io={},
            residency={},
            logprob_atol=0.20,
        )
        self.native = None
        self.url = f"http://127.0.0.1:{find_available_port(31000)}"
        self.cases = known_code_cases(model=self.model, sizes=[PREFIX + 1] * 6)
        self.hashes = get_storage_hash_str(
            self.cases[0][:PREFIX], None, page_size=self.logical_page
        )
        (self.output / "cases.json").write_text(json.dumps(self.cases))
        self._save()

    def _save(self):
        (self.output / "summary.json").write_text(json.dumps(self.report, indent=2))

    def _arguments(self, cached):
        args = [
            "--revision",
            REVISION,
            "--served-model-name",
            MODEL,
            "--trust-remote-code",
            "--tp-size",
            str(self.tp),
            "--dcp-size",
            str(self.tp),
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
            "--max-total-tokens",
            str(self.physical_cap),
            "--max-running-requests",
            "1",
            "--max-mamba-cache-size",
            "16",
            "--mem-fraction-static",
            "0.4",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--enable-metrics",
            "--enable-cache-report",
            "--log-level",
            "debug",
        ]
        if cached:
            config = dict(prefetch_threshold=1, enable_metadata_cache=False)
            if self.native is None:
                config.update(max_size="1Gi", min_free_space=0)
            else:
                config.update(self.native.config, extra_backend_tag="hybrid-ssd")
            args += [
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "1.5",
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
                "--hicache-storage-backend-extra-config",
                json.dumps(config),
            ]
        return args

    @contextmanager
    def _server(self, phase, cached):
        args = self._arguments(cached)
        (self.output / f"{phase}-args.json").write_text(json.dumps(args))
        process = None
        with (self.output / f"{phase}-server.log").open("w") as log:
            try:
                process = popen_launch_server(
                    self.model,
                    self.url,
                    timeout=900,
                    other_args=args,
                    env={
                        "SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1",
                        "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(self.storage),
                    },
                    return_stdout_stderr=(log, log),
                )
                r = requests.get(self.url + "/get_server_info", timeout=30)
                r.raise_for_status()
                (self.output / f"{phase}-server-info.json").write_text(r.text)
                yield
            finally:
                if process is not None:
                    terminate_and_kill_process_tree(process, wait_timeout=60)

    def _flush(self):
        r = requests.post(self.url + "/flush_cache", params={"timeout": 30}, timeout=40)
        r.raise_for_status()

    def _generate(self, name, ids, count):
        result = generate(
            url=self.url, ids=ids, count=count, output=self.output, name=name
        )
        self.report["requests"] += 1
        self._save()
        return result

    def _check(self, name, reference, actual):
        delta = parity(reference=reference, actual=actual)
        self.report["checks"].append(dict(name=name, max_logprob_delta=delta))
        self._save()

    def _references(self):
        self.prime_refs, self.warm_refs, self.cold_refs, self.follows = [], [], [], []
        with self._server("reference", False):
            for i, ids in enumerate(self.cases):
                self._flush()
                prime = self._generate(f"reference-prime-{i}", ids, 16)
                self.assertEqual(prime["text"].strip(), " ".join(["739391"] * 4))
                self.prime_refs.append(prime)
                follow = ids + prime["output_ids"][:2]
                self.follows.append(follow)
                warm = self._generate(f"reference-warm-{i}", follow, 14)
                self.assertEqual(warm["meta_info"]["cached_tokens"], PREFIX)
                self.warm_refs.append(warm)
                self._flush()
                self.cold_refs.append(self._generate(f"reference-cold-{i}", follow, 14))

    def _keys(self):
        if self.native is not None:
            return self.native.keys()
        return [str(p.relative_to(self.storage)) for p in self.storage.glob("*.bin")]

    def _state_rank(self, key):
        pattern = r"mamba_tp(\d+)_" if self.native is None else r"_(\d+)_mamba_v1_"
        match = re.search(pattern, key)
        return int(match[1]) if match is not None else None

    def _target_keys(self):
        selected = []
        for key in self._keys():
            if not any(h in key for h in self.hashes):
                continue
            if self._state_rank(key) is None or self.hashes[-1] in key:
                selected.append(key)
        kv = [k for k in selected if self._state_rank(k) is None]
        ranks = {self._state_rank(k) for k in selected} - {None}
        if len(kv) == PREFIX // 64 and ranks == set(range(self.tp)):
            return selected
        return []

    def _inventory(self, keys):
        keys = sorted(keys)
        if self.native is not None:
            return self.native.inventory(keys)
        return [
            dict(
                path=k,
                bytes=(self.storage / k).stat().st_size,
                sha256=hashlib.sha256((self.storage / k).read_bytes()).hexdigest(),
            )
            for k in keys
        ]

    def _wait_for_disk(self, phase):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            keys = self._target_keys()
            if keys:
                if self.native is None:
                    return keys
                snapshot = self.native.residency(keys)
                self.report["residency"][phase] = snapshot
                self._save()
                if all(
                    ds
                    and all(not d["memory"] for d in ds)
                    and any(
                        d["local_disk"] and d["status"].endswith("COMPLETE") for d in ds
                    )
                    for ds in snapshot.values()
                ):
                    return keys
            time.sleep(1)
        self.fail(f"{phase}: complete disk-only target KV/state not observed")

    def _discard_file_cache(self):
        files = []
        for p in self.storage.rglob("*"):
            if p.is_file() and not p.is_symlink():
                with p.open("rb") as f:
                    os.fsync(f.fileno())
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                files.append(
                    dict(path=str(p.relative_to(self.storage)), bytes=p.stat().st_size)
                )
        self.assertTrue(files)
        (self.output / "disk-files.json").write_text(json.dumps(files))

    def _read_bytes(self):
        for line in Path("/sys/fs/cgroup/io.stat").read_text().splitlines():
            fields = line.split()
            if fields[0] == self.device:
                return int(dict(x.split("=") for x in fields[1:])["rbytes"])
        return 0

    def _restore(self, phase):
        keys = self._wait_for_disk(phase)
        inventory = self._inventory(keys)
        self.assertEqual(inventory, self.target_inventory)
        self._discard_file_cache()
        before = self._read_bytes()
        result = self._generate(phase, self.follows[0], 14)
        read_bytes = self._read_bytes() - before
        self.report["io"][phase] = dict(
            read_bytes=read_bytes,
            expected_payload_bytes=sum(x["bytes"] for x in inventory),
        )
        self._save()
        self._check(phase, self.warm_refs[0], result)
        details = result["meta_info"]["cached_tokens_details"]
        self.assertEqual(
            {k: details[k] for k in ("storage", "host", "device")},
            dict(storage=PREFIX, host=0, device=0),
        )
        self.assertGreaterEqual(read_bytes, sum(x["bytes"] for x in inventory))

    def _writer(self):
        with self._server("writer", True):
            for i, ids in enumerate(self.cases):
                result = self._generate(f"writer-{i}", ids, 16)
                self._check(f"writer-{i}", self.prime_refs[i], result)
            keys = self._wait_for_disk("writer-pressure")
            self.target_inventory = self._inventory(keys)
            (self.output / "target-objects.json").write_text(
                json.dumps(self.target_inventory, indent=2)
            )
            # No flush before this replay: capacity pressure must evict L1/L2.
            self._restore("same-engine-pressure")
            self._flush()
        self._check_rank_log("writer")

    def _fresh_reader(self):
        with self._server("fresh", True):
            self._restore("fresh-reader")
            self._flush()
            victims = [
                k
                for k in self._keys()
                if self._state_rank(k) == self.tp - 1
                and any(h in k for h in self.hashes)
            ]
            self.assertTrue(victims)
            self.report["removed_state_objects"] = self._inventory(victims)
            self._save()
            for key in victims:
                if self.native is None:
                    (self.storage / key).unlink()
                else:
                    self.native.remove(key)
            fallback = self._generate("missing-state", self.follows[0], 14)
            self._check("missing-state", self.cold_refs[0], fallback)
            self.assertEqual(fallback["meta_info"]["cached_tokens"], 0)
        self._check_rank_log("fresh")

    def _check_rank_log(self, phase):
        matches = re.findall(
            r"DCP L3 prefetch: tp_rank=(\d+) tokens=(\d+)",
            (self.output / f"{phase}-server.log").read_text(),
        )
        for rank in range(self.tp):
            self.assertEqual(
                [int(n) for r, n in matches if int(r) == rank and int(n)], [PREFIX]
            )


if __name__ == "__main__":
    unittest.main()
