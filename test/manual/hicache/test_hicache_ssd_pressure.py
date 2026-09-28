"""File/Mooncake SSD inference under bounded cache pressure.

Required environment: HICACHE_SSD_BACKEND=file|mooncake,
HICACHE_SSD_OUTPUT_DIR (new directory), HICACHE_SSD_STORAGE_ROOT (new directory
on a verified SSD filesystem), HICACHE_SSD_BLOCK_DEVICE (cgroup io.stat id).
The supplied device must be independently mapped to the storage mount. Only
owned files are fsynced/advised; no global page-cache flush or device changes.
HICACHE_SSD_TP and HICACHE_SSD_DCP default to 1. Distributed runs also require
all-rank prefix agreement and inject a hole in the last DCP rank's second page.
"""

import gc
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
from contextlib import contextmanager
from pathlib import Path

import requests
import torch
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
PAGE = 64
PREFIX = 1536
NEW_TOKENS = 6
LOGPROB_ATOL = 0.05
STORE_DRAM = 128 << 20


class TestHiCacheSsdPressure(CustomTestCase):
    """Evicted KV must return from disk without changing model continuation."""

    def test_pressure_restore(self):
        backend = os.environ["HICACHE_SSD_BACKEND"]
        self.assertIn(backend, ("file", "mooncake"))
        tp = int(os.environ.get("HICACHE_SSD_TP", "1"))
        dcp = int(os.environ.get("HICACHE_SSD_DCP", "1"))
        self.assertGreaterEqual(dcp, 1)
        self.assertGreaterEqual(tp, dcp)
        self.assertEqual(tp % dcp, 0)
        self.assertGreaterEqual(torch.cuda.device_count(), tp)
        logical_page = PAGE * dcp
        self.assertEqual(PREFIX % logical_page, 0)
        self.assertGreaterEqual(PREFIX // logical_page, 3)
        # max-total-tokens caps physical rows per rank. Leave space for the
        # rounded prompt and allocator's extra logical-page decode reserve.
        # Six distinct prompts still exceed both L1 and ratio-1.5 L2.
        device_token_cap = max(2048, PREFIX + 3 * logical_page) // dcp
        self.assertEqual(device_token_cap % PAGE, 0)
        output = Path(os.environ["HICACHE_SSD_OUTPUT_DIR"])
        storage = Path(os.environ["HICACHE_SSD_STORAGE_ROOT"])
        device = os.environ["HICACHE_SSD_BLOCK_DEVICE"]
        output.mkdir(parents=True, exist_ok=False)
        storage.mkdir(parents=True, exist_ok=False)
        source = Path(__file__).read_bytes()
        (output / "runner.py").write_bytes(source)
        filesystem = subprocess.check_output(
            ["findmnt", "-T", str(storage), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
            text=True,
        )
        self.assertNotIn("tmpfs", filesystem)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        config = AutoConfig.from_pretrained(MODEL, revision=REVISION)
        object_bytes = (
            PAGE
            * config.num_hidden_layers
            * (config.kv_lora_rank + config.qk_rope_head_dim)
            * 2
        )
        # Equivalent MLA ranks read the same stored shard independently. Size
        # SSD staging for every simultaneous reader, not just unique objects.
        # This does not enlarge the persistent 128 MiB DRAM cache segment.
        restore_bytes = PREFIX // PAGE * object_bytes * (tp // dcp)
        ssd_staging_bytes = max(64 << 20, 1 << (restore_bytes - 1).bit_length())
        cases = []
        for i in range(6):
            prefix = tokenizer.encode(f"Record {i}: The access code is {7300 + i}. ")
            suffix = tokenizer.encode("\nThe access code is", add_special_tokens=False)
            filler = tokenizer.encode(
                "The library keeps records of books. ", add_special_tokens=False
            )
            n = PREFIX + 1 - len(prefix) - len(suffix)
            ids = prefix + (filler * (n // len(filler) + 1))[:n] + suffix
            self.assertEqual(len(ids), PREFIX + 1)
            cases.append({"ids": ids, "answer": f"{7300 + i}."})
        self.assertEqual(len({tuple(c["ids"][:PAGE]) for c in cases}), len(cases))
        (output / "cases.json").write_text(json.dumps(cases))
        report = {
            "status": "running",
            "backend": backend,
            "model": MODEL,
            "revision": REVISION,
            "tp": tp,
            "dcp": dcp,
            "logical_page_size": logical_page,
            "filesystem": filesystem,
            "block_device": device,
            "runner_sha256": hashlib.sha256(source).hexdigest(),
            "device_token_cap": device_token_cap,
            "logical_device_token_cap": device_token_cap * dcp,
            "host_to_device_ratio": 1.5,
            "mooncake_dram_bytes": STORE_DRAM if backend == "mooncake" else None,
            "mooncake_ssd_staging_bytes": (
                ssd_staging_bytes if backend == "mooncake" else None
            ),
            "object_bytes": object_bytes,
            "logprob_atol": LOGPROB_ATOL,
            "phases": {},
            "residency": {},
            "io": {},
        }

        def save():
            (output / "summary.json").write_text(json.dumps(report, indent=2))

        def wait_for(predicate, message, timeout=90):
            deadline = time.monotonic() + timeout
            while not predicate():
                self.assertLess(time.monotonic(), deadline, message)
                time.sleep(1)

        def io_stat():
            text = Path("/sys/fs/cgroup/io.stat").read_text()
            for line in text.splitlines():
                fields = line.split()
                if fields[0] == device:
                    return {k: int(v) for k, v in (x.split("=") for x in fields[1:])}
            return {"rbytes": 0, "wbytes": 0}

        def discard_owned_file_cache():
            files = []
            for path in storage.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    with path.open("rb") as f:
                        os.fsync(f.fileno())
                        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    files.append(
                        {
                            "path": str(path.relative_to(storage)),
                            "size": path.stat().st_size,
                        }
                    )
            self.assertTrue(files, "No SSD files were written")
            (output / "disk-files.json").write_text(json.dumps(files, indent=2))

        port = find_available_port(31000)
        url = f"http://127.0.0.1:{port}"
        common = [
            "--revision",
            REVISION,
            "--trust-remote-code",
            "--tp-size",
            str(tp),
            "--dcp-size",
            str(dcp),
            "--dcp-comm-backend",
            "ag_rs",
            "--attention-backend",
            "triton" if dcp == 1 else "flashinfer",
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "auto",
            "--page-size",
            str(PAGE),
            "--context-length",
            "2048",
            "--chunked-prefill-size",
            "1024",
            "--max-total-tokens",
            str(device_token_cap),
            "--max-running-requests",
            "1",
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
        if dcp == 1:
            common.append("--enable-deterministic-inference")
        env = {
            **os.environ,
            "SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE": str(PAGE),
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(storage),
        }

        @contextmanager
        def server(phase, extra):
            (output / f"{phase}-args.json").write_text(json.dumps(common + extra))
            with (output / f"{phase}-server.log").open("w") as log:
                process = None
                try:
                    process = popen_launch_server(
                        MODEL,
                        url,
                        timeout=900,
                        other_args=common + extra,
                        env=env,
                        return_stdout_stderr=(log, log),
                    )
                    response = requests.get(url + "/get_server_info", timeout=30)
                    response.raise_for_status()
                    (output / f"{phase}-server-info.json").write_text(response.text)
                    yield
                finally:
                    if process is not None:
                        terminate_and_kill_process_tree(process)

        def generate(phase, index):
            case = cases[index]
            payload = {
                "input_ids": case["ids"],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": NEW_TOKENS,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": PREFIX,
            }
            record = {"case": index, "request": payload}
            try:
                response = requests.post(url + "/generate", json=payload, timeout=120)
                record.update(
                    status_code=response.status_code, response=response.json()
                )
                response.raise_for_status()
                result = record["response"]
                self.assertEqual(result["text"].strip(), case["answer"])
                report["phases"].setdefault(phase, []).append(result)
                save()
                return result
            except Exception as error:
                record["error"] = repr(error)
                raise
            finally:
                (output / f"{phase}-{index}.json").write_text(
                    json.dumps(record, indent=2)
                )

        def parity(reference, result):
            self.assertEqual(result["output_ids"], reference["output_ids"])
            self.assertEqual(len(result["output_ids"]), NEW_TOKENS)
            a, b = (
                result["meta_info"]["output_token_logprobs"],
                reference["meta_info"]["output_token_logprobs"],
            )
            self.assertEqual(len(a), NEW_TOKENS)
            self.assertEqual(len(b), NEW_TOKENS)
            for got, want in zip(a, b):
                self.assertEqual(got[1], want[1])
                self.assertTrue(math.isfinite(got[0]) and math.isfinite(want[0]))
                self.assertLessEqual(abs(got[0] - want[0]), LOGPROB_ATOL)

        def metrics(name):
            response = requests.get(url + "/metrics", timeout=30)
            response.raise_for_status()
            (output / f"{name}.prom").write_text(response.text)

        tag = "ssd-pressure-smoke"
        hashes = get_storage_hash_str(
            cases[0]["ids"][:PREFIX], None, page_size=logical_page
        )
        model_name = MODEL.replace("/", "-")
        keys = []
        for h in hashes:
            for rank in range(dcp):
                file_suffix = f"_{model_name}"
                native_prefix = f"{tag}_{model_name}"
                native_rank = ""
                if dcp > 1:
                    file_suffix += (
                        f"_tp{tp}_dcp{rank}_{dcp}_page{logical_page}"
                        "_bfloat16_page_first"
                    )
                    native_prefix += (
                        f"_dcp_v1_tp{tp}_ds{dcp}_pp1_cp0of1"
                        f"_page{logical_page}_bfloat16_page_first_pp0"
                    )
                    native_rank = f"dcp{rank}"
                keys.append(
                    f"{h}{file_suffix}.bin"
                    if backend == "file"
                    else f"{native_prefix}_{h}_{native_rank}_k"
                )
        report["target_keys"] = keys.copy()
        master = donor = None

        def exists(key):
            return (
                (storage / key).is_file()
                if backend == "file"
                else donor.is_exist(key) == 1
            )

        def read_hashes():
            result = {}
            for key in keys:
                value = (
                    (storage / key).read_bytes()
                    if backend == "file"
                    else donor.get(key)
                )
                self.assertEqual(len(value), object_bytes, key)
                result[key] = hashlib.sha256(value).hexdigest()
            return result

        def residency(label):
            if backend == "file":
                return True
            snapshot = {
                key: [
                    {
                        "status": str(d.status),
                        "memory": d.is_memory_replica(),
                        "local_disk": d.is_local_disk_replica(),
                    }
                    for d in donor.get_replica_desc(key)
                ]
                for key in keys
            }
            report["residency"][label] = snapshot
            save()
            return all(
                ds
                and all(not d["memory"] for d in ds)
                and any(
                    d["local_disk"] and d["status"].endswith("COMPLETE") for d in ds
                )
                for ds in snapshot.values()
            )

        def restore(phase, expected):
            self.assertTrue(
                residency(phase + "-before"), "Target still has a memory replica"
            )
            discard_owned_file_cache()
            metrics(phase + "-before")
            before = io_stat()
            result = generate(phase, 0)
            after = io_stat()
            report["io"][phase] = {
                "before": before,
                "after": after,
                "read_bytes": after["rbytes"] - before["rbytes"],
            }
            save()
            parity(reference[0], result)
            details = result["meta_info"]["cached_tokens_details"]
            self.assertIsInstance(
                details, dict, f"{phase}: no cache tier hit; inspect prefetch logs"
            )
            self.assertEqual(details["storage"], expected)
            self.assertEqual(details["host"], 0)
            self.assertEqual(details["device"], 0)
            self.assertGreaterEqual(
                report["io"][phase]["read_bytes"], expected // PAGE * object_bytes
            )
            metrics(phase + "-after")
            if dcp > 1:
                log_phase = "writer" if phase == "same-engine-replay" else phase
                matches = re.findall(
                    r"DCP L3 prefetch: tp_rank=(\d+) tokens=(\d+)",
                    (output / f"{log_phase}-server.log").read_text(),
                )
                by_rank = {
                    rank: Counter(
                        int(n) for r, n in matches if int(r) == rank and int(n)
                    )
                    for rank in range(tp)
                }
                report.setdefault("rank_prefixes", {})[phase] = by_rank
                save()
                for rank, counts in by_rank.items():
                    self.assertEqual(counts, Counter([expected]), rank)

        try:
            save()
            with server("reference", ["--disable-radix-cache"]):
                reference = [generate("reference", i) for i in range(len(cases))]
                self.assertTrue(
                    all(r["meta_info"]["cached_tokens"] == 0 for r in reference)
                )
            with server("cold-repeat", ["--disable-radix-cache"]):
                for i in range(len(cases)):
                    result = generate("cold-repeat", i)
                    self.assertEqual(result["meta_info"]["cached_tokens"], 0)
                    parity(reference[i], result)

            storage_config = {"prefetch_threshold": 1}
            if backend == "mooncake":
                from mooncake.store import MooncakeDistributedStore

                # These buffers are staging, separate from the bounded cache segment.
                os.environ["MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES"] = str(
                    ssd_staging_bytes
                )
                os.environ["MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES"] = str(1 << 30)
                os.environ["MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS"] = "1"
                # Default buckets need 256 MiB or 500 keys before flushing,
                # larger than this deliberately tiny DRAM store. Flush each
                # object so pressure can make progress without a full bucket.
                os.environ["MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT"] = "1"
                os.environ["MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES"] = str(8 << 20)
                os.environ["MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE"] = str(1 << 30)
                report["offload_environment"] = {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith("MOONCAKE_OFFLOAD_")
                }
                master_port = find_available_port(50051)
                address = f"127.0.0.1:{master_port}"
                args = [
                    "mooncake_master",
                    f"--rpc_port={master_port}",
                    "--enable_offload=true",
                    "--offload_on_evict=true",
                    "--offload_force_evict=false",
                    "--promotion_on_hit=false",
                    "--default_kv_lease_ttl=500ms",
                    "--eviction_ratio=0.25",
                    "--eviction_high_watermark_ratio=0.5",
                    "--enable_metric_reporting=false",
                ]
                (output / "master-args.json").write_text(json.dumps(args))
                with (output / "master.log").open("w") as log:
                    master = subprocess.Popen(
                        args, stdout=log, stderr=subprocess.STDOUT
                    )

                def ready():
                    self.assertIsNone(master.poll())
                    try:
                        with socket.create_connection(
                            ("127.0.0.1", master_port), timeout=1
                        ):
                            return True
                    except OSError:
                        return False

                wait_for(ready, "Master startup timed out", 30)
                donor = MooncakeDistributedStore()
                self.assertEqual(
                    donor.setup(
                        "127.0.0.1",
                        "P2PHANDSHAKE",
                        STORE_DRAM,
                        16 << 20,
                        "tcp",
                        "",
                        address,
                        enable_ssd_offload=True,
                        ssd_offload_path=str(storage),
                    ),
                    0,
                )
                storage_config.update(
                    master_server_address=address,
                    metadata_server="P2PHANDSHAKE",
                    local_hostname="127.0.0.1",
                    protocol="tcp",
                    device_name="",
                    global_segment_size=0,
                    extra_backend_tag=tag,
                )

            extra = [
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
                backend,
                "--hicache-storage-prefetch-policy",
                "wait_complete",
                "--hicache-storage-backend-extra-config",
                json.dumps(storage_config),
            ]
            with server("writer", extra):
                for i in range(len(cases)):
                    result = generate("writer", i)
                    self.assertEqual(result["meta_info"]["cached_tokens"], 0)
                    parity(reference[i], result)
                    if i == 0:
                        wait_for(
                            lambda: all(exists(k) for k in keys),
                            "Target publication timed out",
                        )
                        original = read_hashes()
                        report["target_sha256"] = original
                        residency("initial-target")
                        save()
                    if backend == "mooncake":
                        # Allow the asynchronous SSD heartbeat to finish a
                        # spill before admitting the next ~46 MiB KV batch.
                        time.sleep(2)
                metrics("after-working-set")
                # The inference working set exceeds both configured cache capacities.
                # Native fillers bound any additional pressure needed to move every
                # target page to SSD; never explicitly delete a memory replica.
                if backend == "mooncake":
                    time.sleep(2)
                    for i in range(33):
                        if residency("after-pressure"):
                            break
                        self.assertLess(
                            i, 32, "Target did not become SSD-only under pressure"
                        )
                        self.assertEqual(
                            donor.put(f"{tag}-pressure-{i}", bytes([i]) * (8 << 20)), 0
                        )
                        report["native_pressure_bytes"] = (i + 1) * (8 << 20)
                        time.sleep(2)
                self.assertEqual(read_hashes(), original)
                restore("same-engine-replay", PREFIX)
            self.assertTrue(all(exists(k) for k in keys))
            with server("fresh-reader", extra):
                restore("fresh-reader", PREFIX)

            # Remove a middle page after both positive disk restores. A fresh
            # process must stop before the hole and recompute the remainder.
            removed = keys[2 * dcp - 1]
            report["missing_shard"] = {
                "key": removed,
                "dcp_rank": dcp - 1,
                "expected_prefix": logical_page,
            }
            if backend == "file":
                (storage / removed).unlink()
            else:
                self.assertEqual(donor.remove(removed, force=True), 0)
            self.assertFalse(exists(removed))
            self.assertTrue(all(exists(k) for k in keys if k != removed))
            # The remaining target prefix is one page; use it for residency checks.
            keys = keys[:dcp]
            with server("missing-page-reader", extra):
                restore("missing-page-reader", logical_page)
            report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            save()
            if donor is not None:
                donor.close()
                donor = None
                gc.collect()
            if master is not None:
                master.terminate()
                try:
                    master.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    master.kill()
                    master.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
