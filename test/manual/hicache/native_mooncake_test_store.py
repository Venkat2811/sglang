"""An independently owned native Mooncake store for bounded manual sweeps."""

import hashlib
import json
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from unittest.mock import patch

import requests

from sglang.test.test_utils import (
    find_available_port,
    terminate_and_kill_process_tree,
)


class NativeTestStore:
    def __init__(self, *, donor, rpc_port, http_port):
        self.donor = donor
        self.http_port = http_port
        self.config = dict(
            master_server_address=f"127.0.0.1:{rpc_port}",
            metadata_server="P2PHANDSHAKE",
            local_hostname="127.0.0.1",
            protocol="tcp",
            device_name="",
            global_segment_size=0,
        )

    def keys(self):
        response = requests.get(
            f"http://127.0.0.1:{self.http_port}/get_all_keys", timeout=30
        )
        response.raise_for_status()
        try:
            payload = response.json()
            keys = payload if isinstance(payload, list) else payload["keys"]
        except ValueError:
            keys = response.text.splitlines()
        return [entry if isinstance(entry, str) else entry["key"] for entry in keys]

    def inventory(self, keys=None):
        records = []
        for key in self.keys() if keys is None else keys:
            data = self.donor.get(key)
            assert data is not None and len(data) > 0, key
            records.append(
                dict(
                    path=key,
                    bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            )
        return records

    def remove(self, key):
        assert self.donor.remove(key, force=True) == 0, key

    def residency(self, keys):
        return {
            key: [
                dict(
                    status=str(d.status),
                    memory=d.is_memory_replica(),
                    local_disk=d.is_local_disk_replica(),
                )
                for d in self.donor.get_replica_desc(key)
            ]
            for key in keys
        }


def _offload_settings(storage):
    if storage is None:
        return {}, [], {}
    env = {
        "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(256 << 20),
        "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES": str(2 << 30),
        "MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS": "1",
        "MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT": "1",
        "MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES": str(16 << 20),
        "MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE": str(2 << 30),
    }
    args = [
        "--enable_offload=true",
        "--offload_on_evict=true",
        "--offload_force_evict=false",
        "--promotion_on_hit=false",
        "--default_kv_lease_ttl=500ms",
        "--eviction_ratio=0.25",
        "--eviction_high_watermark_ratio=0.5",
    ]
    return env, args, dict(enable_ssd_offload=True, ssd_offload_path=str(storage))


@contextmanager
def native_test_store(*, output, dram_bytes=1 << 30, ssd_storage=None):
    from mooncake.store import MooncakeDistributedStore

    rpc = find_available_port(32000)
    http = find_available_port(rpc + 1)
    offload_env, offload_args, setup_args = _offload_settings(ssd_storage)
    args = [
        "mooncake_master",
        f"--rpc_port={rpc}",
        f"--metrics_port={http}",
        "--enable_metric_reporting=false",
    ] + offload_args
    (output / "master-args.json").write_text(json.dumps(args))
    (output / "offload-environment.json").write_text(json.dumps(offload_env))
    donor = None
    with patch.dict(os.environ, offload_env), (output / "master.log").open("w") as log:
        master = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while True:
                assert master.poll() is None and time.monotonic() < deadline
                try:
                    with socket.create_connection(("127.0.0.1", rpc), timeout=1):
                        break
                except OSError:
                    time.sleep(0.1)
            donor = MooncakeDistributedStore()
            assert (
                donor.setup(
                    "127.0.0.1",
                    "P2PHANDSHAKE",
                    dram_bytes,
                    32 << 20,
                    "tcp",
                    "",
                    f"127.0.0.1:{rpc}",
                    **setup_args,
                )
                == 0
            )
            yield NativeTestStore(donor=donor, rpc_port=rpc, http_port=http)
        finally:
            if donor is not None:
                donor.close()
            terminate_and_kill_process_tree(master, wait_timeout=10)
