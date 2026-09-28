"""An independently owned native Mooncake store for bounded manual sweeps."""

import hashlib
import json
import socket
import subprocess
import time
from contextlib import contextmanager

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

    def inventory(self):
        response = requests.get(
            f"http://127.0.0.1:{self.http_port}/get_all_keys", timeout=30
        )
        response.raise_for_status()
        try:
            payload = response.json()
            keys = payload if isinstance(payload, list) else payload["keys"]
        except ValueError:
            keys = response.text.splitlines()
        records = []
        for entry in keys:
            key = entry if isinstance(entry, str) else entry["key"]
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


@contextmanager
def native_test_store(*, output, dram_bytes=1 << 30):
    from mooncake.store import MooncakeDistributedStore

    rpc = find_available_port(32000)
    http = find_available_port(rpc + 1)
    args = [
        "mooncake_master",
        f"--rpc_port={rpc}",
        f"--metrics_port={http}",
        "--enable_metric_reporting=false",
    ]
    (output / "master-args.json").write_text(json.dumps(args))
    donor = None
    with (output / "master.log").open("w") as log:
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
                )
                == 0
            )
            yield NativeTestStore(donor=donor, rpc_port=rpc, http_port=http)
        finally:
            if donor is not None:
                donor.close()
            terminate_and_kill_process_tree(master, wait_timeout=10)
