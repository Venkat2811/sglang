"""Optional output-stage timing for a real scheduler, including spawned ranks.

Put this directory on PYTHONPATH and set OUTPUT_STREAMER_BENCH_DIR. The hook
does not import SGLang early; it instruments the normal imports in each process.
Only the requested batch size is timed. File writes happen after the timer.
"""

import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time
from pathlib import Path

_OUTPUT = os.environ.get("OUTPUT_STREAMER_BENCH_DIR")
_BATCH = int(os.environ.get("OUTPUT_STREAMER_BENCH_BATCH_SIZE", "256"))


class _CurrentBatch(threading.local):
    def __init__(self):
        self.row = None


_CURRENT = _CurrentBatch()
_STREAMER = "sglang.srt.managers.scheduler_components.output_streamer"
_BRIDGE = "sglang.srt.rust_server.server"


def _instrument(module):
    if module.__name__ == _BRIDGE:
        original = module.RustServer.push_generation

        @functools.wraps(original)
        def bridge(self, payload):
            row = _CURRENT.row
            if row is None:
                return original(self, payload)
            details = payload.cached_tokens_details
            row["detail_entries"] = (
                sum(value is not None for value in details) if details else 0
            )
            row["emitted_requests"] = len(payload.rids)
            start = time.perf_counter_ns()
            try:
                return original(self, payload)
            finally:
                row["bridge_ns"] = time.perf_counter_ns() - start

        module.RustServer.push_generation = bridge
        return

    original = module.SchedulerOutputStreamer._stream_output_generation

    @functools.wraps(original)
    def output(self, reqs, *args, **kwargs):
        if len(reqs) != _BATCH or self.rust_server is None:
            return original(self, reqs, *args, **kwargs)
        row = {
            "at_ns": time.time_ns(),
            "pid": os.getpid(),
            "batch_size": len(reqs),
            "first": sum(req.send_token_offset == 0 for req in reqs),
            "finished": sum(req.finished() for req in reqs),
            "cached": sum(req.cached_tokens > 0 for req in reqs),
        }
        _CURRENT.row = row
        start = time.perf_counter_ns()
        try:
            return original(self, reqs, *args, **kwargs)
        finally:
            row["streamer_ns"] = time.perf_counter_ns() - start
            _CURRENT.row = None
            destination = Path(_OUTPUT)
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / f"output-{os.getpid()}.jsonl").open("a") as output:
                output.write(json.dumps(row) + "\n")

    module.SchedulerOutputStreamer._stream_output_generation = output


class _ProfileImports(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.pending = {_STREAMER, _BRIDGE}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.pending:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        execute = spec.loader.exec_module

        def execute_and_profile(module):
            execute(module)
            _instrument(module)

        spec.loader.exec_module = execute_and_profile
        self.pending.remove(fullname)
        if not self.pending:
            sys.meta_path.remove(self)
        return spec


if _OUTPUT:
    sys.meta_path.insert(0, _ProfileImports())
