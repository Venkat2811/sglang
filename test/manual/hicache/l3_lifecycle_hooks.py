"""Test-only observation and a controllable pause at the real file IO boundary.

Installed only in a manual sweep's temporary plugin directory. The hooks never
replace cache decisions, acknowledgments, tensor bytes, or numerical results.
The driver cancels a real HTTP request or removes a real state file while the
decode IO worker is paused, then releases the original backend method.
"""

import json
import os
import time
from pathlib import Path

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.runtime_context import get_parallel

_last_snapshots = {}


def _arm():
    root = Path(os.environ["DCP_L3_TEST_EVENTS_DIR"])
    path = root / "arm.json"
    try:
        return root, json.loads(path.read_text())
    except FileNotFoundError:
        return root, None


def _record(root, arm, kind, rank, payload):
    path = root / f"{arm['id']}-{kind}-{rank}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def _snapshot(cache):
    device_tokens, auxiliary_slots = cache.total_size()
    return dict(
        host_kv=cache.host_pool_group.available_size(PoolName.KV),
        host_state=cache.host_pool_group.available_size(PoolName.MAMBA),
        device_kv=cache.token_to_kv_pool_allocator.available_size(),
        device_state=cache.req_to_token_pool.mamba_allocator.available_size(),
        request_slots=cache.req_to_token_pool.available_size(),
        ongoing_prefetch=len(cache.ongoing_prefetch),
        ongoing_load_back=len(cache.ongoing_load_back),
        tree_device_tokens=device_tokens,
        tree_auxiliary_slots=auxiliary_slots,
    )


def _prefetch(original, cache, request, *args, **kwargs):
    root, arm = _arm()
    if arm is not None and request.rid == arm["rid"]:
        _record(root, arm, "before", get_parallel().tp_rank, _snapshot(cache))
    return original(cache, request, *args, **kwargs)


def _metadata(original, queue, decode_req, *args, **kwargs):
    result = original(queue, decode_req, *args, **kwargs)
    root, arm = _arm()
    if arm is not None and decode_req.req.rid == arm["rid"]:
        _record(
            root,
            arm,
            "promise",
            queue.tp_rank,
            dict(rid=decode_req.req.rid, prefix=kwargs["decode_prefix_len"]),
        )
    return result


def _read(original, backend, transfers, *args, **kwargs):
    root, arm = _arm()
    if (
        arm is not None
        and backend._state_tp_rank == arm["pause_rank"]
        and any(t.name == PoolName.MAMBA for t in transfers)
    ):
        rank = backend._state_tp_rank
        _record(
            root,
            arm,
            "entered",
            rank,
            dict(
                host_state=backend.registered_pools[PoolName.MAMBA].available_size(),
                keys=[
                    key for t in transfers if t.name == PoolName.MAMBA for key in t.keys
                ],
            ),
        )
        deadline = time.monotonic() + 45
        while not (root / f"{arm['id']}.release").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Manual test did not release paused state IO")
            time.sleep(0.01)
    result = original(backend, transfers, *args, **kwargs)
    if arm is not None and backend._state_tp_rank == arm["pause_rank"]:
        _record(root, arm, "read-result", backend._state_tp_rank, result)
    return result


def _abort(original, cache, request):
    result = original(cache, request)
    root, arm = _arm()
    if arm is not None and request.rid == arm["rid"]:
        _record(
            root,
            arm,
            "aborted",
            get_parallel().tp_rank,
            _snapshot(cache),
        )
    return result


def _events(original, cache):
    result = original(cache)
    root, arm = _arm()
    if arm is not None and (root / f"{arm['id']}.release").exists():
        rank = get_parallel().tp_rank
        snapshot = _snapshot(cache)
        identity = (arm["id"], rank)
        if _last_snapshots.get(identity) != snapshot:
            _record(root, arm, "after", rank, snapshot)
            _last_snapshots[identity] = snapshot
    return result


def _probe(original, cache, last_host_node_id, new_input_tokens, *args, **kwargs):
    started = time.perf_counter()
    result = original(cache, last_host_node_id, new_input_tokens, *args, **kwargs)
    elapsed = time.perf_counter() - started
    root = Path(os.environ["DCP_L3_TEST_EVENTS_DIR"])
    record = dict(
        input_tokens=len(new_input_tokens),
        storage_hit_tokens=result,
        elapsed_seconds=elapsed,
    )
    with (root / f"probe-timing-{get_parallel().tp_rank}.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")
    return result


def _publish(original, tree, node_id, key, host_value, hash_value):
    result = original(tree, node_id, key, host_value, hash_value)
    root, arm = _arm()
    record = dict(
        key_tokens=len(key),
        dropped=result.host_insert_dropped,
        interrupted_request=None if arm is None else arm["id"],
    )
    with (root / f"publications-{get_parallel().tp_rank}.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")
    return result


def register():
    cache = "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache"
    for target, hook in (
        (cache + ".prefetch_from_storage", _prefetch),
        (cache + ".release_aborted_request", _abort),
        (cache + ".check_hicache_events", _events),
        (cache + ".query_storage_hit_length", _probe),
        (
            "sglang.srt.disaggregation.decode.DecodePreallocQueue._send_kv_metadata",
            _metadata,
        ),
        ("sglang.srt.mem_cache.hicache_storage.HiCacheFile.batch_get_v2", _read),
    ):
        HookRegistry.register(target, hook, HookType.AROUND)
    for tree in (
        "sglang.srt.mem_cache.unified_cache.unified_tree_core.UnifiedTreeCore",
        "sglang.srt.mem_cache.rust_tree_core.adapter.RustUnifiedTreeCore",
    ):
        HookRegistry.register(tree + ".insert_host", _publish, HookType.AROUND)
