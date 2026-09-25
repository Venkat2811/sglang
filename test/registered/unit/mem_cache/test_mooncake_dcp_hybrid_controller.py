"""Four-rank MLA/KDA checkpoint contracts while hybrid DCP remains gated.

The fixture assembles real host pools below the public capability guard. Workers,
collectives, abort handling and the all-or-nothing acceptance check are real.
CPU CI substitutes a byte-copying native client; the manual test uses TCP. The
final radix publication/lock boundary is supplied by the fixture.
"""

import json
import tempfile
import threading
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_mooncake_dcp_storage import _mamba_pool, _own_mamba, _page_segments
from test_mooncake_dcp_storage_controller import (
    _backup_done,
    _cache,
    _FaultClient,
    _make_controller,
    run_workers,
)

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import _OngoingPrefetch
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=35, suite="base-a-test-cpu")

CASES = (
    "kv_only_rank",
    "sparse",
    "disjoint",
    "complete",
    "missing_state",
    "missing_component",
    "query_cancelled",
    "query_failure",
    "evicted_state",
    "cancel_inflight",
)


def _state_sets(case):
    if case in ("sparse", "kv_only_rank"):
        return [{1, 4}, {1, 3}, {1, 2, 4}, {1, 3, 4}]
    if case == "disjoint":
        return [{2, 4}, {1, 3}, {2, 4}, {1, 3}]
    if case == "missing_state":
        return [{1, 4}, set(), {1, 4}, {1, 4}]
    return [{1, 4} for _ in range(4)]


def _drain(cache, count=0):
    cache._drain_storage_control_queues_impl(
        n_storage_hit=0,
        n_ack_prefetch=count,
        n_backup=0,
        n_release=None,
        extra_release_counts={PoolName.MAMBA: None},
        log_metrics=False,
    )


def _state_tag(rank, component, boundary):
    return 17 * rank + 5 * component + boundary


def _run_case(rank, directory, objects, case, address):
    cc = _make_controller(rank, objects, "hybrid-" + case, address)
    kv, state = cc.storage_host_pool, _mamba_pool()
    cc.mem_pool_host = HostPoolGroup(
        [
            *cc.mem_pool_host.entries,
            PoolEntry(PoolName.MAMBA, state, state.device_pool, lambda layer: layer),
        ]
    )
    cc.extra_host_mem_release_queues = {PoolName.MAMBA: Queue()}
    cc.storage_backend.register_mem_host_pool_v2(state, PoolName.MAMBA)
    cache = _cache(cc)
    cache.tree_core = SimpleNamespace(page_size=128)
    cache.storage_existence_cache = StorageExistenceCache()
    tokens = list(range(512))
    hashes = get_storage_hash_str(tokens, None, page_size=128)
    client = _FaultClient(cc.storage_backend.store, rank, hashes)
    cc.storage_backend.store = client
    legal = _state_sets(case)
    boundaries = sorted(legal[rank])
    kv_source = kv.alloc(512)
    kv.kv_buffer.fill_(rank % 2 + 1)
    kv_oracle = kv.kv_buffer.clone()
    # State slots are neither logical token pages nor divisible by DCP.
    state_guard = state.alloc(3)
    state_source = state.alloc(len(boundaries))
    for component, buffer in enumerate(state.get_hybrid_pool_buffer()):
        for slot, boundary in zip(state_source.tolist(), boundaries):
            buffer[slot].view(torch.uint8).fill_(_state_tag(rank, component, boundary))
    client.own(kv, kv_source)
    if address is None:
        _own_mamba(client.client, state, state_source.tolist())
    outgoing = (
        [
            PoolTransfer(
                PoolName.MAMBA,
                host_indices=state_source,
                keys=[hashes[p - 1] for p in boundaries],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]
        if boundaries
        else None
    )
    # The fixture keeps a separate reference to the seed allocations through
    # restore, so a destination cannot accidentally reuse a source slot.
    cache.dec_host_lock_ref = lambda node, indices: None

    with mock.patch("sglang.srt.managers.cache_controller.STORAGE_BATCH_SIZE", 1):
        HiCacheController._start_storage_threads(cc)
        try:
            ident = cc.write_storage(kv_source, tokens, hashes, extra_pools=outgoing)
            cache.ongoing_backup[ident] = (0, kv_source)
            _backup_done(cc, cache, 512 if rank < 2 or boundaries else 0)
            assert sum(key.endswith("_k") for key in client.put_keys) == (
                4 if rank < 2 else 0
            )
            assert sum(not key.endswith("_k") for key in client.put_keys) == 3 * len(
                boundaries
            )
            dist.barrier()
            incoming = PoolTransfer(
                PoolName.MAMBA,
                keys=["__placeholder__"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )

            def remove_state_component():
                keys, _ = cc.storage_backend._get_hybrid_page_component_keys(
                    [hashes[3]], incoming
                )
                client.remove(cc.storage_backend._tag_keys(keys)[1])

            if case == "missing_component" and rank == 1:
                remove_state_component()
            dist.barrier()
            candidates = [set(pages) for pages in legal]
            if case == "missing_component":
                candidates[1].remove(4)
            expected = max(set.intersection(*candidates), default=0)
            if case in ("query_cancelled", "query_failure"):
                expected = 0
            op = PrefetchOperation(
                CacheRequestHandle(case, 0), tokens, pool_transfers=[incoming]
            )
            if case == "kv_only_rank" and rank == 3:
                # A rank/stage with no needed state still participates in the
                # same query collective as ranks with sparse checkpoints.
                op.pool_transfers = None
            if case == "query_cancelled" and rank == 1:
                op.mark_terminate()
            if case == "query_failure":
                client.mode = "lookup_exception"
            cc.prefetch_queue.put(op)
            assert cc.prefetch_hit_queue.get(timeout=10) is op
            assert op.storage_hit_count == expected * 128, (
                case,
                rank,
                op.storage_hit_count,
                expected * 128,
            )
            assert op.hash_value == hashes[:expected]
            published = []
            if expected and case != "kv_only_rank":
                if case == "evicted_state" and rank == 1:
                    remove_state_component()
                dist.barrier()
                op.host_indices = kv.alloc(expected * 128)
                incoming.host_indices = state.alloc(1)
                client.own(kv, op.host_indices)
                if address is None:
                    _own_mamba(client.client, state, incoming.host_indices.tolist())
                for buffer in state.get_hybrid_pool_buffer():
                    buffer.view(torch.uint8).fill_(165)
                state_before = [
                    b.view(torch.uint8).clone() for b in state.get_hybrid_pool_buffer()
                ]
                cache.ongoing_prefetch[op.handle] = _OngoingPrefetch(
                    0,
                    RadixKey(tokens),
                    op.host_indices,
                    op,
                    None,
                    {PoolName.MAMBA: [incoming]},
                )
                cc.prefetch_tokens_occupied = len(tokens)

                def publish(operation):
                    if not cache._check_hybrid_prefetch_result(
                        op.handle,
                        operation,
                        operation.completed_tokens,
                        operation.hash_value,
                        operation.host_indices,
                        0,
                        None,
                        RadixKey(tokens),
                    ):
                        return
                    assert incoming.keys == [hashes[expected - 1]]
                    for page, dst in enumerate(
                        (op.host_indices[::128] // 128).tolist()
                    ):
                        for actual, reference in zip(
                            _page_segments(kv, kv.kv_buffer, dst),
                            _page_segments(kv, kv_oracle, page),
                        ):
                            torch.testing.assert_close(
                                actual, reference, rtol=0, atol=0
                            )
                    for component, (buffer, oracle) in enumerate(
                        zip(state.get_hybrid_pool_buffer(), state_before)
                    ):
                        oracle[incoming.host_indices] = _state_tag(
                            rank, component, expected
                        )
                        torch.testing.assert_close(
                            buffer.view(torch.uint8), oracle, rtol=0, atol=0
                        )
                    published.append(operation.completed_tokens)
                    cc.append_host_mem_release(operation.host_indices, [incoming])
                    del cache.ongoing_prefetch[operation.handle]

                cache._handle_prefetch_result = publish
                if case == "cancel_inflight":
                    client.mode = case
                cc.prefetch_buffer.put(op)
                if case == "cancel_inflight":
                    if rank == 1:
                        assert client.entered.wait(10)
                    dist.barrier()
                    before_available = state.available_size()
                    cache.release_aborted_request(op.handle)
                    _drain(cache)
                    assert kv.slot_used[op.host_indices].all()
                    assert state.available_size() == before_available, (
                        "state released before IO completion"
                    )
                    dist.barrier()
                    client.release.set()
                acks = []
                while True:
                    ack = cc.ack_prefetch_queue.get(timeout=10)
                    acks.append(ack)
                    if ack.completed_req:
                        break
                assert len(acks) == expected + 2, (
                    "KV progress, state result and final ACK must all arrive"
                )
                for ack in acks:
                    cc.ack_prefetch_queue.put(ack)
                _drain(cache, len(acks))
                assert published == (
                    []
                    if case in ("evicted_state", "cancel_inflight")
                    else [expected * 128]
                )
                assert not cache.ongoing_prefetch
            kv.free(kv_source)
            state.free(state_source)
            state.free(state_guard)
            assert int(kv.slot_used.sum()) == 0
            assert state.available_size() == state.size
            free = torch.cat([state.free_slots, *state.release_slots])
            assert free.unique().numel() == state.size, "duplicate state release"
            assert (
                cc.prefetch_thread.is_alive() and cc.prefetch_io_aux_thread.is_alive()
            )
            return dict(
                case=case,
                rank=rank,
                tokens=published[0] if published else 0,
                lookup=expected * 128,
                remaining_slots=0,
            )
        finally:
            client.release.set()
            HiCacheController._stop_storage_threads(cc)
            cc._destroy_sync_groups(cc.prefetch_hits_sync_groups)
            cc._destroy_sync_groups(cc.prefetch_completion_sync_groups)
            if address is not None:
                client.client.close()


def _worker(rank, directory, objects, cases=CASES, address=None):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    reports = []
    try:
        for case in cases:
            reports.append(_run_case(rank, directory, objects, case, address))
            dist.barrier()
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(reports))
    except Exception:
        print(f"FAILED hybrid case={case} rank={rank}", flush=True)
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


class TestMooncakeDcpHybridController(CustomTestCase):
    def test_trailing_maximum_does_not_prove_earlier_checkpoints(self):
        """Legacy results and assume-stored hints prove only the trailing endpoint."""
        for policy, assume_stored, expected in (
            (PoolHitPolicy.TRAILING_PAGES, False, 0),
            (PoolHitPolicy.ALL_PAGES, False, 8),
            (PoolHitPolicy.TRAILING_PAGES, True, 0),
        ):
            with self.subTest(policy=policy, assume_stored=assume_stored):
                cc = HybridCacheController.__new__(HybridCacheController)
                cc.page_size = 4
                cc.prefetch_queue, cc.prefetch_hit_queue = Queue(), Queue()
                cc.storage_stop_event = threading.Event()
                cc.prefetch_hits_sync_groups = []
                cc.storage_backend = mock.Mock()
                cc.storage_backend.batch_exists_v2.return_value = PoolTransferResult(
                    4, {}
                )
                if assume_stored:
                    cc.storage_backend.batch_exists_v2.side_effect = AssertionError(
                        "hint should skip lookup"
                    )

                def peer_reduction(tensor, reduce_op, groups):
                    # External collective boundary: the peer can restore only
                    # page 2. The production query/worker chooses the payload.
                    if tensor.ndim == 0:
                        tensor.fill_(min(int(tensor), 8))
                    else:
                        peer = torch.zeros_like(tensor)
                        peer[2] = 1
                        tensor.mul_(peer)
                    cc.storage_stop_event.set()

                cc._all_reduce = peer_reduction
                transfer = PoolTransfer(
                    PoolName.MAMBA, keys=["__placeholder__"], hit_policy=policy
                )
                operation = PrefetchOperation(
                    CacheRequestHandle("hint", 0),
                    list(range(16)),
                    pool_transfers=[transfer],
                    assume_stored=assume_stored,
                )
                cc.prefetch_queue.put(operation)
                cc.prefetch_thread_func()
                self.assertIs(cc.prefetch_hit_queue.get_nowait(), operation)
                self.assertEqual(operation.storage_hit_count, expected)

    def test_sparse_checkpoints_and_state_lifetime_on_four_ranks(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mp.get_context("spawn").Manager() as manager,
        ):
            reports = run_workers(
                directory, manager.dict(), cases=CASES, worker=_worker
            )
            self.assertTrue(all(len(rows) == len(CASES) for rows in reports))


if __name__ == "__main__":
    unittest.main()
