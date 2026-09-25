"""Checkpoint donation and storage-key publication through the real radix tree.

CPU pools exercise the logical checkpoint protocol, not DCP physical KV layout.
The forward kernel is the boundary: it writes a tagged snapshot at the scheduler's
chosen depth. Separate GPU tests qualify device/host copies of Kimi state bytes.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, release_req
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams, MatchPrefixParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.runtime_context import reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestMambaCheckpointPublication(CustomTestCase):
    def setUp(self):
        self.addCleanup(reset_context)

    def _fixture(self, page):
        cfg = CacheConfig(
            page_size=page,
            components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
            num_layers=2,
            full_attention_layer_ids=(0,),
            kv_size=page * 16,
            max_context_len=page * 8,
        )
        # Hardware selection only; pools, allocators, components and tree are real.
        with (
            patch("test_unified_radix_cache_unittest.get_device", return_value="cpu"),
            patch(
                "test_unified_radix_cache_unittest._TREE_CORE_TEST_BACKEND", "python"
            ),
        ):
            cache, allocator, pool = build_fixture(cfg, mamba_cache_chunk_size=64)
        cache.tree_core.enable_storage = True
        return cache, allocator, pool

    def _request(self, cache, allocator, pool, tokens, rid):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=32),
        )
        pool.alloc([req])
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, len(tokens))
        req.kv.kv_committed_len = req.kv.kv_allocated_len = len(tokens)
        req.last_node = cache.root_node_handle()
        req.lock_receipt = DecLockRefParams()
        indices = allocator.alloc(len(tokens))
        self.assertIsNotNone(indices)
        pool.write((req.kv.req_pool_idx, slice(0, len(tokens))), indices)
        return req

    def _forward_snapshot(self, cache, pool, req):
        batch = ScheduleBatch(reqs=[req])
        batch.tree_cache = cache
        batch.req_to_token_pool = pool
        batch.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(mamba_chunk_size=64)
        )
        with patch(
            "sglang.srt.managers.schedule_batch.get_parallel",
            return_value=SimpleNamespace(dcp_enabled=True),
        ):
            entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)
        self.assertTrue(entry.track_mask)
        depth = req.kv.mamba_last_track_seqlen
        # Independent prefix-dependent byte oracle at the forward boundary.
        signature = sum(req.origin_input_ids[:depth]) % 16000
        expected = []
        for component, buffer in enumerate(
            (pool.mamba_pool.mamba_cache.temporal, *pool.mamba_pool.mamba_cache.conv)
        ):
            value = buffer[:, entry.track_index]
            words = torch.arange(value.numel(), dtype=torch.int32).reshape(value.shape)
            value.copy_(words + signature + component * 31)
            expected.append(value.clone())
        return entry.track_index, depth, expected

    def test_donated_bytes_and_terminal_hash_survive_request_writes_and_split(self):
        """A split prefix cannot inherit a later state; a donor must own its slot."""
        for page in (128, 512):
            with self.subTest(page=page):
                cache, allocator, pool = self._fixture(page)
                tokens = list(range(4 * page + 31))
                req = self._request(cache, allocator, pool, tokens, "owner")
                tracked_slot, depth, expected = self._forward_snapshot(cache, pool, req)
                self.assertEqual(depth, 4 * page)
                cache.cache_unfinished_req(req)
                leaf = cache.tree_core.node_by_id(req.last_node)
                state = leaf.component_data[ComponentType.MAMBA]
                self.assertEqual(state.value.tolist(), [tracked_slot])
                self.assertNotIn(
                    tracked_slot, req.kv.mamba_ping_pong_track_buffer.tolist()
                )
                # Later kernels may overwrite every request-owned state slot.
                request_slots = torch.cat(
                    (
                        req.kv.mamba_pool_idx.view(-1),
                        req.kv.mamba_ping_pong_track_buffer,
                    )
                )
                buffers = [
                    pool.mamba_pool.mamba_cache.temporal,
                    *pool.mamba_pool.mamba_cache.conv,
                ]
                for buffer, wanted in zip(buffers, expected):
                    buffer[:, request_slots] = -7
                    torch.testing.assert_close(
                        buffer[:, tracked_slot], wanted, rtol=0, atol=0
                    )
                # Model successful D->H completion with real host tensors and
                # the actual component transfer/commit boundary, without CUDA.
                host = MambaPoolHost(
                    pool.mamba_pool, 2, 0, pin_memory=False, layout="page_first"
                )
                self.addCleanup(host.destroy)
                slots = host.alloc(1)
                transfers = cache.tree_core.build_hicache_transfers(
                    ComponentType.MAMBA, leaf.id, CacheTransferPhase.BACKUP_HOST
                )
                for src, dst in zip(buffers, host.get_hybrid_pool_buffer()):
                    dst[slots[0], :, 0] = src[:, transfers[0].device_indices[0]]
                transfers[0].host_indices = slots
                cache.tree_core.commit_backup(
                    leaf.id, torch.arange(depth), {ComponentType.MAMBA: transfers}
                )
                hashes = get_storage_hash_str(tokens[:depth], None, page_size=page)
                before = cache.tree_core.build_storage_backup_spec(leaf.id, True)
                self.assertEqual(before.hash_value, hashes)
                receipt = cache.inc_host_lock_ref(leaf.id).to_dec_params()
                cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", tokens[: 2 * page])))
                )
                parent = leaf.parent
                self.assertIsNone(parent.component_data[ComponentType.MAMBA].value)
                self.assertIsNone(parent.component_data[ComponentType.MAMBA].host_value)
                self.assertEqual(state.host_lock_ref, 1)
                after = cache.tree_core.build_storage_backup_spec(leaf.id, True)
                self.assertEqual(after.hash_value, hashes[2:])
                self.assertEqual(after.prefix_keys, hashes[:2])
                for spec in (before, after):
                    transfer = spec.comp_xfers[ComponentType.MAMBA][0]
                    self.assertEqual(transfer.keys, [hashes[-1]])
                    for actual, wanted in zip(host.get_hybrid_pool_buffer(), expected):
                        torch.testing.assert_close(
                            actual[transfer.host_indices[0], :, 0],
                            wanted,
                            rtol=0,
                            atol=0,
                        )
                cache.dec_host_lock_ref(leaf.id, receipt)
                self.assertEqual(state.host_lock_ref, 0)
                # True retraction releases the live request, while the donated
                # checkpoint remains keyed and owned by the tree.
                self.assertTrue(
                    release_req(
                        req=req,
                        remaing_req_count=0,
                        req_to_token_pool=pool,
                        token_to_kv_pool_allocator=allocator,
                        tree_cache=cache,
                        hisparse_coordinator=None,
                        offload_kv=False,
                    )
                )
                self.assertIsNone(req.kv.mamba_last_track_seqlen)
                self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)
                self.assertEqual(
                    pool.mamba_allocator.available_size(), pool.mamba_pool.size - 1
                )
                self.assertNotIn(tracked_slot, pool.mamba_allocator.free_slots.tolist())
                for actual, wanted in zip(buffers, expected):
                    torch.testing.assert_close(
                        actual[:, tracked_slot], wanted, rtol=0, atol=0
                    )

    def test_duplicate_donation_releases_only_the_redundant_state(self):
        """Finishing/retracting a duplicate must not free the canonical checkpoint."""
        for finish_duplicate in (False, True):
            with self.subTest(finish_duplicate=finish_duplicate):
                page = 128
                cache, allocator, pool = self._fixture(page)
                tokens = list(range(2 * page + 31))
                initial = pool.mamba_allocator.available_size()
                original = self._request(cache, allocator, pool, tokens, "original")
                slot, depth, expected = self._forward_snapshot(cache, pool, original)
                release_kv_cache(original, cache, is_insert=True)
                self.assertEqual(pool.mamba_allocator.available_size(), initial - 1)
                duplicate = self._request(cache, allocator, pool, tokens, "duplicate")
                extra_slot, _, _ = self._forward_snapshot(cache, pool, duplicate)
                self.assertNotEqual(extra_slot, slot)
                if finish_duplicate:
                    release_kv_cache(duplicate, cache, is_insert=True)
                else:
                    cache.cache_unfinished_req(duplicate)
                    self.assertIn(extra_slot, pool.mamba_allocator.free_slots.tolist())
                    self.assertTrue(
                        release_req(
                            req=duplicate,
                            remaing_req_count=0,
                            req_to_token_pool=pool,
                            token_to_kv_pool_allocator=allocator,
                            tree_cache=cache,
                            hisparse_coordinator=None,
                            offload_kv=False,
                        )
                    )
                self.assertEqual(pool.mamba_allocator.available_size(), initial - 1)
                free = pool.mamba_allocator.free_slots
                self.assertEqual(free.unique().numel(), free.numel())
                self.assertNotIn(slot, free.tolist())
                match = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", tokens)))
                )
                self.assertEqual(len(match.device_indices), depth)
                leaf = cache.tree_core.node_by_id(match.last_device_node)
                self.assertEqual(
                    leaf.component_data[ComponentType.MAMBA].value.tolist(), [slot]
                )
                for actual, wanted in zip(
                    (
                        pool.mamba_pool.mamba_cache.temporal,
                        *pool.mamba_pool.mamba_cache.conv,
                    ),
                    expected,
                ):
                    torch.testing.assert_close(actual[:, slot], wanted, rtol=0, atol=0)
                self.assertEqual(allocator.available_size(), allocator.size - depth)
                cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
