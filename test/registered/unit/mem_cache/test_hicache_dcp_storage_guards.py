"""Startup and runtime support boundaries for DCP storage."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from test_hicache_dcp_host_pool import _make_host_pool
from test_mooncake_dcp_storage import _mamba_pool

from sglang.srt.arg_groups.hicache_hook import (
    resolve_hicache_dcp_compatibility,
    validate_hicache_dcp_storage,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _args(**changes):
    options = dict(
        model_path="dummy",
        tp_size=4,
        dcp_size=2,
        enable_hierarchical_cache=True,
        hicache_storage_backend="file",
        hicache_mem_layout="page_first",
        hicache_io_backend="kernel",
        dtype="bfloat16",
        kv_cache_dtype="auto",
        hicache_write_policy="write_through",
        hicache_storage_prefetch_policy="wait_complete",
    )
    options.update(changes)
    return ServerArgs(**options)


class TestDcpStorageGuards(CustomTestCase):
    def test_pd_storage_stays_gated_until_transfer_support(self):
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for role in ("prefill", "decode"):
                with (
                    self.subTest(role=role),
                    self.assertRaisesRegex(NotImplementedError, "aggregated serving"),
                ):
                    resolve_hicache_dcp_compatibility(_args(disaggregation_mode=role))

    def test_materialized_mamba_registration_preserves_independent_slot_pool(self):
        pool = _make_host_pool(0, dcp_size=2, layout="page_first")
        state = _mamba_pool()
        self.addCleanup(pool.destroy)
        self.addCleanup(state.destroy)
        anchor = PoolEntry(
            PoolName.KV,
            pool,
            pool.device_pool,
            lambda x: x,
            is_primary_index_anchor=True,
        )
        sidecar = PoolEntry(PoolName.MAMBA, state, state.device_pool, lambda x: x)
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = HostPoolGroup([anchor])
        controller.enable_storage = True
        controller.storage_config = SimpleNamespace(dcp_size=2)
        controller.storage_backend = mock.Mock()
        controller.extra_host_mem_release_queues = {}
        available = pool.available_size()
        controller.register_host_pool_entry(sidecar)
        self.assertIs(controller.mem_pool_host.anchor_entry, anchor)
        self.assertIs(controller.mem_pool_host.get_pool(PoolName.MAMBA), state)
        slots = controller.mem_pool_host.alloc(1, pool=PoolName.MAMBA)
        self.assertEqual(len(slots), 1)
        self.assertEqual(pool.available_size(), available)
        controller.mem_pool_host.free(slots, pool=PoolName.MAMBA)
        self.assertIn(PoolName.MAMBA, controller.extra_host_mem_release_queues)

    def test_mamba_label_cannot_register_a_token_pool_as_state(self):
        pool = _make_host_pool(0, dcp_size=2, layout="page_first")
        self.addCleanup(pool.destroy)
        anchor = PoolEntry(
            PoolName.KV,
            pool,
            pool.device_pool,
            lambda x: x,
            is_primary_index_anchor=True,
        )
        disguised = PoolEntry(PoolName.MAMBA, pool, pool.device_pool, lambda x: x)
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = HostPoolGroup([anchor])
        controller.enable_storage = True
        controller.storage_config = SimpleNamespace(dcp_size=2)
        controller.extra_host_mem_release_queues = {}
        with self.assertRaisesRegex(NotImplementedError, "materialized Mamba"):
            controller.register_host_pool_entry(disguised)
        self.assertEqual(controller.mem_pool_host.entries, [anchor])
        self.assertEqual(controller.extra_host_mem_release_queues, {})

    def test_mamba_registration_cannot_enable_packed_draft_storage(self):
        pool = _make_host_pool(0, dcp_size=2, layout="page_first")
        state = _mamba_pool()
        self.addCleanup(pool.destroy)
        self.addCleanup(state.destroy)
        anchor = PoolEntry(
            PoolName.KV,
            pool,
            pool.device_pool,
            lambda x: x,
            is_primary_index_anchor=True,
        )
        sidecar = PoolEntry(PoolName.MAMBA, state, state.device_pool, lambda x: x)
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.mem_pool_host = HostPoolGroup([anchor])
        # Model a draft packed into the primary pool: it is not a separate
        # entry, so checking only the sidecar would accidentally admit it.
        controller.mem_pool_host.entries[0] = replace(
            anchor, packed_draft_device_pools=(pool.device_pool,)
        )
        controller.enable_storage = True
        controller.storage_config = SimpleNamespace(dcp_size=2)
        controller.extra_host_mem_release_queues = {}
        controller.storage_backend = mock.Mock()
        with self.assertRaisesRegex(NotImplementedError, "materialized Mamba"):
            controller.register_host_pool_entry(sidecar)
        self.assertEqual(len(controller.mem_pool_host.entries), 1)
        self.assertEqual(controller.extra_host_mem_release_queues, {})

    def test_dynamic_sidecar_cannot_bypass_single_pool_storage_guard(self):
        for dcp in (1, 2):
            with self.subTest(dcp=dcp):
                pool = _make_host_pool(0, dcp_size=dcp, layout="page_first")
                anchor = PoolEntry(
                    PoolName.KV,
                    pool,
                    pool.device_pool,
                    lambda x: x,
                    is_primary_index_anchor=True,
                )
                sidecar = PoolEntry(PoolName.DRAFT, pool, pool.device_pool, lambda x: x)
                controller = HybridCacheController.__new__(HybridCacheController)
                controller.mem_pool_host = HostPoolGroup([anchor])
                controller.enable_storage = True
                controller.storage_config = SimpleNamespace(dcp_size=dcp)
                controller.storage_backend = mock.Mock()
                controller.extra_host_mem_release_queues = {}
                if dcp > 1:
                    with self.assertRaisesRegex(
                        NotImplementedError, "one materialized MLA"
                    ):
                        controller.register_host_pool_entry(sidecar)
                    self.assertEqual(controller.mem_pool_host.entries, [anchor])
                    self.assertEqual(controller.extra_host_mem_release_queues, {})
                else:
                    controller.register_host_pool_entry(sidecar)
                    self.assertEqual(
                        controller.mem_pool_host.entries, [anchor, sidecar]
                    )
                    self.assertIn(
                        PoolName.DRAFT, controller.extra_host_mem_release_queues
                    )

    def test_mooncake_uses_the_same_startup_and_attach_validation(self):
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            args = _args(hicache_storage_backend="mooncake")
            resolve_hicache_dcp_compatibility(args)
            validate_hicache_dcp_storage(_args(), storage_backend="mooncake")

    def test_supported_topologies(self):
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for tp, dcp in ((2, 2), (4, 2), (4, 4)):
                with self.subTest(tp=tp, dcp=dcp):
                    resolve_hicache_dcp_compatibility(_args(tp_size=tp, dcp_size=dcp))
            for dtype in ("bf16", "bfloat16"):
                with self.subTest(kv_cache_dtype=dtype):
                    resolve_hicache_dcp_compatibility(_args(kv_cache_dtype=dtype))

    def test_inherits_mla_hicache_options(self):
        cases = (
            dict(hicache_mem_layout="layer_first"),
            dict(hicache_mem_layout="page_first_direct", hicache_io_backend="direct"),
            dict(dtype="float16"),
            dict(kv_cache_dtype="fp8_e4m3"),
            dict(hicache_write_policy="write_back"),
            dict(hicache_write_policy="write_through_selective"),
            dict(hicache_storage_prefetch_policy="best_effort"),
            dict(hicache_storage_prefetch_policy="timeout"),
            dict(hicache_host_memory_mode="buffer_only"),
            dict(pp_size=2),
            dict(dp_size=2),
            dict(attn_cp_size=2),
            dict(dp_size=2, enable_dp_attention=True),
            dict(speculative_algorithm="DSPARK"),
            dict(disaggregation_mode="prefill", hicache_storage_backend=None),
        )
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for options in cases:
                with self.subTest(options=options):
                    resolve_hicache_dcp_compatibility(_args(**options))

    def test_keeps_existing_dcp_constraints(self):
        with mock.patch(
            "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
        ):
            for options in (
                dict(enable_hisparse=True),
                dict(enable_lmcache=True),
                dict(speculative_algorithm="EAGLE"),
            ):
                with (
                    self.subTest(options=options),
                    self.assertRaises(NotImplementedError),
                ):
                    resolve_hicache_dcp_compatibility(_args(**options))

    def test_runtime_rejection_has_no_side_effects(self):
        for attached in (False, True):
            for mla, backend, message in (
                (False, "file", "MLA"),
                (True, "nixl", "file or Mooncake storage"),
            ):
                cache = SimpleNamespace(
                    cache_controller=SimpleNamespace(write_policy="write_back"),
                    enable_storage=attached,
                    prefetch_stop_policy="timeout",
                    write_through_threshold=2,
                    is_write_back=True,
                )
                attachment = StorageAttachment(cache)
                with (
                    self.subTest(attached=attached, mla=mla, backend=backend),
                    mock.patch(
                        "sglang.srt.runtime_context.get_parallel",
                        return_value=SimpleNamespace(attn_dcp_size=2),
                    ),
                    mock.patch(
                        "sglang.srt.runtime_context.get_server_args",
                        return_value=_args(),
                    ),
                    mock.patch(
                        "sglang.srt.arg_groups.hicache_hook.use_mla_backend",
                        return_value=mla,
                    ),
                ):
                    ok, reason = attachment.attach(
                        backend,
                        hicache_storage_prefetch_policy="wait_complete",
                        hicache_write_policy="write_through",
                    )
                    self.assertFalse(ok)
                    self.assertIn(message, reason)
                    with self.assertRaisesRegex(NotImplementedError, message):
                        validate_hicache_dcp_storage(_args(), storage_backend=backend)
                self.assertEqual(cache.prefetch_stop_policy, "timeout")
                self.assertEqual(cache.cache_controller.write_policy, "write_back")
                self.assertEqual(cache.write_through_threshold, 2)
                self.assertTrue(cache.is_write_back)
                self.assertEqual(cache.enable_storage, attached)

    def test_runtime_policy_updates_use_existing_validation(self):
        cache = SimpleNamespace(
            cache_controller=SimpleNamespace(storage_backend_type="file"),
            enable_storage=True,
        )
        attachment = StorageAttachment(cache)
        with (
            mock.patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=2),
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_server_args", return_value=_args()
            ),
            mock.patch(
                "sglang.srt.arg_groups.hicache_hook.use_mla_backend", return_value=True
            ),
        ):
            for write in ("write_back", "write_through", "write_through_selective"):
                for prefetch in ("best_effort", "timeout", "wait_complete"):
                    with self.subTest(write=write, prefetch=prefetch):
                        ok, reason = attachment.attach(
                            "file",
                            hicache_write_policy=write,
                            hicache_storage_prefetch_policy=prefetch,
                        )
                        self.assertTrue(ok, reason)
                        self.assertEqual(cache.prefetch_stop_policy, prefetch)
                        self.assertEqual(cache.cache_controller.write_policy, write)
                        self.assertEqual(cache.is_write_back, write == "write_back")
                        self.assertEqual(
                            cache.write_through_threshold,
                            1 if write == "write_through" else 2,
                        )
            ok, _ = attachment.attach(
                "file",
                hicache_storage_prefetch_policy="best_effort",
                hicache_write_policy="invalid",
            )
            self.assertFalse(ok)
            self.assertEqual(cache.prefetch_stop_policy, "wait_complete")
            self.assertEqual(
                cache.cache_controller.write_policy, "write_through_selective"
            )
            self.assertEqual(cache.write_through_threshold, 2)
            self.assertFalse(cache.is_write_back)


if __name__ == "__main__":
    unittest.main()
