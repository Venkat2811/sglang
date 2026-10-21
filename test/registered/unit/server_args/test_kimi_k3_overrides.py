"""Kimi-K3's config-time overrides pick settings its expert-parallel paths can run.

python -m pytest test/registered/unit/server_args/test_kimi_k3_overrides.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_overrides import kimi_k3 as k3_overrides
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MXFP4_HF_CONFIG = SimpleNamespace(
    quantization_config={
        "config_groups": {"experts": {"format": "mxfp4-pack-quantized"}}
    }
)
BLACKWELL = SimpleNamespace(is_sm100=True, device_sm=103)


class TestKimiK3MoeRunnerOverride(CustomTestCase):
    def runner_for(self, *, a2a, runner="auto"):
        server_args = SimpleNamespace(moe_runner_backend=runner, moe_a2a_backend=a2a)
        with patch.object(k3_overrides, "get_platform", return_value=BLACKWELL):
            return k3_overrides._kimi_k3_moe_runner_overrides(
                server_args, MXFP4_HF_CONFIG
            ).get("moe_runner_backend")

    def test_deepep_backends_resolve_to_the_runner_that_consumes_them(self):
        """flashinfer_mxfp4 has no DeepEP fused function: resolving auto to it
        under deepep failed at model construction, under deepep_v2 at startup."""
        self.assertEqual(self.runner_for(a2a="deepep"), "deep_gemm")
        self.assertEqual(self.runner_for(a2a="deepep_v2"), "deep_gemm")
        self.assertEqual(self.runner_for(a2a="none"), "flashinfer_mxfp4")

    def test_an_explicit_runner_is_kept(self):
        self.assertIsNone(self.runner_for(a2a="deepep", runner="flashinfer_mxfp4"))


if __name__ == "__main__":
    unittest.main()
