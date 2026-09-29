import unittest
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.models.kimi_linear as kimi
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestKimiMoEStreamOrder(CustomTestCase):
    def test_shared_input_survives_inplace_routed_expert(self):
        # Isolate stream ownership without model weights or a TP process group.
        model = kimi.KimiMoE.__new__(kimi.KimiMoE)
        nn.Module.__init__(model)
        model.alt_stream = torch.cuda.Stream()
        model.num_shared_experts = 1
        model.tp_size = 1
        model.gate = lambda h: (h, None)
        model.topk = lambda h, logits: None
        model.experts = lambda h, routing: h.fill_(7)
        model.shared_experts = lambda h: h * 2
        hidden = torch.ones(4, 64, device="cuda")
        clone = torch.Tensor.clone

        def delayed_clone(tensor, *args, **kwargs):
            # Let the routed stream reach its write before the shared copy's
            # read. Correct dependencies must survive this scheduling order.
            if tensor.data_ptr() == hidden.data_ptr():
                torch.cuda._sleep(100_000_000)
            return clone(tensor, *args, **kwargs)

        with (
            patch.object(kimi, "get_is_capture_mode", return_value=True),
            patch.object(torch.Tensor, "clone", delayed_clone),
        ):
            for _ in range(2):
                hidden.fill_(1)
                model(hidden)
                torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                hidden.fill_(1)
                output = model(hidden)
            graph.replay()
            torch.cuda.synchronize()

        # Shared contribution is 2 * original input, routed contribution is 7.
        torch.testing.assert_close(output, torch.full_like(output, 9), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
