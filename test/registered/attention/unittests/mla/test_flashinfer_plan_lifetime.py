"""A host planner must not overwrite staging memory still read by a queued copy."""

import unittest

import torch
from flashinfer.mla import BatchMLAPagedAttentionWrapper

from sglang.srt.layers.attention.flashinfer_mla_backend import fast_mla_decode_plan
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestMLAPlanLifetime(CustomTestCase):
    def test_queued_replays_keep_their_own_plan(self):
        torch.manual_seed(42)
        for bs in (1, 4):
            with self.subTest(batch_size=bs):
                self._check_queued_replays(bs)

    def _check_queued_replays(self, bs):
        device, dtype, heads = "cuda", torch.bfloat16, 8
        capacity = 32768
        q = torch.randn(bs, heads, 512, device=device, dtype=dtype)
        q_pe = torch.randn(bs, heads, 64, device=device, dtype=dtype)
        kv = torch.randn(capacity, 1, 576, device=device, dtype=dtype)
        indices = torch.arange(capacity, device=device, dtype=torch.int32)
        qo = torch.arange(bs + 1, dtype=torch.int32)

        def plan_args(lengths):
            lens = torch.tensor(lengths, dtype=torch.int32)
            indptr = torch.cat((torch.zeros(1, dtype=torch.int32), lens.cumsum(0)))
            return (
                qo,
                indptr.to(torch.int32),
                indices,
                lens,
                heads,
                512,
                64,
                1,
                False,
                192**-0.5,
                dtype,
                dtype,
            )

        def workspace():
            return torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)

        reference = BatchMLAPagedAttentionWrapper(workspace(), backend="fa2")
        captured = BatchMLAPagedAttentionWrapper(
            workspace(),
            use_cuda_graph=True,
            qo_indptr=qo.to(device),
            kv_indptr=qo.to(device),
            kv_indices=indices,
            kv_len_arr=torch.ones(bs, device=device, dtype=torch.int32),
            backend="fa2",
        )
        captured.plan(*plan_args([1] * bs))

        def run(wrapper):
            return wrapper.run(q, q_pe, kv[..., :512], kv[..., 512:])

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run(captured)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = run(captured)

        cases, expected = [], []
        for length in (30, 1959, 8000):
            args = plan_args([length + i for i in range(bs)])
            reference.plan(*args)
            cases.append(args)
            expected.append(run(reference).clone())
        torch.cuda.synchronize()

        # Emulate a busy device while Python plans later decode steps. Keeping
        # each output on-device avoids accidentally synchronizing the planner.
        torch.cuda._sleep(100_000_000)
        observed = []
        for i in range(12):
            fast_mla_decode_plan(captured, *cases[i % len(cases)])
            graph.replay()
            observed.append(output.clone())
        torch.cuda.synchronize()
        for i, actual in enumerate(observed):
            torch.testing.assert_close(actual, expected[i % len(cases)], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
