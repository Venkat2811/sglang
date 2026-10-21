"""Cache-source metadata must survive the Python-to-Rust positional wire."""

import unittest
from array import array
from types import SimpleNamespace

import msgspec

from sglang.srt.rust_server.server import RustServer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGenerationCacheDetails(CustomTestCase):
    def test_details_follow_existing_columns_with_and_without_logprobs(self):
        """Details cannot occupy an old logprob slot or disappear on the fast path."""
        details = [
            None,
            {"device": 2, "host": 3, "storage": 3, "storage_backend": "test-store"},
        ]
        for include_details, logprobs in [(False, False), (True, False), (True, True)]:
            with self.subTest(include_details=include_details, logprobs=logprobs):
                captured = []

                def push(header, columns):
                    captured.append((msgspec.msgpack.decode(header), columns))
                    return True

                bridge = RustServer.__new__(RustServer)
                bridge.server = SimpleNamespace(push_decode_result_batch=push)
                fields = {
                    "rids": ["r0", "r1"],
                    "finished_reasons": [None, None],
                    "output_ids": [array("q", [1]), array("q", [2])],
                    "prompt_tokens": [10, 10],
                    "cached_tokens": [0, 8],
                    "reasoning_tokens": [0, 0],
                    "cached_tokens_details": details if include_details else None,
                    "output_hidden_states": None,
                }
                for family in [
                    "output_token_logprobs",
                    "input_token_logprobs",
                    "output_top_logprobs",
                    "input_top_logprobs",
                    "output_token_ids_logprobs",
                    "input_token_ids_logprobs",
                ]:
                    fields[family + "_val"] = None
                    fields[family + "_idx"] = None
                if logprobs:
                    fields["output_token_logprobs_val"] = [[-0.5], None]
                    fields["output_token_logprobs_idx"] = [[1], None]
                RustServer.push_generation(bridge, SimpleNamespace(**fields))
                header, columns = captured[0]
                self.assertEqual(
                    header[:6],
                    [["r0", "r1"], [None, None], [10, 10], [1, 1], [0, 8], [0, 0]],
                )
                self.assertEqual(columns[0], array("q", [1, 2]).tobytes())
                if include_details:
                    self.assertEqual(len(header), 19)
                    self.assertEqual(header[18], details)
                    self.assertEqual(header[6], [1, 0] if logprobs else [])
                else:
                    self.assertEqual(len(header), 6)


if __name__ == "__main__":
    unittest.main()
