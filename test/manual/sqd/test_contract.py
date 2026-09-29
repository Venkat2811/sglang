import unittest

from sglang.benchmark.sqd_contract import DecodePlan


class TestDecodePlan(unittest.TestCase):
    def plan(self, **changes):
        fields = dict(
            request_ids=("a", "b"),
            prompt_lengths=(17, 31),
            mla_layers=(3, 7),
            hidden_size=2048,
            dtype="torch.bfloat16",
            tp_size=4,
            dcp_size=2,
        )
        return DecodePlan(**(fields | changes))

    def test_rejects_incompatible_peer_before_tensor_transfer(self):
        plan = self.plan()
        plan.check_peer(self.plan())
        for change in (
            {"request_ids": ("b", "a")},
            {"prompt_lengths": (31, 17)},
            {"mla_layers": (3, 8)},
            {"hidden_size": 4096},
            {"dtype": "torch.float16"},
            {"tp_size": 2},
            {"dcp_size": 1},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                plan.check_peer(self.plan(**change))

    def test_rejects_ambiguous_or_invalid_plan(self):
        for change in (
            {"request_ids": ("a", "a")},
            {"prompt_lengths": (17,)},
            {"prompt_lengths": (17, 0)},
            {"mla_layers": ()},
            {"mla_layers": (7, 3)},
            {"mla_layers": (3, 3)},
            {"mla_layers": (-1, 3)},
            {"hidden_size": 0},
            {"tp_size": 0},
            {"dcp_size": 3},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.plan(**change)


if __name__ == "__main__":
    unittest.main()
