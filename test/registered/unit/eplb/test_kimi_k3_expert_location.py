"""Kimi-K3 declares its expert layout for EPLB and the expert-distribution
recorder, and every top-k call in its MoE carries the EPLB dispatch info.

The model module needs CUDA kernels to import, so the structural checks read the
source; the config arithmetic runs the classmethod bodies on a stand-in config.
"""

import ast
import types
import unittest
from pathlib import Path

MODEL_FILE = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/models/kimi_k3.py"
)


def _class_node(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _has_classmethod(cls_node, name):
    for node in cls_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return any(
                isinstance(d, ast.Name) and d.id == "classmethod"
                for d in node.decorator_list
            )
    return False


class TestKimiK3ExpertLocation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = MODEL_FILE.read_text()
        cls.tree = ast.parse(cls.source)

    def test_entry_classes_declare_expert_location(self):
        for name in ("KimiK3LinearForCausalLM", "KimiK3ForConditionalGeneration"):
            self.assertTrue(
                _has_classmethod(
                    _class_node(self.tree, name), "get_model_config_for_expert_location"
                ),
                name,
            )

    def test_every_moe_topk_call_passes_dispatch_info(self):
        moe = _class_node(self.tree, "KimiK3MoE")
        calls = [
            node
            for node in ast.walk(moe)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "topk"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ]
        self.assertGreaterEqual(len(calls), 4)
        for call in calls:
            self.assertIn(
                "expert_location_dispatch_info",
                [kw.arg for kw in call.keywords],
                ast.get_source_segment(self.source, call),
            )

    def test_physical_experts_include_redundant_experts(self):
        self.assertIn("ep_num_redundant_experts", self.source)
        self.assertIn(
            'num_experts=getattr(config, "n_routed_experts", config.num_experts)\n'
            "            + get_exec().moe.ep_num_redundant_experts",
            self.source,
        )

    def test_classmethod_arithmetic_on_the_k3_config(self):
        # Run the two classmethod bodies as plain functions, without importing
        # the model module (it needs CUDA kernels).
        from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

        def body(class_name):
            node = next(
                n
                for n in _class_node(self.tree, class_name).body
                if isinstance(n, ast.FunctionDef)
                and n.name == "get_model_config_for_expert_location"
            )
            for arg in node.args.args:
                arg.annotation = None
            node = ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.FunctionDef(
                            name="fn",
                            args=node.args,
                            body=node.body,
                            decorator_list=[],
                            returns=None,
                        )
                    ],
                    type_ignores=[],
                )
            )
            return node

        text = types.SimpleNamespace(
            num_hidden_layers=93, num_experts=896, num_expert_group=1
        )
        ns = {"ModelConfigForExpertLocation": ModelConfigForExpertLocation}
        exec(compile(body("KimiK3LinearForCausalLM"), "<text>", "exec"), ns)
        text_fn = ns["fn"]
        got = text_fn(None, text)
        self.assertEqual(
            (got.num_layers, got.num_logical_experts, got.num_groups), (93, 896, 1)
        )

        ns2 = {
            "KimiK3LinearForCausalLM": types.SimpleNamespace(
                get_model_config_for_expert_location=lambda cfg: text_fn(None, cfg)
            )
        }
        exec(compile(body("KimiK3ForConditionalGeneration"), "<wrapper>", "exec"), ns2)
        got = ns2["fn"](None, types.SimpleNamespace(text_config=text))
        self.assertEqual(
            (got.num_layers, got.num_logical_experts, got.num_groups), (93, 896, 1)
        )


if __name__ == "__main__":
    unittest.main()
