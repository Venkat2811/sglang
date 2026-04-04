from __future__ import annotations

from pathlib import Path

from benchmarks.bfcl_subset import BfclWorkspace
from benchmarks.bfcl_subset import load_bfcl_subset_scenarios
from benchmarks.bfcl_subset import materialize_bfcl_scenario


def test_bfcl_subset_fixture_materialization_and_base1_tool_flow(tmp_path: Path):
    scenarios = load_bfcl_subset_scenarios()
    scenario = next(item for item in scenarios if item.id == "bfcl_multi_turn_base_1_subset")

    materialize_bfcl_scenario(tmp_path, scenario)
    workspace = BfclWorkspace(tmp_path)

    listing = workspace.execute(
        "fs_list_dir", {"path": "alex/workspace", "include_hidden": True}
    )
    assert [entry["name"] for entry in listing["entries"]] == [
        ".hidden_file",
        "archive",
        "log.txt",
    ]

    workspace.execute(
        "fs_move_path",
        {
            "source_path": "alex/workspace/log.txt",
            "destination_path": "alex/workspace/archive/log.txt",
        },
    )
    assert not (tmp_path / "alex/workspace/log.txt").exists()
    assert (tmp_path / "alex/workspace/archive/log.txt").exists()

    grep_result = workspace.execute(
        "fs_grep_text",
        {"path": "alex/workspace/archive/log.txt", "pattern": "Error"},
    )
    assert grep_result["matches"] == [
        {"line_number": 5, "line": "Error: Something went wrong."}
    ]

    tail_result = workspace.execute(
        "fs_tail_text",
        {"path": "alex/workspace/archive/log.txt", "line_count": 20},
    )
    assert tail_result["lines"][-1] == "Final line."
