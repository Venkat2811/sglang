from __future__ import annotations

import difflib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BFCL_MULTI_TURN_SUBSET_PATH = (
    Path(__file__).resolve().parent / "data" / "bfcl_multi_turn_subset.json"
)


@dataclass(frozen=True)
class BfclTurn:
    prompt: str
    expected_tool: str


@dataclass(frozen=True)
class BfclScenario:
    id: str
    source_dataset: str
    source_id: str
    description: str
    directories: tuple[str, ...]
    files: tuple[dict[str, str], ...]
    turns: tuple[BfclTurn, ...]


def load_bfcl_subset_scenarios() -> list[BfclScenario]:
    raw_scenarios = json.loads(BFCL_MULTI_TURN_SUBSET_PATH.read_text())
    return [
        BfclScenario(
            id=raw["id"],
            source_dataset=raw["source_dataset"],
            source_id=raw["source_id"],
            description=raw["description"],
            directories=tuple(raw["directories"]),
            files=tuple(raw["files"]),
            turns=tuple(BfclTurn(**turn) for turn in raw["turns"]),
        )
        for raw in raw_scenarios
    ]


def materialize_bfcl_scenario(root: Path, scenario: BfclScenario) -> None:
    for directory in scenario.directories:
        _resolve_workspace_path(root, directory).mkdir(parents=True, exist_ok=True)

    for file_spec in scenario.files:
        path = _resolve_workspace_path(root, file_spec["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file_spec["content"])


def bfcl_subset_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "fs_list_dir",
            "description": "List visible or hidden entries in a directory relative to the scenario root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "include_hidden": {"type": "boolean"},
                },
                "required": ["path"],
            },
        },
        {
            "type": "function",
            "name": "fs_move_path",
            "description": "Move a file or directory from one relative path to another inside the scenario root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_path": {"type": "string"},
                    "destination_path": {"type": "string"},
                },
                "required": ["source_path", "destination_path"],
            },
        },
        {
            "type": "function",
            "name": "fs_grep_text",
            "description": "Search a text file for lines containing a pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                },
                "required": ["path", "pattern"],
            },
        },
        {
            "type": "function",
            "name": "fs_tail_text",
            "description": "Return the last N lines from a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line_count": {"type": "integer"},
                },
                "required": ["path", "line_count"],
            },
        },
        {
            "type": "function",
            "name": "fs_touch_file",
            "description": "Create an empty file if it does not already exist.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "type": "function",
            "name": "fs_write_text",
            "description": "Write text to a file. When append is false, replace existing content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "type": "function",
            "name": "fs_diff_text",
            "description": "Return a unified diff between two text files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "left_path": {"type": "string"},
                    "right_path": {"type": "string"},
                },
                "required": ["left_path", "right_path"],
            },
        },
        {
            "type": "function",
            "name": "fs_copy_path",
            "description": "Copy a file from one relative path to another inside the scenario root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_path": {"type": "string"},
                    "destination_path": {"type": "string"},
                },
                "required": ["source_path", "destination_path"],
            },
        },
        {
            "type": "function",
            "name": "fs_read_text",
            "description": "Read and return the full contents of a text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    ]


class BfclWorkspace:
    def __init__(self, root: Path):
        self.root = root

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "fs_list_dir": self._list_dir,
            "fs_move_path": self._move_path,
            "fs_grep_text": self._grep_text,
            "fs_tail_text": self._tail_text,
            "fs_touch_file": self._touch_file,
            "fs_write_text": self._write_text,
            "fs_diff_text": self._diff_text,
            "fs_copy_path": self._copy_path,
            "fs_read_text": self._read_text,
        }
        if tool_name not in handlers:
            raise ValueError(f"unsupported BFCL subset tool: {tool_name}")
        return handlers[tool_name](**arguments)

    def list_files(self) -> list[str]:
        files = [path.relative_to(self.root).as_posix() for path in self.root.rglob("*")]
        return sorted(files)

    def _list_dir(self, path: str, include_hidden: bool = True) -> dict[str, Any]:
        target = self._existing_path(path)
        if not target.is_dir():
            raise ValueError(f"path is not a directory: {path}")
        entries = []
        for child in sorted(target.iterdir(), key=lambda value: value.name):
            if not include_hidden and child.name.startswith("."):
                continue
            entries.append(
                {
                    "name": child.name,
                    "type": "directory" if child.is_dir() else "file",
                }
            )
        return {"path": path, "entries": entries}

    def _move_path(self, source_path: str, destination_path: str) -> dict[str, Any]:
        source = self._existing_path(source_path)
        destination = _resolve_workspace_path(self.root, destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return {"source_path": source_path, "destination_path": destination_path}

    def _grep_text(self, path: str, pattern: str) -> dict[str, Any]:
        target = self._existing_path(path)
        matches = []
        for line_number, line in enumerate(target.read_text().splitlines(), start=1):
            if pattern in line:
                matches.append({"line_number": line_number, "line": line})
        return {"path": path, "pattern": pattern, "matches": matches}

    def _tail_text(self, path: str, line_count: int) -> dict[str, Any]:
        target = self._existing_path(path)
        lines = target.read_text().splitlines()
        return {
            "path": path,
            "line_count": line_count,
            "lines": lines[-line_count:],
        }

    def _touch_file(self, path: str) -> dict[str, Any]:
        target = _resolve_workspace_path(self.root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=True)
        return {"path": path}

    def _write_text(self, path: str, content: str, append: bool = False) -> dict[str, Any]:
        target = _resolve_workspace_path(self.root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if append:
            existing = target.read_text() if target.exists() else ""
            target.write_text(f"{existing}{content}")
        else:
            target.write_text(content)
        return {"path": path, "bytes_written": len(content), "append": append}

    def _diff_text(self, left_path: str, right_path: str) -> dict[str, Any]:
        left = self._existing_path(left_path).read_text().splitlines()
        right = self._existing_path(right_path).read_text().splitlines()
        diff = list(
            difflib.unified_diff(
                left,
                right,
                fromfile=left_path,
                tofile=right_path,
                lineterm="",
            )
        )
        return {"left_path": left_path, "right_path": right_path, "diff": diff}

    def _copy_path(self, source_path: str, destination_path: str) -> dict[str, Any]:
        source = self._existing_path(source_path)
        destination = _resolve_workspace_path(self.root, destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return {"source_path": source_path, "destination_path": destination_path}

    def _read_text(self, path: str) -> dict[str, Any]:
        target = self._existing_path(path)
        return {"path": path, "content": target.read_text()}

    def _existing_path(self, path: str) -> Path:
        target = _resolve_workspace_path(self.root, path)
        if not target.exists():
            raise ValueError(f"path does not exist: {path}")
        return target


def _resolve_workspace_path(root: Path, relative_path: str) -> Path:
    if not relative_path:
        raise ValueError("empty workspace path")
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes BFCL workspace root: {relative_path}") from exc
    return candidate
