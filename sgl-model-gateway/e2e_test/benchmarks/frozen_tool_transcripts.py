from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DATA_PATH = (
    Path(__file__).resolve().parent / "data" / "frozen_tool_transcript_subset.json"
)


@dataclass(frozen=True)
class FrozenToolTranscriptTurn:
    turn_id: str
    call_id: str
    tool_name: str
    tool_arguments: dict[str, Any]
    tool_output: Any
    user_text: str
    max_output_tokens: int


@dataclass(frozen=True)
class FrozenToolTranscriptScenario:
    id: str
    source_dataset: str
    source_id: str
    description: str
    seed_prompt: str
    turns: tuple[FrozenToolTranscriptTurn, ...]


def load_frozen_tool_transcript_scenarios() -> list[FrozenToolTranscriptScenario]:
    raw_payload = json.loads(_DATA_PATH.read_text())
    return [
        FrozenToolTranscriptScenario(
            id=str(raw["id"]),
            source_dataset=str(raw["source_dataset"]),
            source_id=str(raw["source_id"]),
            description=str(raw["description"]),
            seed_prompt=str(raw["seed_prompt"]),
            turns=tuple(
                FrozenToolTranscriptTurn(
                    turn_id=str(turn["turn_id"]),
                    call_id=str(turn["call_id"]),
                    tool_name=str(turn["tool_name"]),
                    tool_arguments=dict(turn["tool_arguments"]),
                    tool_output=turn["tool_output"],
                    user_text=str(turn["user_text"]),
                    max_output_tokens=int(turn["max_output_tokens"]),
                )
                for turn in raw["turns"]
            ),
        )
        for raw in raw_payload
    ]
