"""Fixed-batch ownership contract for the experimental SQD runner."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodePlan:
    request_ids: tuple[str, ...]
    prompt_lengths: tuple[int, ...]
    mla_layers: tuple[int, ...]
    hidden_size: int
    dtype: str
    tp_size: int
    dcp_size: int

    def __post_init__(self):
        if (
            not self.request_ids
            or len(set(self.request_ids)) != len(self.request_ids)
            or len(self.request_ids) != len(self.prompt_lengths)
            or any(length <= 0 for length in self.prompt_lengths)
        ):
            raise ValueError("SQD requires unique requests with nonempty prompts")
        if (
            not self.mla_layers
            or tuple(sorted(set(self.mla_layers))) != self.mla_layers
            or self.mla_layers[0] < 0
        ):
            raise ValueError("SQD requires increasing, unique MLA layer IDs")
        if (
            self.hidden_size <= 0
            or self.tp_size <= 0
            or self.dcp_size <= 0
            or self.tp_size % self.dcp_size
        ):
            raise ValueError("Invalid SQD dimensions or TP/DCP layout")

    def check_peer(self, peer: "DecodePlan"):
        if self != peer:
            raise ValueError(f"SQD peers disagree: local={self}, peer={peer}")
