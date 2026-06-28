from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BFCLEvalSample:
    id: str
    category: str
    source_file: str
    turns: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    expected_type: str
    expected: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_multi_turn(self) -> bool:
        return len(self.turns) > 1 or self.source_file.startswith("BFCL_v3_multi_turn")

    @property
    def messages(self) -> list[dict[str, Any]]:
        if self.is_multi_turn:
            raise ValueError("Use sample.turns for multi-turn samples")
        return self.turns[0]["messages"]


@dataclass(frozen=True, slots=True)
class BFCLRequest:
    sample: BFCLEvalSample
    turn_index: int
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]

    @property
    def is_multi_turn(self) -> bool:
        return self.sample.is_multi_turn


@dataclass(frozen=True, slots=True)
class MatchIssue:
    code: str
    message: str
    path: str = ""
    expected: Any = None
    predicted: Any = None


@dataclass(frozen=True, slots=True)
class TurnResult:
    turn_index: int
    passed: bool
    expected_num_calls: int
    predicted_num_calls: int
    issues: list[MatchIssue] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SampleResult:
    id: str
    category: str
    source_file: str
    expected_type: str
    is_multi_turn: bool
    passed: bool
    reason: str
    expected_num_calls: int
    predicted_num_calls: int
    issues: list[MatchIssue] = field(default_factory=list)
    turns_total: int | None = None
    turns_passed: int | None = None
    turn_accuracy: float | None = None
    turn_results: list[TurnResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
