from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

LossKind = Literal["sft_target", "sft_tool", "dpo_target"]
Split = Literal["train", "valid", "test"]


@dataclass(frozen=True)
class CanonicalRow:
    sample_id: str
    split: Split
    loss_kind: LossKind
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    parallel_tool_calls: bool | None = None
    prompt: list[dict[str, Any]] | None = None
    chosen: dict[str, Any] | None = None
    rejected: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssistantSpan:
    message_index: int
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class TargetSpan:
    message_index: int
    start: int
    end: int
    reason: str


@dataclass(frozen=True)
class RenderedSample:
    rendered_text: str
    assistant_spans: list[AssistantSpan]
    render_metadata: dict[str, Any]

