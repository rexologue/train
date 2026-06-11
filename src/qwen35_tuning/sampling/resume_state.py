from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplerResumeState:
    epoch: int
    batch_offset: int
    consumed_sample_ids: list[str]

