from __future__ import annotations

import pytest

from qwen35_tuning.config.loader import load_config
from qwen35_tuning.rendering.audit import RenderingAuditError, reject_forbidden_raw_markers
from qwen35_tuning.rendering.reasoning import ReasoningAuditError, audit_reasoning


def test_nonempty_think_block_fails_when_reasoning_disabled():
    config = load_config("configs/smoke.yaml")
    with pytest.raises(ReasoningAuditError):
        audit_reasoning("a <think>hidden reasoning</think> b", [(0, 35)], config.reasoning)


def test_raw_special_marker_is_rejected():
    with pytest.raises(RenderingAuditError):
        reject_forbidden_raw_markers([{"role": "user", "content": "bad <|im_start|> marker"}])

