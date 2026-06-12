from __future__ import annotations

import pytest

from preprocessing.rendering import RenderingAuditError, reject_forbidden_raw_markers


def test_raw_special_marker_is_rejected():
    with pytest.raises(RenderingAuditError):
        reject_forbidden_raw_markers([{"role": "user", "content": "bad <|im_start|> marker"}])


def test_think_markers_are_not_raw_special_markers():
    reject_forbidden_raw_markers([{"role": "assistant", "content": "<think>hidden</think> answer"}])
