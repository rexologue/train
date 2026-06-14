from __future__ import annotations

import pytest

from trainer import modeling


def test_flash_attention_runtime_preflight_fails_before_model_load(monkeypatch):
    monkeypatch.setattr(modeling, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match="requires the flash_attn package"):
        modeling.validate_model_runtime_requirements({"attn_implementation": "flash_attention_2"})


def test_flash_attention_runtime_preflight_accepts_installed_package(monkeypatch):
    monkeypatch.setattr(modeling, "find_spec", lambda name: object())

    modeling.validate_model_runtime_requirements({"attn_implementation": "flash_attention_2"})
