from __future__ import annotations

from qwen35_tuning.config.loader import load_config


def test_smoke_config_loads_and_enforces_reasoning_disabled():
    config = load_config("configs/smoke.yaml")
    assert config.reasoning["enabled"] is False
    assert config.section("model")["use_fp8_base"] is False
    assert config.section("data")["truncation"] is False


def test_dpo_config_extends_smoke():
    config = load_config("configs/dpo_qwen35_a3b.yaml")
    assert config.section("dpo")["enabled"] is True
    assert config.section("loss_routing")["routes"]["dpo_target"]["enabled"] is True

