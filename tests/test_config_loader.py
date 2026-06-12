from __future__ import annotations

import copy

import pytest

from config import ConfigError, load_config, validate_config


def test_preprocess_config_loads_and_configures_reasoning_switch():
    config = load_config("configs/config.preprocess.yaml")
    assert config.reasoning["enable_thinking"] is False
    assert config.section("model")["use_fp8_base"] is False
    assert config.sequence["truncation"] is False
    assert config.preprocessing["output"]["debug_examples_per_loss_kind"] == 5


def test_example_config_is_full_valid_template():
    config = load_config("configs/config.example.yaml")
    assert config.section("project")["name"] == "dummy-qwen35-a3b-run"
    assert config.section("tokenizer")["tokenizer_id"] == "/path/to/tokenizer"
    assert config.section("dpo")["candidates_path"] == "data/dpo/candidates.jsonl"
    assert config.section("loss_routing")["routes"]["dpo_target"]["type"] == "dpo"


def test_preprocessing_rejects_unknown_sections():
    raw = copy.deepcopy(load_config("configs/config.preprocess.yaml").raw)
    raw["preprocessing"]["dvc"] = {"repo": "."}

    with pytest.raises(ConfigError, match="Unknown preprocessing config sections"):
        validate_config(raw)
