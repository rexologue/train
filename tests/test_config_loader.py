from __future__ import annotations

import copy

import pytest

from config import ConfigError, load_config, validate_config


def test_preprocess_config_loads_and_configures_reasoning_switch():
    config = load_config("configs/config.preprocess.yaml")
    assert config.reasoning["enable_thinking"] is False
    assert config.section("model")["use_fp8_base"] is False
    assert config.sequence["truncation"] is False
    assert config.section("training")["drop_last"] is False
    assert config.preprocessing["output"]["debug_examples_per_loss_kind"] == 5


def test_example_config_is_full_valid_template():
    config = load_config("configs/config.example.yaml")
    assert config.section("project")["name"] == "dummy-qwen35-a3b-run"
    assert config.section("tokenizer")["source"] == "model"
    assert config.section("tokenizer")["tokenizer_id"] is None
    assert config.section("dpo")["candidates_path"] == "data/dpo/candidates.jsonl"
    assert config.section("loss_routing")["routes"]["dpo_target"]["type"] == "dpo"


def test_preprocessing_rejects_unknown_sections():
    raw = copy.deepcopy(load_config("configs/config.preprocess.yaml").raw)
    raw["preprocessing"]["dvc"] = {"repo": "."}

    with pytest.raises(ConfigError, match="Unknown preprocessing config sections"):
        validate_config(raw)


def test_registry_model_source_requires_exactly_one_alias_or_version():
    raw = copy.deepcopy(load_config("configs/config.preprocess.yaml").raw)
    raw["model"]["source"] = {
        "kind": "registry",
        "model_name": "qwen35",
        "alias": "champion",
        "version": "3",
        "local_dir": "artifacts/model_cache/qwen35/champion",
        "pull_policy": "if_local_empty",
        "verify_local_hash": True,
        "verify_remote_ref": False,
        "require_registry_metadata": True,
    }

    with pytest.raises(ConfigError, match="exactly one of alias or version"):
        validate_config(raw)


def test_explicit_tokenizer_source_requires_tokenizer_id():
    raw = copy.deepcopy(load_config("configs/config.preprocess.yaml").raw)
    raw["tokenizer"]["source"] = "explicit"
    raw["tokenizer"]["tokenizer_id"] = None

    with pytest.raises(ConfigError, match="tokenizer.tokenizer_id"):
        validate_config(raw)
