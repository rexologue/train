from __future__ import annotations

import copy

import pytest

from config import ConfigError, load_config, validate_config


def test_preprocess_config_loads_and_configures_reasoning_switch():
    config = load_config("configs/config.preprocess.yaml")
    assert config.reasoning["enable_thinking"] is False
    assert config.section("model")["use_fp8_base"] is False
    assert config.sequence["truncation"] is False
    assert config.section("training")["enabled"] is False
    assert config.section("training")["drop_last"] is False
    assert config.section("training")["adamw_betas"] == [0.9, 0.999]
    assert config.section("distributed")["fsdp"]["transformer_cls_names_to_wrap"] == ["Qwen3_5MoeDecoderLayer"]
    assert config.preprocessing["output"]["debug_examples_per_loss_kind"] == 5


def test_example_config_is_full_valid_template():
    config = load_config("configs/config.example.yaml")
    assert config.section("project")["name"] == "dummy-qwen35-a3b-run"
    assert config.section("tokenizer")["source"] == "model"
    assert config.section("tokenizer")["tokenizer_id"] is None
    assert "dpo_target" not in config.section("loss_routing")["routes"]
    assert config.section("training")["enabled"] is True
    assert config.section("training")["adamw_betas"] == [0.9, 0.999]
    assert config.section("eval")["every_train_steps"] == 25
    assert config.section("checkpointing")["save_every_n_validations"] == 1
    assert config.section("checkpointing")["save_total_limit"] == 3
    assert config.section("mlflow")["async_logging"]["enabled"] is True


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


def test_eval_cadence_must_be_positive():
    raw = copy.deepcopy(load_config("configs/config.preprocess.yaml").raw)
    raw["eval"]["every_train_steps"] = 0

    with pytest.raises(ConfigError, match="eval.every_train_steps"):
        validate_config(raw)


def test_mlflow_async_logging_flag_must_be_boolean():
    raw = copy.deepcopy(load_config("configs/config.preprocess.yaml").raw)
    raw["mlflow"]["async_logging"]["enabled"] = "yes"

    with pytest.raises(ConfigError, match="mlflow.async_logging.enabled"):
        validate_config(raw)


def test_training_lora_whitelist_must_not_be_empty_when_training_enabled():
    raw = copy.deepcopy(load_config("configs/config.preprocess.yaml").raw)
    raw["training"]["enabled"] = True
    raw["lora"]["target_modules"] = []

    with pytest.raises(ConfigError, match="lora.target_modules"):
        validate_config(raw)


def test_adamw_betas_must_be_two_valid_floats():
    raw = copy.deepcopy(load_config("configs/config.preprocess.yaml").raw)
    raw["training"]["adamw_betas"] = [0.9, 1.0]

    with pytest.raises(ConfigError, match="training.adamw_betas"):
        validate_config(raw)


def test_frozen_lm_head_cannot_be_lora_module_to_save():
    raw = copy.deepcopy(load_config("configs/config.preprocess.yaml").raw)
    raw["training"]["enabled"] = True
    raw["lora"]["modules_to_save"] = ["lm_head"]

    with pytest.raises(ConfigError, match="freeze_lm_head"):
        validate_config(raw)


def test_cpu_ram_efficient_loading_requires_module_state_sync():
    raw = copy.deepcopy(load_config("configs/config.preprocess.yaml").raw)
    raw["distributed"]["fsdp"]["cpu_ram_efficient_loading"] = True
    raw["distributed"]["fsdp"]["sync_module_states"] = False

    with pytest.raises(ConfigError, match="requires sync_module_states=true"):
        validate_config(raw)
