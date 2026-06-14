from __future__ import annotations

import copy

import pytest

from config import ConfigError, load_config, validate_config


def test_example_config_is_full_valid_template():
    config = load_config("configs/config.example.yaml")

    assert config.reasoning["enable_thinking"] is False
    assert config.section("model")["name"] == "estadel-llm"
    assert config.section("model")["alias"] == "champion"
    assert config.section("model")["use_fp8_base"] is False
    assert config.section("tokenizer") == {
        "use_fast": True,
        "add_special_tokens": False,
        "padding_side": "right",
    }
    assert config.section("training")["num_epochs"] == 3
    assert config.pretokenized_dir == config.output_dir / "pretokenized"
    assert config.checkpoint_dir == config.output_dir / "checkpoints"
    assert config.bfcl_rows_path == config.output_dir / "eval" / "bfcl_rows.jsonl"
    assert config.section("registry")["selection"] == {"metric": "eval/loss", "mode": "min"}


def test_preprocessing_rejects_unknown_sections():
    raw = copy.deepcopy(load_config("configs/config.example.yaml").raw)
    raw["preprocessing"]["dvc"] = {"repo": "."}

    with pytest.raises(ConfigError, match="Unknown preprocessing fields"):
        validate_config(raw)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "source", {}),
        ("tokenizer", "source", "explicit"),
        ("loss_routing", "sampler_weights", {}),
        ("training", "max_steps", 100),
        ("checkpointing", "root_dir", "checkpoints"),
        ("mlflow", "experiment_name", "duplicate"),
        ("registry", "modelctl_path", "modelctl"),
    ],
)
def test_removed_config_fields_are_rejected(section, field, value):
    raw = copy.deepcopy(load_config("configs/config.example.yaml").raw)
    raw[section][field] = value

    with pytest.raises(ConfigError, match="Unknown"):
        validate_config(raw)


def test_bfcl_selection_requires_bfcl_eval():
    raw = copy.deepcopy(load_config("configs/config.example.yaml").raw)
    raw["eval"]["bfcl"]["enabled"] = False
    raw["registry"]["selection"] = {"metric": "eval/bfcl/accuracy", "mode": "max"}

    with pytest.raises(ConfigError, match="BFCL"):
        validate_config(raw)


def test_bfcl_selection_requires_bfcl_on_every_checkpoint_boundary():
    raw = copy.deepcopy(load_config("configs/config.example.yaml").raw)
    raw["registry"]["selection"] = {"metric": "eval/bfcl/accuracy", "mode": "max"}
    raw["eval"]["bfcl"]["run_every_n_validations"] = 2
    raw["checkpointing"]["save_every_n_validations"] = 1

    with pytest.raises(ConfigError, match="every checkpoint boundary"):
        validate_config(raw)


def test_eval_cadence_must_be_positive():
    raw = copy.deepcopy(load_config("configs/config.example.yaml").raw)
    raw["eval"]["every_train_steps"] = 0

    with pytest.raises(ConfigError, match="eval.every_train_steps"):
        validate_config(raw)


def test_adamw_betas_must_be_two_valid_floats():
    raw = copy.deepcopy(load_config("configs/config.example.yaml").raw)
    raw["training"]["adamw_betas"] = [0.9, 1.0]

    with pytest.raises(ConfigError, match="training.adamw_betas"):
        validate_config(raw)


def test_frozen_lm_head_cannot_be_lora_module_to_save():
    raw = copy.deepcopy(load_config("configs/config.example.yaml").raw)
    raw["lora"]["modules_to_save"] = ["lm_head"]

    with pytest.raises(ConfigError, match="freeze_lm_head"):
        validate_config(raw)


def test_cpu_ram_efficient_loading_requires_module_state_sync():
    raw = copy.deepcopy(load_config("configs/config.example.yaml").raw)
    raw["distributed"]["fsdp"]["sync_module_states"] = False

    with pytest.raises(ConfigError, match="requires sync_module_states=true"):
        validate_config(raw)
