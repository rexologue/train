from __future__ import annotations

import pytest

from registry.modelctl_client import build_modelctl_info_args, build_modelctl_pull_args, build_modelctl_register_args
from registry.tags import candidate_alias, validate_training_aliases


def test_candidate_alias_is_explicit_and_never_champion():
    assert candidate_alias("candidate-{candidate_index:06d}", 1) == "candidate-000001"
    with pytest.raises(ValueError):
        validate_training_aliases(["candidate-000001", "champion"])


def test_modelctl_register_args_match_vendored_cli_contract():
    args = build_modelctl_register_args(
        "modelctl",
        "qwen35",
        "checkpoints/best",
        ["candidate-000001", "candidate-latest"],
        tracking_uri="http://mlflow:5000",
        training_tags_json="training.json",
    )

    assert args == [
        "modelctl",
        "register",
        "checkpoints/best",
        "qwen35",
        "--kind",
        "generic",
        "--alias",
        "candidate-000001",
        "--alias",
        "candidate-latest",
        "--tracking-uri",
        "http://mlflow:5000",
        "--training-tags-json",
        "training.json",
    ]


def test_modelctl_pull_and_info_args_match_vendored_cli_contract():
    assert build_modelctl_pull_args("modelctl", "models:/qwen35@champion", "models/qwen35") == [
        "modelctl",
        "pull",
        "models:/qwen35@champion",
        "models/qwen35",
    ]
    assert build_modelctl_info_args("modelctl", "models:/qwen35/3", tracking_uri="http://mlflow:5000") == [
        "modelctl",
        "info",
        "models:/qwen35/3",
        "--tracking-uri",
        "http://mlflow:5000",
    ]
