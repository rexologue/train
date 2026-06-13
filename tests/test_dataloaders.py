from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import load_config
from data.dataloaders import build_dataloaders
from data.inspection import inspect_random_batch
from preprocessing.io import PretokSplitResult


def _pretok_result(split: str, path: Path) -> PretokSplitResult:
    return PretokSplitResult(
        split=split,
        raw_path=path,
        output_dir=path.parent,
        pretok_path=path,
        manifest_path=path.parent / "manifest.json",
        reused=False,
        manifest={},
    )


def _write_pretok(path: Path) -> None:
    rows = [
        {
            "sample_id": "target-0",
            "row_index": 0,
            "loss_kind": "sft_target",
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "labels": [-100, 2],
            "length": 2,
            "num_supervised_tokens": 1,
        },
        {
            "sample_id": "tool-0",
            "row_index": 1,
            "loss_kind": "sft_tool",
            "input_ids": [3],
            "attention_mask": [1],
            "labels": [3],
            "length": 1,
            "num_supervised_tokens": 1,
        },
        {
            "sample_id": "target-1",
            "row_index": 2,
            "loss_kind": "sft_target",
            "input_ids": [4],
            "attention_mask": [1],
            "labels": [4],
            "length": 1,
            "num_supervised_tokens": 1,
        },
        {
            "sample_id": "dpo-0",
            "row_index": 3,
            "loss_kind": "dpo_target",
            "chosen_input_ids": [5, 6],
            "chosen_attention_mask": [1, 1],
            "chosen_labels": [-100, 6],
            "chosen_length": 2,
            "chosen_completion_token_count": 1,
            "rejected_input_ids": [7],
            "rejected_attention_mask": [1],
            "rejected_labels": [7],
            "rejected_length": 1,
            "rejected_completion_token_count": 1,
        },
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_build_dataloaders_requires_train_and_valid(tmp_path):
    valid_path = tmp_path / "valid.parquet"
    _write_pretok(valid_path)
    config = load_config("configs/config.preprocess.yaml")

    with pytest.raises(ValueError, match="missing=\\['train'\\]"):
        build_dataloaders(config, [_pretok_result("valid", valid_path)], pad_token_id=0)


def test_build_dataloaders_from_pretokenized_parquet(tmp_path):
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    _write_pretok(train_path)
    _write_pretok(valid_path)
    config = load_config("configs/config.preprocess.yaml")
    config.raw["training"]["per_device_train_batch_size"] = 2
    config.raw["training"]["drop_last"] = False

    bundle = build_dataloaders(
        config,
        [_pretok_result("train", train_path), _pretok_result("valid", valid_path)],
        pad_token_id=0,
    )

    assert set(bundle.splits) == {"train", "valid"}
    assert bundle["train"].summary["loss_kind_counts"] == {"dpo_target": 1, "sft_target": 2, "sft_tool": 1}
    for split_loader in bundle.splits.values():
        for batch in split_loader.dataloader:
            assert batch["loss_kind"] in {"sft_target", "sft_tool", "dpo_target"}


def test_inspect_random_batch_reports_shapes_and_element(tmp_path):
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    _write_pretok(train_path)
    _write_pretok(valid_path)
    config = load_config("configs/config.preprocess.yaml")
    config.raw["training"]["per_device_train_batch_size"] = 2

    bundle = build_dataloaders(
        config,
        [_pretok_result("train", train_path), _pretok_result("valid", valid_path)],
        pad_token_id=0,
    )
    report = inspect_random_batch(bundle, split="train", seed=1, token_limit=1)

    assert report["split"] == "train"
    assert report["keys"]
    assert report["shapes"]
    assert report["element"]
    tensor_previews = [value for value in report["element"].values() if isinstance(value, dict) and value.get("num_values")]
    assert tensor_previews
    assert all(len(preview["values"]) <= 1 for preview in tensor_previews)
