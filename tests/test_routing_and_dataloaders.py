from __future__ import annotations

import pandas as pd
import pytest

from data.collators import RoutedCollator
from data.dataloaders import build_dataloaders, validate_training_inputs
from data.pretokenized_dataset import PretokenizedDataset
from data.routed_batch_sampler import RoutedBatchSampler
from config import Config
from conftest import example_config, pretok_result


def write_pretok(path) -> None:
    pd.DataFrame(
        [
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
                "sample_id": "dpo-0",
                "row_index": 2,
                "loss_kind": "dpo_target",
                "chosen_input_ids": [4, 5],
                "chosen_attention_mask": [1, 1],
                "chosen_labels": [-100, 5],
                "chosen_length": 2,
                "chosen_completion_token_count": 1,
                "chosen_render_hash": "chosen-hash",
                "chosen_ref_logp": -0.25,
                "rejected_input_ids": [6],
                "rejected_attention_mask": [1],
                "rejected_labels": [6],
                "rejected_length": 1,
                "rejected_completion_token_count": 1,
                "rejected_render_hash": "rejected-hash",
                "rejected_ref_logp": -0.75,
            },
        ]
    ).to_parquet(path, index=False)


def test_routed_sampler_batches_are_homogeneous_and_seeded() -> None:
    loss_kinds = ["sft_tool", "sft_target", "sft_tool", "dpo_target", "sft_target"]
    sampler = RoutedBatchSampler(loss_kinds, batch_size=2, seed=42)

    assert list(RoutedBatchSampler(loss_kinds, batch_size=1, seed=7)) == list(
        RoutedBatchSampler(loss_kinds, batch_size=1, seed=7)
    )
    for batch in sampler:
        assert len({loss_kinds[index] for index in batch}) == 1


def test_routed_sampler_drop_last_discards_short_route_batches() -> None:
    loss_kinds = ["sft_tool", "sft_tool", "sft_tool", "sft_target"]
    sampler = RoutedBatchSampler(loss_kinds, batch_size=2, drop_last=True, shuffle=False)

    assert list(sampler) == [[0, 1]]


def test_routed_sampler_reshuffles_inside_route_per_epoch() -> None:
    loss_kinds = ["sft_target"] * 8
    sampler = RoutedBatchSampler(loss_kinds, batch_size=2, seed=5, shuffle=True)
    epoch_zero = list(sampler)

    sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert epoch_zero != epoch_one
    assert sorted(index for batch in epoch_one for index in batch) == list(range(8))


def test_routed_sampler_groups_batches_by_replica_route() -> None:
    loss_kinds = ["dpo_target", "sft_target", "sft_tool", "dpo_target", "sft_target"]
    sampler = RoutedBatchSampler(
        loss_kinds,
        batch_size=1,
        seed=0,
        shuffle=False,
        replica_group_size=2,
    )
    batches = list(sampler)

    assert len(batches) % 2 == 0
    for start in range(0, len(batches), 2):
        route_pair = {loss_kinds[batch[0]] for batch in batches[start : start + 2]}
        assert len(route_pair) == 1

    assert sampler.summary()["num_padded_replica_batches"] == 1


def test_routed_collator_pads_sft_and_dpo() -> None:
    collator = RoutedCollator(pad_token_id=0, ignore_index=-100)

    sft = collator(
        [
            {"loss_kind": "sft_target", "input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2], "sample_id": "a", "row_index": 0},
            {"loss_kind": "sft_target", "input_ids": [3], "attention_mask": [1], "labels": [3], "sample_id": "b", "row_index": 1},
        ]
    )
    assert sft["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert sft["labels"].tolist() == [[-100, 2], [3, -100]]

    dpo = collator(
        [
            {
                "loss_kind": "dpo_target",
                "sample_id": "a",
                "row_index": 0,
                "chosen_input_ids": [1, 2],
                "chosen_attention_mask": [1, 1],
                "chosen_labels": [-100, 2],
                "rejected_input_ids": [3],
                "rejected_attention_mask": [1],
                "rejected_labels": [3],
            },
            {
                "loss_kind": "dpo_target",
                "sample_id": "b",
                "row_index": 1,
                "chosen_input_ids": [4],
                "chosen_attention_mask": [1],
                "chosen_labels": [4],
                "rejected_input_ids": [5, 6],
                "rejected_attention_mask": [1, 1],
                "rejected_labels": [-100, 6],
            },
        ]
    )
    assert dpo["chosen_input_ids"].tolist() == [[1, 2], [4, 0]]
    assert dpo["rejected_labels"].tolist() == [[3, -100], [-100, 6]]


def test_routed_collator_rejects_mixed_loss_kind_batch() -> None:
    collator = RoutedCollator(pad_token_id=0, ignore_index=-100)
    with pytest.raises(ValueError, match="homogeneous"):
        collator(
            [
                {"loss_kind": "sft_target", "input_ids": [1], "attention_mask": [1], "labels": [1], "sample_id": "a"},
                {"loss_kind": "sft_tool", "input_ids": [2], "attention_mask": [1], "labels": [2], "sample_id": "b"},
            ]
        )


def test_pretokenized_dataset_normalizes_dpo_ref_logps(tmp_path) -> None:
    path = tmp_path / "train.parquet"
    write_pretok(path)

    dataset = PretokenizedDataset.from_parquet(path, split="train")
    dpo = next(row for row in dataset.rows if row["loss_kind"] == "dpo_target")

    assert dpo["chosen_ref_logp"] == -0.25
    assert dpo["rejected_ref_logp"] == -0.75
    assert dataset.loss_kind_counts == {"sft_target": 1, "sft_tool": 1, "dpo_target": 1}


def test_build_dataloaders_from_pretokenized_parquet(tmp_path) -> None:
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    write_pretok(train_path)
    write_pretok(valid_path)
    config = example_config(training={"per_device_train_batch_size": 1})

    bundle = build_dataloaders(
        config,
        [pretok_result("train", train_path), pretok_result("valid", valid_path)],
        pad_token_id=0,
    )

    assert set(bundle.splits) == {"train", "valid"}
    assert bundle["train"].summary["loss_kind_counts"] == {"dpo_target": 1, "sft_target": 1, "sft_tool": 1}
    validate_training_inputs(config, bundle)
    for split_loader in bundle.splits.values():
        for batch in split_loader.dataloader:
            assert batch["loss_kind"] in {"sft_target", "sft_tool", "dpo_target"}


def test_build_dataloaders_aligns_route_batches_for_distributed_sharding(tmp_path) -> None:
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    write_pretok(train_path)
    write_pretok(valid_path)
    config = example_config(training={"per_device_train_batch_size": 1})

    bundle = build_dataloaders(
        config,
        [pretok_result("train", train_path), pretok_result("valid", valid_path)],
        pad_token_id=0,
        num_processes=2,
    )

    summary = bundle["train"].summary
    assert summary["replica_group_size"] == 2
    assert summary["num_padded_replica_batches"] == 3
    batches = list(bundle["train"].sampler)
    for start in range(0, len(batches), 2):
        route_pair = {bundle["train"].dataset.loss_kinds[batch[0]] for batch in batches[start : start + 2]}
        assert len(route_pair) == 1


def test_validate_training_inputs_rejects_unconfigured_data_route(tmp_path) -> None:
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    write_pretok(train_path)
    write_pretok(valid_path)
    config_data = example_config(training={"per_device_train_batch_size": 1}).to_dict()
    del config_data["loss_routing"]["routes"]["dpo_target"]
    config = Config.from_dict(config_data)
    bundle = build_dataloaders(
        config,
        [pretok_result("train", train_path), pretok_result("valid", valid_path)],
        pad_token_id=0,
    )

    with pytest.raises(ValueError, match="missing from config"):
        validate_training_inputs(config, bundle)
