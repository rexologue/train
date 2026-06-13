from __future__ import annotations

import pytest

from data.collators import RoutedCollator
from data.routed_batch_sampler import RoutedBatchSampler


def test_routed_sampler_batches_are_homogeneous():
    loss_kinds = ["sft_tool", "sft_target", "sft_tool", "dpo_target"]
    sampler = RoutedBatchSampler(loss_kinds, batch_size=2, seed=42)
    for batch in sampler:
        assert len({loss_kinds[index] for index in batch}) == 1


def test_routed_sampler_chunks_each_loss_kind_before_shuffle():
    loss_kinds = ["sft_tool", "sft_target", "sft_tool", "dpo_target", "sft_tool", "sft_target"]
    sampler = RoutedBatchSampler(loss_kinds, batch_size=2, shuffle=False)

    assert list(sampler) == [[0, 2], [4], [1, 5], [3]]


def test_routed_sampler_drop_last_discards_short_route_batches():
    loss_kinds = ["sft_tool", "sft_tool", "sft_tool", "sft_target"]
    sampler = RoutedBatchSampler(loss_kinds, batch_size=2, drop_last=True, shuffle=False)

    assert list(sampler) == [[0, 1]]


def test_routed_sampler_shuffle_is_seeded():
    loss_kinds = ["sft_tool", "sft_tool", "sft_tool", "sft_tool", "sft_target", "sft_target", "dpo_target", "dpo_target"]

    assert list(RoutedBatchSampler(loss_kinds, batch_size=1, seed=7)) == list(RoutedBatchSampler(loss_kinds, batch_size=1, seed=7))


def test_collator_rejects_mixed_loss_kind_batch():
    collator = RoutedCollator(pad_token_id=0, ignore_index=-100)
    with pytest.raises(ValueError, match="homogeneous"):
        collator(
            [
                {"loss_kind": "sft_target", "input_ids": [1], "attention_mask": [1], "labels": [1], "sample_id": "a"},
                {"loss_kind": "sft_tool", "input_ids": [2], "attention_mask": [1], "labels": [2], "sample_id": "b"},
            ]
        )


def test_routed_collator_pads_sft_to_longest_in_batch():
    collator = RoutedCollator(pad_token_id=0, ignore_index=-100)

    batch = collator(
        [
            {"loss_kind": "sft_target", "input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2], "sample_id": "a"},
            {"loss_kind": "sft_target", "input_ids": [3], "attention_mask": [1], "labels": [3], "sample_id": "b"},
        ]
    )

    assert batch["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]


def test_routed_collator_pads_dpo_sides_independently():
    collator = RoutedCollator(pad_token_id=0, ignore_index=-100)

    batch = collator(
        [
            {
                "loss_kind": "dpo_target",
                "sample_id": "a",
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
                "chosen_input_ids": [4],
                "chosen_attention_mask": [1],
                "chosen_labels": [4],
                "rejected_input_ids": [5, 6],
                "rejected_attention_mask": [1, 1],
                "rejected_labels": [-100, 6],
            },
        ]
    )

    assert batch["chosen_input_ids"].tolist() == [[1, 2], [4, 0]]
    assert batch["chosen_labels"].tolist() == [[-100, 2], [4, -100]]
    assert batch["rejected_input_ids"].tolist() == [[3, 0], [5, 6]]
    assert batch["rejected_labels"].tolist() == [[3, -100], [-100, 6]]
