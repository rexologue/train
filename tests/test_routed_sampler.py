from __future__ import annotations

from data.collators import SFTCollator
from sampling.routed_batch_sampler import RoutedBatchSampler


def test_routed_sampler_batches_are_homogeneous():
    loss_kinds = ["sft_tool", "sft_target", "sft_tool", "dpo_target"]
    sampler = RoutedBatchSampler(loss_kinds, batch_size=2)
    for batch in sampler:
        assert len({loss_kinds[index] for index in batch}) == 1


def test_collator_rejects_mixed_loss_kind_batch():
    collator = SFTCollator(pad_token_id=0, ignore_index=-100)
    try:
        collator(
            [
                {"loss_kind": "sft_target", "input_ids": [1], "attention_mask": [1], "labels": [1], "sample_id": "a"},
                {"loss_kind": "sft_tool", "input_ids": [2], "attention_mask": [1], "labels": [2], "sample_id": "b"},
            ]
        )
    except ValueError as exc:
        assert "homogeneous" in str(exc)
    else:
        raise AssertionError("mixed loss_kind batch was accepted")

