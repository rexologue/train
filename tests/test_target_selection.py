from __future__ import annotations

from qwen35_tuning.data.schemas import AssistantSpan, CanonicalRow
from qwen35_tuning.masking.target_selection import select_sft_target_spans


def test_dialog_starting_with_assistant_is_selected_by_plain_assistant_policy():
    row = CanonicalRow(
        sample_id="s1",
        split="train",
        loss_kind="sft_target",
        messages=[
            {"role": "assistant", "content": "prior assistant text that must not train"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "long enough answer"},
        ],
    )
    spans = [
        AssistantSpan(0, 0, 40, "content"),
        AssistantSpan(2, 50, 68, "content"),
    ]
    selected, summary = select_sft_target_spans(
        row,
        spans,
        {
            "min_guaranteed_assistant_chars": 10,
            "loss_on_short_assistant_reply_prob": 0.0,
            "short_response_sampling_seed": 42,
        },
    )
    assert [span.message_index for span in selected] == [0, 2]
    assert summary.num_target_candidates == 2


def test_short_response_policy_can_drop_or_keep_deterministically():
    row = CanonicalRow(
        sample_id="s2",
        split="train",
        loss_kind="sft_target",
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "ok"}],
    )
    spans = [AssistantSpan(1, 10, 12, "content")]
    base_policy = {
        "min_guaranteed_assistant_chars": 80,
        "short_response_sampling_seed": 42,
    }
    dropped, _ = select_sft_target_spans(row, spans, {**base_policy, "loss_on_short_assistant_reply_prob": 0.0})
    kept, _ = select_sft_target_spans(row, spans, {**base_policy, "loss_on_short_assistant_reply_prob": 1.0})
    assert dropped == []
    assert [span.message_index for span in kept] == [1]
