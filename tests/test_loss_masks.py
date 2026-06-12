from __future__ import annotations

from config import load_config
from preprocessing.masking import canonicalize_row
from preprocessing.pipeline import preprocess_dpo_row, preprocess_sft_row
from preprocessing.rendering import QwenTemplateRenderer
from conftest import CharTokenizer


def supervised_text(rendered: str, input_ids: list[int], labels: list[int]) -> str:
    chars = []
    for token_id, label in zip(input_ids, labels):
        if label != -100:
            chars.append(chr(token_id))
    return "".join(chars)


def test_sft_masks_all_long_assistant_replies_for_sft_target():
    config = load_config("configs/config.preprocess.yaml")
    tokenizer = CharTokenizer()
    renderer = QwenTemplateRenderer(None, config.rendering)
    row = canonicalize_row(
        {
            "sample_id": "mask1",
            "loss_kind": "sft_target",
            "messages": [
                {"role": "assistant", "content": "not a target " * 10},
                {"role": "user", "content": "tell me"},
                {"role": "assistant", "content": "this is a long response " * 5},
            ],
        },
        "train",
        0,
    )

    processed, audit = preprocess_sft_row(row, renderer, tokenizer, config)
    text = supervised_text(audit["rendered_text"], processed["input_ids"], processed["labels"])
    assert "not a target" in text
    assert "this is a long response" in text
    assert processed["num_supervised_tokens"] > 0


def test_tool_policy_supervises_tool_call_and_final_answer_not_tool_response():
    config = load_config("configs/config.preprocess.yaml")
    tokenizer = CharTokenizer()
    renderer = QwenTemplateRenderer(None, config.rendering)
    row = canonicalize_row(
        {
            "sample_id": "tool1",
            "loss_kind": "sft_tool",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "find"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": "search_books", "arguments": "{\"q\":\"x\"}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_0", "content": "tool secret result"},
                {"role": "assistant", "content": "final answer after tool " * 4},
            ],
            "tools": [{"type": "function", "function": {"name": "search_books"}}],
        },
        "train",
        0,
    )

    processed, audit = preprocess_sft_row(row, renderer, tokenizer, config)
    text = supervised_text(audit["rendered_text"], processed["input_ids"], processed["labels"])
    assert "search_books" in text
    assert "final answer after tool" in text
    assert "tool secret result" not in text


def test_tool_policy_supervises_text_answer_after_user_when_no_tool_needed():
    config = load_config("configs/config.preprocess.yaml")
    tokenizer = CharTokenizer()
    renderer = QwenTemplateRenderer(None, config.rendering)
    row = canonicalize_row(
        {
            "sample_id": "tool-no-call",
            "loss_kind": "sft_tool",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "answer directly"},
                {"role": "assistant", "content": "direct answer"},
            ],
        },
        "train",
        0,
    )

    processed, audit = preprocess_sft_row(row, renderer, tokenizer, config)
    text = supervised_text(audit["rendered_text"], processed["input_ids"], processed["labels"])
    assert text == "direct answer"


def test_dpo_masks_only_chosen_and_rejected_completion_text():
    config = load_config("configs/config.preprocess.yaml")
    tokenizer = CharTokenizer()
    renderer = QwenTemplateRenderer(None, config.rendering)
    row = canonicalize_row(
        {
            "id": "dpo1",
            "loss_kind": "dpo_target",
            "prompt": [{"role": "system", "content": "system"}, {"role": "user", "content": "prompt"}],
            "chosen": {"role": "assistant", "content": "chosen completion"},
            "rejected": {"role": "assistant", "content": "rejected completion"},
        },
        "train",
        0,
    )

    processed, _ = preprocess_dpo_row(row, renderer, tokenizer, config)
    chosen = supervised_text("", processed["chosen_input_ids"], processed["chosen_labels"])
    rejected = supervised_text("", processed["rejected_input_ids"], processed["rejected_labels"])
    assert chosen == "chosen completion"
    assert rejected == "rejected completion"
