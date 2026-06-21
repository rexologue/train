from __future__ import annotations

import json

import pandas as pd
import pytest

from preprocessing.io import ParquetSchemaError, read_rows
from preprocessing.masking import canonicalize_row
from preprocessing.pipeline import preprocess_dpo_row, preprocess_sft_row, preprocess_split
from preprocessing.rendering import QwenTemplateRenderer
from conftest import CharTokenizer, example_config, renderer_config


class ChatTemplateTokenizer(CharTokenizer):
    is_fast = True

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> str:
        del tokenize, add_generation_prompt, kwargs
        rendered = []
        for message in messages:
            rendered.append(f"<|im_start|>{message['role']}\n{message.get('content', '')}<|im_end|>")
        return "".join(rendered)


def supervised_text(input_ids: list[int], labels: list[int], *, ignore_index: int = -100) -> str:
    return "".join(chr(token_id) for token_id, label in zip(input_ids, labels) if label != ignore_index)


def test_read_rows_decodes_authoritative_type_column(tmp_path) -> None:
    path = tmp_path / "valid.parquet"
    pd.DataFrame(
        [
            {
                "data": json.dumps({"messages": [{"role": "user", "content": "q"}]}, ensure_ascii=False),
                "type": "sft_target",
            }
        ]
    ).to_parquet(path, index=False)

    rows = read_rows(path)

    assert rows[0]["loss_kind"] == "sft_target"
    assert rows[0]["messages"][0]["content"] == "q"
    assert rows[0]["metadata"]["parquet_type_column"] == "type"


def test_read_rows_rejects_legacy_target_column(tmp_path) -> None:
    path = tmp_path / "legacy.parquet"
    pd.DataFrame(
        [
            {
                "data": json.dumps({"messages": [{"role": "user", "content": "q"}]}, ensure_ascii=False),
                "target": "sft_tool",
            }
        ]
    ).to_parquet(path, index=False)

    with pytest.raises(ParquetSchemaError, match="must not contain target"):
        read_rows(path)


def test_preprocess_split_force_refresh_rebuilds_valid_cache(tmp_path) -> None:
    raw_path = tmp_path / "train.parquet"
    pd.DataFrame(
        [
            {
                "data": json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "question"},
                            {"role": "assistant", "content": "long supervised answer " * 8},
                        ]
                    },
                    ensure_ascii=False,
                ),
                "type": "sft_target",
            }
        ]
    ).to_parquet(raw_path, index=False)
    config = example_config(
        project={"output_dir": str(tmp_path / "run")},
        preprocessing={
            "raw": {"train_path": str(raw_path), "valid_path": str(raw_path)},
            "quality": {"min_processed_rows_per_loss_kind": {"sft_tool": 0, "dpo_target": 0}},
        },
    )
    tokenizer = ChatTemplateTokenizer()

    first = preprocess_split("train", raw_path, tokenizer, config, preprocessing_signature="sig")
    reused = preprocess_split("train", raw_path, tokenizer, config, preprocessing_signature="sig")
    refreshed = preprocess_split(
        "train",
        raw_path,
        tokenizer,
        config,
        preprocessing_signature="sig",
        force_refresh=True,
    )

    assert first.reused is False
    assert reused.reused is True
    assert refreshed.reused is False


def test_sft_target_masks_selected_assistant_tokens_only() -> None:
    config = example_config()
    tokenizer = CharTokenizer()
    renderer = QwenTemplateRenderer(None, renderer_config(config))
    row = canonicalize_row(
        {
            "sample_id": "mask1",
            "loss_kind": "sft_target",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "assistant", "content": "not a target " * 10},
                {"role": "user", "content": "tell me"},
                {"role": "assistant", "content": "this is a long response " * 5},
            ],
        },
        "train",
        0,
    )

    processed, audit = preprocess_sft_row(row, renderer, tokenizer, config)
    text = supervised_text(processed["input_ids"], processed["labels"])

    assert "system prompt" not in text
    assert "tell me" not in text
    assert "not a target" in text
    assert "this is a long response" in text
    assert audit["target_selection"]["num_long_targets_kept"] == 2


def test_sft_tool_masks_assistant_tool_call_and_final_answer_not_tool_response() -> None:
    config = example_config()
    tokenizer = CharTokenizer()
    renderer = QwenTemplateRenderer(None, renderer_config(config))
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

    processed, _audit = preprocess_sft_row(row, renderer, tokenizer, config)
    text = supervised_text(processed["input_ids"], processed["labels"])

    assert "search_books" in text
    assert "final answer after tool" in text
    assert "tool secret result" not in text


def test_reasoning_disabled_masks_think_blocks() -> None:
    config = example_config(
        preprocessing={
            "masking": {
                "policies": {
                    "sft_target": {
                        "min_guaranteed_assistant_chars": 0,
                        "loss_on_short_assistant_reply_prob": 1.0,
                    }
                }
            }
        }
    )
    tokenizer = CharTokenizer()
    renderer = QwenTemplateRenderer(None, renderer_config(config))
    row = canonicalize_row(
        {
            "sample_id": "think1",
            "loss_kind": "sft_target",
            "messages": [
                {
                    "role": "assistant",
                    "content": "<think>hidden reasoning</think>visible answer",
                }
            ],
        },
        "train",
        0,
    )

    processed, _audit = preprocess_sft_row(row, renderer, tokenizer, config)
    text = supervised_text(processed["input_ids"], processed["labels"])

    assert "visible answer" in text
    assert "hidden reasoning" not in text


def test_dpo_masks_chosen_and_rejected_completion_text() -> None:
    config = example_config()
    tokenizer = CharTokenizer()
    renderer = QwenTemplateRenderer(None, renderer_config(config))
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

    processed, _audit = preprocess_dpo_row(row, renderer, tokenizer, config)

    assert supervised_text(processed["chosen_input_ids"], processed["chosen_labels"]) == "chosen completion"
    assert supervised_text(processed["rejected_input_ids"], processed["rejected_labels"]) == "rejected completion"
