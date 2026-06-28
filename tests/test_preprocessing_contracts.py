from __future__ import annotations

import json

import pandas as pd
import pytest

from preprocessing.io import ParquetSchemaError, read_rows
from preprocessing.masking import canonicalize_row
from preprocessing.pipeline import (
    _preprocess_raw_sft_row,
    preprocess_dpo_row,
    preprocess_sft_row,
    preprocess_split,
)
from preprocessing.rendering import ChatTemplateRenderer
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


class GenerationMaskTokenizer(CharTokenizer):
    """Char tokenizer that also emits a {% generation %}-style assistant mask.

    Mirrors the real Qwen template contract used by the production masking path:
    the generated span is the assistant body + the assistant <|im_end|> (the
    role header and any think scaffold are excluded). Lets CI exercise
    tokenize_with_generation_mask without the real tokenizer.
    """

    is_fast = True

    def _render(self, messages):
        parts: list[str] = []
        spans: list[tuple[int, int]] = []
        cursor = 0
        for message in messages:
            header = f"<|im_start|>{message['role']}\n"
            parts.append(header)
            cursor += len(header)
            body = str(message.get("content") or "")
            start = cursor
            parts.append(body)
            cursor += len(body)
            end = "<|im_end|>"
            parts.append(end)
            cursor += len(end)
            if message.get("role") == "assistant":
                spans.append((start, cursor))  # body + <|im_end|>
            parts.append("\n")
            cursor += 1
        return "".join(parts), spans

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False,
                            return_dict=False, return_assistant_tokens_mask=False, tools=None, **kwargs):
        del add_generation_prompt, tools, kwargs
        text, spans = self._render(messages)
        if not tokenize:
            return text
        ids = [ord(ch) for ch in text]
        result = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_assistant_tokens_mask:
            mask = [0] * len(ids)
            for start, end in spans:
                for index in range(start, end):
                    mask[index] = 1
            result["assistant_masks"] = mask
        return result


def test_generation_mask_path_supervises_assistant_only() -> None:
    config = example_config()
    tokenizer = GenerationMaskTokenizer()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "the question"},
        {"role": "assistant", "content": "a sufficiently long supervised answer " * 3},
    ]
    processed, _debug, stats = _preprocess_raw_sft_row(_sft_target_row(messages), tokenizer, config)
    text = supervised_text(processed["input_ids"], processed["labels"])

    assert stats["masking_path/generation_mask"] == 1  # used the new path, not the fallback
    assert processed["num_supervised_tokens"] > 0
    assert "system prompt" not in text and "the question" not in text
    assert "a sufficiently long supervised answer" in text
    assert "<|im_end|>" in text


def test_generation_mask_path_honors_user_anchor() -> None:
    messages = [
        {"role": "assistant", "content": "unprompted opener that is plenty long here " * 2},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "anchored reply that is also plenty long here " * 2},
    ]
    default_proc, _d, _s = _preprocess_raw_sft_row(_sft_target_row(messages), GenerationMaskTokenizer(), example_config())
    assert "unprompted opener" in supervised_text(default_proc["input_ids"], default_proc["labels"])

    anchored = example_config(
        preprocessing={"masking": {"policies": {"sft_target": {"require_user_anchor": True}}}}
    )
    proc, _d, stats = _preprocess_raw_sft_row(_sft_target_row(messages), GenerationMaskTokenizer(), anchored)
    text = supervised_text(proc["input_ids"], proc["labels"])
    assert stats["masking_path/generation_mask"] == 1
    assert "unprompted opener" not in text
    assert "anchored reply" in text


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


def _sft_target_row(messages):
    return {"payload": {"sample_id": "prod1", "messages": messages}, "loss_kind": "sft_target", "row_index": 0}


def test_production_path_supervises_assistant_completions_and_eom() -> None:
    # Exercises the REAL training path (_preprocess_raw_sft_row ->
    # select_sft_supervision_ranges -> assistant_blocks), not the test-helper
    # renderer. Locks in: system/user not supervised, assistant content + the
    # <|im_end|> stop token supervised.
    config = example_config()
    tokenizer = ChatTemplateTokenizer()
    messages = [
        {"role": "system", "content": "system prompt here"},
        {"role": "user", "content": "please answer"},
        {"role": "assistant", "content": "this is a sufficiently long supervised answer " * 3},
    ]
    processed, _debug, _stats = _preprocess_raw_sft_row(_sft_target_row(messages), tokenizer, config)
    text = supervised_text(processed["input_ids"], processed["labels"])

    assert processed["num_supervised_tokens"] > 0
    assert "system prompt here" not in text
    assert "please answer" not in text
    assert "this is a sufficiently long supervised answer" in text
    assert "<|im_end|>" in text  # assistant stop token must be supervised so the model learns to stop


def test_production_path_user_anchor_drops_unprompted_opener() -> None:
    # Dialog starts with assistant (no preceding user) -> opener is unprompted.
    messages = [
        {"role": "assistant", "content": "unprompted opener line that is quite long " * 2},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "anchored answer that is also quite long enough " * 2},
    ]
    tokenizer = ChatTemplateTokenizer()

    default_cfg = example_config()  # require_user_anchor defaults to false
    default_text = supervised_text(
        *(lambda p: (p["input_ids"], p["labels"]))(
            _preprocess_raw_sft_row(_sft_target_row(messages), tokenizer, default_cfg)[0]
        )
    )
    assert "unprompted opener line" in default_text  # current behavior keeps openers

    anchored_cfg = example_config(
        preprocessing={"masking": {"policies": {"sft_target": {"require_user_anchor": True}}}}
    )
    anchored = _preprocess_raw_sft_row(_sft_target_row(messages), tokenizer, anchored_cfg)[0]
    anchored_text = supervised_text(anchored["input_ids"], anchored["labels"])
    assert "unprompted opener line" not in anchored_text  # opener dropped
    assert "anchored answer" in anchored_text  # user-anchored turn kept


def test_sft_target_masks_selected_assistant_tokens_only() -> None:
    config = example_config()
    tokenizer = CharTokenizer()
    renderer = ChatTemplateRenderer(None, renderer_config(config))
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
    renderer = ChatTemplateRenderer(None, renderer_config(config))
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
    renderer = ChatTemplateRenderer(None, renderer_config(config))
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
    renderer = ChatTemplateRenderer(None, renderer_config(config))
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
