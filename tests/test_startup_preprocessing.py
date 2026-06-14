from __future__ import annotations

import pytest

from config import load_config
from preprocessing.io import load_manifest, load_pretokenized_split_results, split_cache_is_valid, write_split_cache
from preprocessing.pipeline import (
    _preprocess_sft_dataset_row,
    enforce_max_seq_len,
    validate_configured_max_seq_len,
)
from preprocessing.rendering import RenderingAuditError


class StartupTokenizer:
    chat_template = "startup-test-template"
    model_max_length = 32

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        result = {"input_ids": [ord(char) for char in text], "attention_mask": [1] * len(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        parts = []
        for message in messages:
            role = message["role"]
            parts.append(f"<|im_start|>{role}\n")
            if role == "assistant":
                parts.append(str(message.get("content") or message.get("tool_calls") or ""))
            else:
                parts.append(str(message.get("content") or ""))
            parts.append("<|im_end|>\n")
        return "".join(parts)


class ThinkingTokenizer(StartupTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        return "<think>hidden</think>\n" + super().apply_chat_template(messages, tokenize, add_generation_prompt, tools)


def sft_row(sample_id: str, user_content: str = "q", assistant_content: str = "long answer " * 10) -> dict:
    return {
        "row_index": 0,
        "loss_kind": "sft_target",
        "payload": {
            "sample_id": sample_id,
            "messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}],
        },
    }


def test_split_cache_reuse_depends_on_split_hash_signature_and_content_hash(tmp_path):
    root = tmp_path / "pretok"
    manifest = write_split_cache(
        root,
        "valid",
        [{"sample_id": "a", "input_ids": [1]}],
        [],
        {
            "input_sha256": "sha256:valid-a",
            "num_raw_rows": 2,
            "num_rows": 1,
            "num_rejected_rows": 1,
            "rejected_counts": {"ValueError: bad row": 1},
            "preprocessing_signature": "sha256:signature-a",
        },
        base_manifest={"splits": {"train": "sha256:train-a"}},
        examples_per_loss_kind=5,
    )

    assert manifest["splits"] == {"train": "sha256:train-a", "valid": "sha256:valid-a"}
    assert manifest["rejections"]["valid"] == {"ValueError: bad row": 1}
    assert split_cache_is_valid(root, "valid", "sha256:valid-a", "sha256:signature-a")[0] is True
    assert split_cache_is_valid(root, "valid", "sha256:valid-a", "sha256:signature-b")[0] is False
    assert split_cache_is_valid(root, "valid", "sha256:valid-b")[0] is False
    assert split_cache_is_valid(root, "train", "sha256:train-a")[0] is False

    (root / "valid.parquet").write_bytes(b"corrupt")
    assert split_cache_is_valid(root, "valid", "sha256:valid-a", "sha256:signature-a")[0] is False


def test_load_pretokenized_split_results_from_existing_manifest(tmp_path):
    root = tmp_path / "pretok"
    raw_train = tmp_path / "train.parquet"
    raw_valid = tmp_path / "valid.parquet"
    raw_train.write_bytes(b"train")
    raw_valid.write_bytes(b"valid")
    write_split_cache(
        root,
        "train",
        [{"sample_id": "a", "input_ids": [1]}],
        [],
        {
            "input_sha256": "sha256:train",
            "num_raw_rows": 1,
            "num_rows": 1,
            "num_rejected_rows": 0,
            "rejected_counts": {},
        },
        base_manifest={},
        examples_per_loss_kind=5,
    )
    write_split_cache(
        root,
        "valid",
        [{"sample_id": "b", "input_ids": [2]}],
        [],
        {
            "input_sha256": "sha256:valid",
            "num_raw_rows": 1,
            "num_rows": 1,
            "num_rejected_rows": 0,
            "rejected_counts": {},
        },
        base_manifest=load_manifest(root) or {},
        examples_per_loss_kind=5,
    )
    config = load_config("configs/config.preprocess.yaml")
    config.raw["preprocessing"]["output"]["root_dir"] = str(root)
    config.raw["preprocessing"]["raw"]["train_path"] = str(raw_train)
    config.raw["preprocessing"]["raw"]["valid_path"] = str(raw_valid)

    results = load_pretokenized_split_results(config, ["train", "valid"])

    assert [result.split for result in results] == ["train", "valid"]
    assert results[0].reused is True
    assert results[0].manifest["input_sha256"] == "sha256:train"


def test_startup_template_falls_back_when_no_thinking_kwarg_is_unsupported():
    config = load_config("configs/config.preprocess.yaml")

    _processed, debug, stats = _preprocess_sft_dataset_row(sft_row("fallback"), StartupTokenizer(), config)
    assert debug["unsupported_apply_chat_template_kwargs"] == ["enable_thinking"]
    assert stats["unsupported_apply_chat_template_kwargs/enable_thinking"] == 1


def test_startup_debug_loss_only_text_is_decoded_from_labels():
    config = load_config("configs/config.preprocess.yaml")
    processed, debug, _stats = _preprocess_sft_dataset_row(sft_row("loss-only"), StartupTokenizer(), config)

    expected = "".join(chr(token_id) for token_id, label in zip(processed["input_ids"], processed["labels"]) if label != config.ignore_index)
    assert debug["loss_only_text"] == expected
    assert debug["loss_only_text"] == "long answer " * 10 + "<|im_end|>"


def test_think_block_is_excluded_from_loss_when_thinking_disabled():
    config = load_config("configs/config.preprocess.yaml")
    visible = "visible " * 20
    _processed, debug, _stats = _preprocess_sft_dataset_row(
        sft_row("think-off", assistant_content=f"<think>hidden</think>{visible}"),
        StartupTokenizer(),
        config,
    )

    assert debug["loss_only_text"] == visible + "<|im_end|>"


def test_think_block_is_included_in_loss_when_thinking_enabled():
    config = load_config("configs/config.preprocess.yaml")
    config.raw["preprocessing"]["reasoning"]["enable_thinking"] = True
    visible = "visible " * 20
    _processed, debug, _stats = _preprocess_sft_dataset_row(
        sft_row("think-on", assistant_content=f"<think>hidden</think>{visible}"),
        StartupTokenizer(),
        config,
    )

    assert debug["loss_only_text"] == f"<think>hidden</think>{visible}<|im_end|>"


def test_startup_route_rejects_raw_template_markers():
    config = load_config("configs/config.preprocess.yaml")

    with pytest.raises(RenderingAuditError):
        _preprocess_sft_dataset_row(sft_row("raw-marker", user_content="bad <|im_start|> marker"), StartupTokenizer(), config)


def test_overlong_startup_row_is_rejected_by_max_seq_len():
    config = load_config("configs/config.preprocess.yaml")
    processed = {"input_ids": list(range(33))}

    with pytest.raises(ValueError, match="exceeds preprocessing.sequence.max_seq_len"):
        enforce_max_seq_len(processed, max_seq_len=32)


def test_configured_max_seq_len_must_not_exceed_model_context():
    config = load_config("configs/config.preprocess.yaml")
    config.raw["preprocessing"]["sequence"]["max_seq_len"] = 64

    with pytest.raises(ValueError, match="preprocessing.sequence.max_seq_len"):
        validate_configured_max_seq_len(config, StartupTokenizer())
