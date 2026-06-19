from __future__ import annotations

from pathlib import Path
from typing import Any

from config import Config, load_config


class CharTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 3
    chat_template = "char-tokenizer-test-template"
    model_max_length = 4096

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False, **kwargs: Any):
        del add_special_tokens, kwargs
        input_ids = [ord(char) for char in text]
        result = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(self, token_ids: list[int], **kwargs: Any) -> str:
        del kwargs
        return "".join(chr(token_id) for token_id in token_ids)


def example_config(**updates: Any) -> Config:
    data = load_config("configs/config.example.yaml").to_dict()
    deep_update(data, updates)
    return Config.from_dict(data)


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def renderer_config(config: Config) -> dict[str, Any]:
    return {
        "use_system": config.preprocessing.rendering.use_system,
        "reject_raw_special_markers": config.preprocessing.rendering.reject_raw_special_markers,
    }


def pretok_result(split: str, path: Path):
    from preprocessing.io import PretokSplitResult

    return PretokSplitResult(
        split=split,
        raw_path=path,
        output_dir=path.parent,
        pretok_path=path,
        manifest_path=path.parent / "manifest.json",
        reused=False,
        manifest={},
    )
