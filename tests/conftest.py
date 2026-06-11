from __future__ import annotations


class CharTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        input_ids = [ord(char) for char in text]
        result = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

