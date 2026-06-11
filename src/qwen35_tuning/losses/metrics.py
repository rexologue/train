from __future__ import annotations


def mask_metrics(labels: list[int], ignore_index: int) -> dict[str, int]:
    supervised = sum(1 for label in labels if label != ignore_index)
    return {"num_tokens": len(labels), "num_supervised_tokens": supervised, "num_prompt_tokens": len(labels) - supervised}

