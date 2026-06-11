from __future__ import annotations

from typing import Any

from qwen35_tuning.config.hashing import stable_hash

from .schemas import CanonicalRow, LossKind, Split


class CanonicalizationError(ValueError):
    pass


def canonicalize_row(row: dict[str, Any], split: Split, row_index: int) -> CanonicalRow:
    if not isinstance(row, dict):
        raise CanonicalizationError("raw row must be a mapping")

    loss_kind = row.get("loss_kind")
    if loss_kind not in {"sft_target", "sft_tool", "dpo_target"}:
        raise CanonicalizationError("canonical row requires explicit loss_kind from parquet type/target column")
    sample_id = str(row.get("sample_id") or row.get("id") or stable_hash({"split": split, "row_index": row_index, "row": row}))
    metadata = dict(row.get("metadata") or {})
    metadata.setdefault("source_hash", stable_hash(row))

    if loss_kind == "dpo_target":
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not isinstance(prompt, list) or not isinstance(chosen, dict) or not isinstance(rejected, dict):
            raise CanonicalizationError("dpo_target row requires prompt list and chosen/rejected mappings")
        return CanonicalRow(
            sample_id=sample_id,
            split=split,
            loss_kind=loss_kind,
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata=metadata,
        )

    messages = row.get("messages")
    if not isinstance(messages, list):
        raise CanonicalizationError(f"{loss_kind} row requires messages list")
    tools = row.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise CanonicalizationError("tools must be a list when present")

    return CanonicalRow(
        sample_id=sample_id,
        split=split,
        loss_kind=loss_kind,
        messages=messages,
        tools=tools,
        parallel_tool_calls=row.get("parallel_tool_calls"),
        metadata=metadata,
    )
