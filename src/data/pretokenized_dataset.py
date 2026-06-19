from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any

import pandas as pd


SFT_LOSS_KINDS = {"sft_target", "sft_tool"}
DPO_LOSS_KIND = "dpo_target"
LOSS_KINDS = SFT_LOSS_KINDS | {DPO_LOSS_KIND}


def _is_missing(value: Any) -> bool:
    """Return whether a pandas/parquet value should be treated as absent."""

    return value is None or (isinstance(value, float) and math.isnan(value))


def _int_list(value: Any, *, field: str, row_index: int) -> list[int]:
    """Normalize parquet list-like token columns to plain `list[int]`.

    PyArrow-backed pandas reads can return Python lists, tuples, numpy arrays,
    or extension-array objects depending on installed versions. The Dataset
    boundary normalizes those variants once, so collators and losses do not
    need parquet-specific branches.
    """

    if _is_missing(value):
        raise ValueError(f"row {row_index}: missing required field {field!r}")
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif hasattr(value, "tolist"):
        items = value.tolist()
    else:
        raise ValueError(f"row {row_index}: field {field!r} must be a list of token ids")
    if not isinstance(items, list):
        raise ValueError(f"row {row_index}: field {field!r} must decode to a list")
    return [int(item) for item in items]


def _optional(value: Any) -> Any:
    """Convert parquet null/NaN values into `None` while preserving real data."""

    return None if _is_missing(value) else value


def _optional_float(value: Any) -> float | None:
    """Convert an optional parquet scalar into a float."""

    value = _optional(value)
    return None if value is None else float(value)


class PretokenizedDataset:
    """Map-style dataset over cached pretokenized parquet rows.

    This dataset is deliberately route-agnostic at the container level: one
    parquet split may contain `sft_target`, `sft_tool`, and `dpo_target`
    rows. The authoritative route remains the preprocessed
    `loss_kind` column; batching and loss dispatch happen downstream.
    """

    def __init__(self, rows: list[dict[str, Any]], *, split: str | None = None, path: str | Path | None = None):
        self.split = split
        self.path = Path(path) if path is not None else None
        self.rows = [self._normalize_row(row, fallback_index=index) for index, row in enumerate(rows)]
        self.loss_kinds = [row["loss_kind"] for row in self.rows]
        self.loss_kind_counts = Counter(self.loss_kinds)


    @classmethod
    def from_parquet(cls, path: str | Path, *, split: str | None = None) -> "PretokenizedDataset":
        """Read a pretokenized parquet split and validate its training schema."""

        parquet_path = Path(path)
        frame = pd.read_parquet(parquet_path)
        return cls(frame.to_dict(orient="records"), split=split, path=parquet_path)


    def __len__(self) -> int:
        return len(self.rows)


    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


    def _normalize_row(self, row: dict[str, Any], *, fallback_index: int) -> dict[str, Any]:
        """Normalize one physical parquet row into the in-memory Dataset schema."""

        loss_kind = row.get("loss_kind")
        if loss_kind not in LOSS_KINDS:
            raise ValueError(f"row {fallback_index}: unknown loss_kind {loss_kind!r}")

        row_index = row.get("row_index")
        normalized: dict[str, Any] = {
            "sample_id": str(row.get("sample_id") or fallback_index),
            "row_index": fallback_index if _is_missing(row_index) else int(row_index),
            "loss_kind": loss_kind,
        }

        if self.split is not None:
            normalized["split"] = self.split

        # SFT/tool rows are a single prompt+completion sequence. DPO rows carry
        # two independently rendered branches and must keep both sides intact.
        if loss_kind in SFT_LOSS_KINDS:
            normalized.update(self._normalize_sft(row, fallback_index))
        else:
            normalized.update(self._normalize_dpo(row, fallback_index))

        for field in ("source_hash", "render_hash", "chosen_render_hash", "rejected_render_hash"):
            value = _optional(row.get(field))
            if value is not None:
                normalized[field] = value

        return normalized


    def _normalize_sft(self, row: dict[str, Any], row_index: int) -> dict[str, Any]:
        """Validate fields used by the SFT cross-entropy route."""

        input_ids = _int_list(row.get("input_ids"), field="input_ids", row_index=row_index)
        attention_mask = _int_list(row.get("attention_mask"), field="attention_mask", row_index=row_index)
        labels = _int_list(row.get("labels"), field="labels", row_index=row_index)

        if not (len(input_ids) == len(attention_mask) == len(labels)):
            raise ValueError(f"row {row_index}: input_ids, attention_mask and labels must have equal length")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "length": int(_optional(row.get("length")) or len(input_ids)),
            "num_supervised_tokens": int(_optional(row.get("num_supervised_tokens")) or 0),
        }


    def _normalize_dpo(self, row: dict[str, Any], row_index: int) -> dict[str, Any]:
        """Validate chosen/rejected branches used by the DPO route."""

        normalized: dict[str, Any] = {}
        for side in ("chosen", "rejected"):
            input_ids = _int_list(row.get(f"{side}_input_ids"), field=f"{side}_input_ids", row_index=row_index)
            attention_mask = _int_list(row.get(f"{side}_attention_mask"), field=f"{side}_attention_mask", row_index=row_index)
            labels = _int_list(row.get(f"{side}_labels"), field=f"{side}_labels", row_index=row_index)
            
            if not (len(input_ids) == len(attention_mask) == len(labels)):
                raise ValueError(f"row {row_index}: {side} input_ids, attention_mask and labels must have equal length")

            normalized[f"{side}_input_ids"] = input_ids
            normalized[f"{side}_attention_mask"] = attention_mask
            normalized[f"{side}_labels"] = labels
            normalized[f"{side}_length"] = int(_optional(row.get(f"{side}_length")) or len(input_ids))
            normalized[f"{side}_completion_token_count"] = int(_optional(row.get(f"{side}_completion_token_count")) or 0)
            ref_logp = _optional_float(row.get(f"{side}_ref_logp"))
            if ref_logp is not None:
                normalized[f"{side}_ref_logp"] = ref_logp

        return normalized
