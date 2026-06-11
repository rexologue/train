from __future__ import annotations

from typing import Any


class DPODataset:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if row.get("loss_kind") != "dpo_target":
            raise ValueError("DPODataset only accepts dpo_target rows")
        return row

