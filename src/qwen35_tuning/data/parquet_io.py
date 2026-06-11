from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ParquetSchemaError(ValueError):
    pass


def _decode_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    if "data" not in row:
        return row

    raw_data = row["data"]
    if isinstance(raw_data, str):
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise ParquetSchemaError(f"row {row_index} data column is not valid JSON") from exc
    elif isinstance(raw_data, dict):
        payload = raw_data
    else:
        raise ParquetSchemaError(f"row {row_index} data column must be JSON string or mapping")

    if not isinstance(payload, dict):
        raise ParquetSchemaError(f"row {row_index} data JSON must decode to an object")

    type_value = row.get("type")
    target_value = row.get("target")
    if type_value is not None and target_value is not None and type_value != target_value:
        raise ParquetSchemaError(f"row {row_index} has conflicting type={type_value!r} and target={target_value!r}")

    loss_kind = type_value if type_value is not None else target_value
    if loss_kind is not None:
        payload = dict(payload)
        payload["loss_kind"] = loss_kind

    metadata = dict(payload.get("metadata") or {})
    metadata["parquet_row_index"] = row_index
    metadata["parquet_type_column"] = "type" if "type" in row else "target" if "target" in row else None
    payload["metadata"] = metadata
    return payload


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    return [_decode_row(row, index) for index, row in enumerate(frame.to_dict(orient="records"))]


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
