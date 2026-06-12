from __future__ import annotations

import json

import pandas as pd
import pytest

from preprocessing.io import ParquetSchemaError, read_rows


def test_read_rows_decodes_data_json_and_type_column(tmp_path):
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


def test_read_rows_accepts_current_target_column(tmp_path):
    path = tmp_path / "valid.parquet"
    pd.DataFrame(
        [
            {
                "data": json.dumps({"messages": [{"role": "user", "content": "q"}]}, ensure_ascii=False),
                "target": "sft_tool",
            }
        ]
    ).to_parquet(path, index=False)

    rows = read_rows(path)
    assert rows[0]["loss_kind"] == "sft_tool"
    assert rows[0]["metadata"]["parquet_type_column"] == "target"


def test_read_rows_accepts_matching_type_and_target_with_type_priority(tmp_path):
    path = tmp_path / "valid.parquet"
    pd.DataFrame(
        [
            {
                "data": json.dumps({"messages": [{"role": "user", "content": "q"}]}, ensure_ascii=False),
                "type": "sft_target",
                "target": "sft_target",
            }
        ]
    ).to_parquet(path, index=False)

    rows = read_rows(path)
    assert rows[0]["loss_kind"] == "sft_target"
    assert rows[0]["metadata"]["parquet_type_column"] == "type"


def test_read_rows_rejects_conflicting_type_and_target(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame(
        [
            {
                "data": json.dumps({"messages": [{"role": "user", "content": "q"}]}, ensure_ascii=False),
                "type": "sft_target",
                "target": "sft_tool",
            }
        ]
    ).to_parquet(path, index=False)

    with pytest.raises(ParquetSchemaError):
        read_rows(path)
