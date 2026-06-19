from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from preprocessing.io import LOSS_KINDS
from registry.modelctl_client import ModelctlClient
from tracking.lineage import collect_tracking_lineage
from conftest import example_config


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


@pytest.mark.integration
def test_opt_in_dataset_root_has_supported_raw_splits() -> None:
    data_root = env_path("ESTADEL_TEST_DATA_ROOT")
    if data_root is None:
        pytest.skip("set ESTADEL_TEST_DATA_ROOT to run dataset integration checks")

    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    for split in ("train", "valid"):
        path = data_root / "data" / f"{split}.parquet"
        assert path.is_file(), f"missing {split} parquet: {path}"
        parquet_file = pyarrow_parquet.ParquetFile(path)
        column_names = set(parquet_file.schema_arrow.names)
        assert {"data", "type"} <= column_names
        assert "target" not in column_names
        assert parquet_file.metadata.num_rows > 0

        first_batch = next(parquet_file.iter_batches(batch_size=1, columns=["data", "type"]))
        first = first_batch.to_pylist()[0]
        assert first["type"] in LOSS_KINDS
        assert isinstance(json.loads(first["data"]), dict)


@pytest.mark.integration
def test_opt_in_dataset_root_exposes_dvc_lineage() -> None:
    data_root = env_path("ESTADEL_TEST_DATA_ROOT")
    if data_root is None:
        pytest.skip("set ESTADEL_TEST_DATA_ROOT to run dataset integration checks")

    config = example_config(preprocessing={"raw": {"train_path": str(data_root / "data" / "train.parquet")}})

    lineage = collect_tracking_lineage(config)

    assert lineage["dvc"]["enabled"] is True
    assert lineage["dvc"]["targets"]["data"]["outs"]


@pytest.mark.integration
def test_opt_in_modelctl_can_resolve_registry_source() -> None:
    tracking_uri = os.environ.get("ESTADEL_TEST_MLFLOW_URI")
    model_ref = os.environ.get("ESTADEL_TEST_MODEL_REF")
    if not tracking_uri or not model_ref:
        pytest.skip("set ESTADEL_TEST_MLFLOW_URI and ESTADEL_TEST_MODEL_REF to run registry integration checks")

    info = ModelctlClient(tracking_uri=tracking_uri, timeout_seconds=60).info(model_ref)

    assert info.ref == model_ref
    assert info.name or info.version or info.aliases
