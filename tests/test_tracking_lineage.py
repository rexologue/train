from __future__ import annotations

import subprocess

from config import load_config
from tracking.lineage import collect_tracking_lineage
from tracking.params import flatten_config_params


def test_collect_tracking_lineage_reads_dvc_file_without_dvc_package(tmp_path):
    repo = tmp_path / "dataset"
    repo.mkdir()
    (repo / "data.dvc").write_text(
        "outs:\n"
        "- md5: abc123.dir\n"
        "  size: 12\n"
        "  nfiles: 2\n"
        "  hash: md5\n"
        "  path: data\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repo), "init"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "tests@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Tests"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "data.dvc"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, stdout=subprocess.DEVNULL)

    config = load_config("configs/config.example.yaml")
    train_path = repo / "data" / "train.parquet"
    config.raw["preprocessing"]["raw"]["train_path"] = str(train_path)

    lineage = collect_tracking_lineage(config)

    assert lineage["dvc"]["enabled"] is True
    assert lineage["dvc"]["git"]["dirty"] is False
    out = lineage["dvc"]["targets"]["data"]["outs"][0]
    assert out["md5"] == "abc123.dir"
    assert out["path"] == "data"


def test_collect_tracking_lineage_is_optional_when_data_dvc_is_missing(tmp_path):
    config = load_config("configs/config.example.yaml")
    config.raw["preprocessing"]["raw"]["train_path"] = str(tmp_path / "data" / "train.parquet")

    lineage = collect_tracking_lineage(config)

    assert lineage["dvc"]["enabled"] is False
    assert len(lineage["dvc"]["searched"]) == 2


def test_flatten_config_params_drops_secret_like_keys():
    params = flatten_config_params({"mlflow": {"tracking_uri": "http://example", "password": "hidden"}, "x": 1})

    assert params["mlflow.tracking_uri"] == "http://example"
    assert params["x"] == "1"
    assert "mlflow.password" not in params
    assert "config_hash" in params
