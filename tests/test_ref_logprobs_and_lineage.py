from __future__ import annotations

import subprocess

import pandas as pd
import pytest

from data.dataloaders import build_dataloaders
from data.ref_logprobs import (
    build_ref_logprob_cache_signature,
    load_and_apply_ref_logprob_cache,
    ref_logprob_cache_dir,
    write_ref_logprob_split_cache,
)
import trainer.ref_logprobs as ref_logprob_pipeline
from trainer.ref_logprobs import estimate_ref_logprob_work
from tracking.lineage import collect_tracking_lineage
from tracking.params import flatten_config_params
from conftest import example_config, pretok_result


def write_dpo_pretok(path) -> None:
    pd.DataFrame(
        [
            {
                "sample_id": "dpo-0",
                "row_index": 7,
                "loss_kind": "dpo_target",
                "chosen_input_ids": [1, 2],
                "chosen_attention_mask": [1, 1],
                "chosen_labels": [-100, 2],
                "chosen_length": 2,
                "chosen_completion_token_count": 1,
                "chosen_render_hash": "chosen-hash",
                "rejected_input_ids": [1, 3],
                "rejected_attention_mask": [1, 1],
                "rejected_labels": [-100, 3],
                "rejected_length": 2,
                "rejected_completion_token_count": 1,
                "rejected_render_hash": "rejected-hash",
            }
        ]
    ).to_parquet(path, index=False)


def test_ref_logprob_cache_is_loaded_and_applied_to_dpo_rows(tmp_path) -> None:
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    write_dpo_pretok(train_path)
    write_dpo_pretok(valid_path)
    config = example_config(
        project={"output_dir": str(tmp_path / "run")},
        training={"per_device_train_batch_size": 1},
    )
    bundle = build_dataloaders(
        config,
        [pretok_result("train", train_path), pretok_result("valid", valid_path)],
        pad_token_id=0,
    )
    signature = build_ref_logprob_cache_signature(config, model_source=None)
    cache_dir = ref_logprob_cache_dir(config, signature)
    for split_loader in bundle.splits.values():
        write_ref_logprob_split_cache(
            cache_dir,
            split_loader,
            [
                {
                    "sample_id": "dpo-0",
                    "row_index": 7,
                    "chosen_render_hash": "chosen-hash",
                    "rejected_render_hash": "rejected-hash",
                    "chosen_ref_logp": -1.25,
                    "rejected_ref_logp": -2.5,
                }
            ],
        )

    state = load_and_apply_ref_logprob_cache(config, bundle, model_source=None)

    assert state.complete is True
    assert state.applied_rows == 2
    assert state.missing_rows == 0
    for split_loader in bundle.splits.values():
        row = split_loader.dataset.rows[0]
        assert row["chosen_ref_logp"] == -1.25
        assert row["rejected_ref_logp"] == -2.5


def test_ref_logprob_pipeline_reuses_complete_cache(tmp_path, monkeypatch) -> None:
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    write_dpo_pretok(train_path)
    write_dpo_pretok(valid_path)
    config = example_config(
        project={"output_dir": str(tmp_path / "run")},
        training={"per_device_train_batch_size": 1},
    )
    bundle = build_dataloaders(
        config,
        [pretok_result("train", train_path), pretok_result("valid", valid_path)],
        pad_token_id=0,
    )
    signature = build_ref_logprob_cache_signature(config, model_source=None)
    cache_dir = ref_logprob_cache_dir(config, signature)
    for split_loader in bundle.splits.values():
        write_ref_logprob_split_cache(
            cache_dir,
            split_loader,
            [
                {
                    "sample_id": "dpo-0",
                    "row_index": 7,
                    "chosen_render_hash": "chosen-hash",
                    "rejected_render_hash": "rejected-hash",
                    "chosen_ref_logp": -1.25,
                    "rejected_ref_logp": -2.5,
                }
            ],
        )

    def fail_compute(**kwargs):
        del kwargs
        raise AssertionError("complete reference cache should be reused")

    monkeypatch.setattr(ref_logprob_pipeline, "compute_ref_logprob_cache", fail_compute)
    state = ref_logprob_pipeline.ensure_ref_logprob_cache(
        config=config,
        dataloaders=bundle,
        accelerator=object(),
        model_source=None,
    )

    assert state.complete is True
    assert state.applied_rows == 2


def test_ref_logprob_work_estimate_reports_forward_batches(tmp_path) -> None:
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    write_dpo_pretok(train_path)
    write_dpo_pretok(valid_path)
    config = example_config(
        project={"output_dir": str(tmp_path / "run")},
        training={"per_device_train_batch_size": 1},
    )
    bundle = build_dataloaders(
        config,
        [pretok_result("train", train_path), pretok_result("valid", valid_path)],
        pad_token_id=0,
    )

    estimate = estimate_ref_logprob_work(bundle, num_processes=2)

    assert estimate["rows"] == 2
    assert estimate["chosen_tokens"] == 4
    assert estimate["rejected_tokens"] == 4
    assert estimate["forward_batches_per_rank"] == 4
    assert estimate["total_forward_calls"] == 8


def test_ref_logprob_pipeline_rejects_disabled_cache_with_dpo_rows(tmp_path) -> None:
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    write_dpo_pretok(train_path)
    write_dpo_pretok(valid_path)
    config = example_config(
        project={"output_dir": str(tmp_path / "run")},
        training={"per_device_train_batch_size": 1},
        loss_routing={"dpo": {"reference": {"cache_enabled": False}}},
    )
    bundle = build_dataloaders(
        config,
        [pretok_result("train", train_path), pretok_result("valid", valid_path)],
        pad_token_id=0,
    )

    with pytest.raises(ValueError, match="cache_enabled=true"):
        ref_logprob_pipeline.ensure_ref_logprob_cache(
            config=config,
            dataloaders=bundle,
            accelerator=object(),
            model_source=None,
        )


def test_collect_tracking_lineage_reads_data_dvc_without_dvc_package(tmp_path) -> None:
    repo = tmp_path / "dataset"
    data_dir = repo / "data"
    data_dir.mkdir(parents=True)
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
    config = example_config(preprocessing={"raw": {"train_path": str(data_dir / "train.parquet")}})

    lineage = collect_tracking_lineage(config)

    assert lineage["dvc"]["enabled"] is True
    assert lineage["dvc"]["git"]["dirty"] is False
    out = lineage["dvc"]["targets"]["data"]["outs"][0]
    assert out["md5"] == "abc123.dir"
    assert out["path"] == "data"


def test_flatten_config_params_drops_secret_like_keys() -> None:
    params = flatten_config_params({"mlflow": {"tracking_uri": "http://example", "password": "hidden"}, "x": 1})

    assert params["mlflow.tracking_uri"] == "http://example"
    assert params["x"] == "1"
    assert "mlflow.password" not in params
    assert "config_hash" in params
