from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow

from config import Config
from preprocessing.io import PretokSplitResult, debug_path, load_manifest
from tracking.async_worker import AsyncTrackingWorker
from tracking.lineage import collect_code_metadata, collect_tracking_lineage
from tracking.model_source import ModelSourceResolution, resolve_model_source
from tracking.params import flatten_config_params


class ExperimentTracker:
    """Single facade for MLflow logging, data lineage, and model source metadata."""

    def __init__(self, config: Config):
        self.config = config
        self.enabled = True
        self.tracking_uri = config.mlflow.tracking_uri
        self.mlflow: Any = None
        self.run: Any = None
        self.model_source_resolution: ModelSourceResolution | None = None

    @classmethod
    def from_config(cls, config: Config) -> "ExperimentTracker":
        return cls(config)

    def __enter__(self) -> "ExperimentTracker":
        if not self.enabled:
            return self

        self.mlflow = mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.config.project.name)
        resume_run_id = self.config.mlflow.resume_run_id
        run_name = self.config.project.run_name
        self.run = mlflow.start_run(run_id=resume_run_id or None, run_name=str(run_name) if run_name else None)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.enabled and self.mlflow is not None:
            self.mlflow.end_run(status="FAILED" if exc_type is not None else "FINISHED")

    def resolve_model_source(self) -> ModelSourceResolution:
        """Resolve/pull the configured model source and log the result."""

        resolution = resolve_model_source(self.config, tracking_uri=self.tracking_uri or None)
        self.model_source_resolution = resolution
        self.log_model_source_resolution(resolution)
        return resolution

    def log_run_start(self, *, config_path: str | Path | None = None) -> None:
        """Log effective config, run-level params, and code metadata."""

        if not self.enabled:
            return
        config_dict = self.config.to_dict()
        params = flatten_config_params(config_dict)
        if config_path is not None:
            params["config.path"] = str(config_path)
        self._log_params(params)

        code = collect_code_metadata(Path.cwd())
        resolved_model_id = (
            self.model_source_resolution.effective_model_id if self.model_source_resolution is not None else None
        )
        tags = {
            "stage": "training_pipeline",
            "project.name": self.config.project.name,
            "project.run_name": str(self.config.project.run_name),
            "model.registry_name": self.config.model.name,
            "model.registry_alias": self.config.model.alias,
            "model.resolved_model_id": str(resolved_model_id),
            "code.git_commit": str(code.get("commit")),
            "code.git_dirty": "true" if code.get("dirty") else "false",
        }
        self.mlflow.set_tags(tags)
        self.mlflow.log_dict(config_dict, "config/effective_config.json")
        self.mlflow.log_dict({"code": code}, "lineage/code.json")

    def log_lineage(self) -> dict[str, Any]:
        """Collect and log DVC dataset lineage."""

        lineage = collect_tracking_lineage(self.config)
        if not self.enabled:
            return lineage
        self.mlflow.log_dict(lineage, "lineage/dvc.json")
        dvc = lineage.get("dvc") if isinstance(lineage, dict) else None
        if isinstance(dvc, dict) and dvc.get("enabled"):
            tags: dict[str, str] = {}
            git = dvc.get("git")
            if isinstance(git, dict):
                tags["data.git_commit"] = str(git.get("commit"))
                tags["data.git_dirty"] = "true" if git.get("dirty") else "false"
            targets = dvc.get("targets")
            if isinstance(targets, dict):
                for name, target in targets.items():
                    outs = target.get("outs") if isinstance(target, dict) else None
                    if isinstance(outs, list) and outs:
                        first = outs[0]
                        if isinstance(first, dict) and first.get("md5"):
                            tags[f"data.dvc.{name}.md5"] = str(first["md5"])
            if tags:
                self.mlflow.set_tags(tags)
        return lineage

    def log_model_source_resolution(self, resolution: ModelSourceResolution) -> None:
        """Log registry model resolution."""

        if not self.enabled:
            return
        data = resolution.to_dict()
        self.mlflow.log_dict(data, "model/source_resolution.json")
        tags = {
            "model.registry_source": "true",
            "model.effective_model_id": resolution.effective_model_id,
        }
        if resolution.ref:
            tags["model.registry_ref"] = resolution.ref
        if resolution.resolved_version:
            tags["model.resolved_version"] = resolution.resolved_version
        if resolution.expected_payload_hash:
            tags["model.expected_payload_hash"] = resolution.expected_payload_hash
        if resolution.source_dir_hash:
            tags["model.source_dir_hash"] = resolution.source_dir_hash
        self.mlflow.set_tags(tags)

    def log_preprocessing_results(self, results: list[PretokSplitResult]) -> None:
        """Log split manifest data and preprocessing artifacts."""

        if not self.enabled:
            return
        manifests_logged: set[str] = set()
        summaries: dict[str, Any] = {}
        for result in results:
            manifest = result.manifest
            split = result.split
            summaries[split] = {
                "raw_path": str(result.raw_path),
                "pretok_path": str(result.pretok_path),
                "reused": result.reused,
                "manifest": manifest,
            }
            self.mlflow.log_metric(f"preprocessing/{split}/rows_processed", int(manifest.get("num_rows") or 0))
            self.mlflow.log_metric(f"preprocessing/{split}/rows_rejected", int(manifest.get("num_rejected_rows") or 0))
            self.mlflow.log_metric(f"preprocessing/{split}/rows_raw", int(manifest.get("num_raw_rows") or 0))
            stats = manifest.get("stats")
            if isinstance(stats, dict):
                for key in ("tokens", "supervised_tokens", "rejected_rows"):
                    if key in stats:
                        self.mlflow.log_metric(f"preprocessing/{split}/{key}", float(stats[key]))
            manifest_key = str(result.manifest_path)
            if result.manifest_path.exists() and manifest_key not in manifests_logged:
                self.mlflow.log_artifact(str(result.manifest_path), artifact_path="preprocessing")
                full_manifest = load_manifest(result.output_dir)
                if full_manifest is not None:
                    self.mlflow.log_dict(full_manifest, "preprocessing/manifest.json")
                manifests_logged.add(manifest_key)
            if self.config.mlflow.log_rendered_samples:
                path = debug_path(result.output_dir)
                if path.exists():
                    self.mlflow.log_artifact(str(path), artifact_path="preprocessing")
        self.mlflow.log_dict(summaries, "preprocessing/split_results.json")

    def log_dataloaders(self, dataloaders: Any) -> None:
        """Log routed DataLoader summaries."""

        if not self.enabled:
            return
        summaries = {split: split_loader.summary for split, split_loader in dataloaders.splits.items()}
        self.mlflow.log_dict(summaries, "data/dataloaders.json")
        for split, summary in summaries.items():
            self.mlflow.log_metric(f"dataloader/{split}/rows", int(summary.get("num_rows") or 0))
            self.mlflow.log_metric(f"dataloader/{split}/batches", int(summary.get("num_batches") or 0))
            self.mlflow.log_metric(f"dataloader/{split}/short_batches", int(summary.get("num_short_batches") or 0))
            loss_kind_counts = summary.get("loss_kind_counts")
            if isinstance(loss_kind_counts, dict):
                for loss_kind, count in loss_kind_counts.items():
                    self.mlflow.log_metric(f"dataloader/{split}/loss_kind/{loss_kind}", int(count))

    def create_async_worker(self) -> AsyncTrackingWorker | None:
        """Create a CPU-only async worker for metrics, artifacts, and registry jobs."""

        async_config = self.config.mlflow.async_logging
        if not self.enabled or not async_config.enabled:
            return None
        if self.run is None:
            raise RuntimeError("MLflow run must be active before creating async tracking worker")
        return AsyncTrackingWorker(
            tracking_uri=self.tracking_uri,
            run_id=self.run.info.run_id,
            queue_max_items=async_config.queue_max_items,
            flush_timeout_seconds=float(async_config.flush_timeout_seconds),
            fail_on_worker_error=async_config.fail_on_worker_error,
        )

    def _log_params(self, params: dict[str, str]) -> None:
        """Log params in small batches for MLflow backends with request limits."""

        items = list(params.items())
        for offset in range(0, len(items), 100):
            self.mlflow.log_params(dict(items[offset : offset + 100]))


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
