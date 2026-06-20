from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.dataloaders import DataLoaderBundle, SplitDataLoader
from utils.hashing import file_sha256, stable_hash


REF_LOGPROB_CACHE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RefLogprobCacheState:
    """Resolved reference-logprob cache location and application summary."""

    cache_dir: Path
    signature: str
    applied_rows: int
    missing_rows: int
    complete: bool


def build_ref_logprob_cache_signature(config: Any, model_source: Any | None) -> str:
    """Build the model-dependent cache signature for DPO reference logprobs."""

    return stable_hash(
        {
            "schema_version": REF_LOGPROB_CACHE_SCHEMA_VERSION,
            "model": {
                "ref": getattr(model_source, "ref", None),
                "expected_payload_hash": getattr(model_source, "expected_payload_hash", None),
                "source_dir_hash": getattr(model_source, "source_dir_hash", None),
                "cache_dir": str(config.model.cache_dir),
            },
            "reference": {
                "policy": "base_model_precompute",
                "ignore_index": config.ignore_index,
            },
        }
    )


def ref_logprob_cache_dir(config: Any, signature: str) -> Path:
    """Return the directory for one reference-logprob cache signature."""

    return config.output_dir / "ref_logprobs" / signature.replace(":", "-")


def split_cache_path(cache_dir: Path, split: str) -> Path:
    """Return the parquet path for one cached split."""

    return cache_dir / f"{split}.parquet"


def manifest_path(cache_dir: Path) -> Path:
    """Return the cache manifest path."""

    return cache_dir / "manifest.json"


def load_and_apply_ref_logprob_cache(
    config: Any,
    dataloaders: DataLoaderBundle,
    *,
    model_source: Any | None,
) -> RefLogprobCacheState:
    """Load any valid reference-logprob cache rows and attach them to datasets."""

    signature = build_ref_logprob_cache_signature(config, model_source)
    cache_dir = ref_logprob_cache_dir(config, signature)
    applied = 0
    missing = 0
    for split_loader in dataloaders.splits.values():
        cached_rows = load_valid_split_cache(cache_dir, split_loader)
        applied += apply_ref_logprobs_to_split(split_loader, cached_rows)
        missing += count_missing_dpo_ref_logprobs(split_loader)
    return RefLogprobCacheState(
        cache_dir=cache_dir,
        signature=signature,
        applied_rows=applied,
        missing_rows=missing,
        complete=missing == 0,
    )


def load_valid_split_cache(cache_dir: Path, split_loader: SplitDataLoader) -> list[dict[str, Any]]:
    """Read one split cache when its manifest still matches the pretokenized split."""

    if not split_cache_is_valid(cache_dir, split_loader):
        return []
    frame = pd.read_parquet(split_cache_path(cache_dir, split_loader.split))
    return frame.to_dict(orient="records")


def split_cache_is_valid(cache_dir: Path, split_loader: SplitDataLoader) -> bool:
    """Return whether one split cache matches the current pretokenized parquet."""

    path = split_cache_path(cache_dir, split_loader.split)
    manifest = load_manifest(cache_dir)
    if manifest is None or not path.exists():
        return False
    split_hashes = manifest.get("pretokenized")
    if not isinstance(split_hashes, dict):
        return False
    expected_hash = split_hashes.get(split_loader.split)
    return isinstance(expected_hash, str) and expected_hash == file_sha256(split_loader.path)


def load_manifest(cache_dir: Path) -> dict[str, Any] | None:
    """Load a cache manifest if present and valid JSON."""

    path = manifest_path(cache_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def apply_ref_logprobs_to_split(split_loader: SplitDataLoader, cached_rows: list[dict[str, Any]]) -> int:
    """Attach cached DPO reference logprobs to matching dataset rows."""

    if not cached_rows:
        return 0
    cache = {int(row["row_index"]): row for row in cached_rows}
    applied = 0
    for row in split_loader.dataset.rows:
        if row.get("loss_kind") != "dpo_target":
            continue
        cached = cache.get(int(row["row_index"]))
        if cached is None:
            continue
        if not cache_row_matches_dataset_row(cached, row):
            continue
        row["chosen_ref_logp"] = float(cached["chosen_ref_logp"])
        row["rejected_ref_logp"] = float(cached["rejected_ref_logp"])
        applied += 1
    return applied


def cache_row_matches_dataset_row(cached: dict[str, Any], row: dict[str, Any]) -> bool:
    """Return whether a cached row was computed for the same rendered branches."""

    return (
        str(cached.get("chosen_render_hash")) == str(row.get("chosen_render_hash"))
        and str(cached.get("rejected_render_hash")) == str(row.get("rejected_render_hash"))
    )


def count_missing_dpo_ref_logprobs(split_loader: SplitDataLoader) -> int:
    """Count DPO rows without both cached reference logprobs."""

    missing = 0
    for row in split_loader.dataset.rows:
        if row.get("loss_kind") != "dpo_target":
            continue
        if "chosen_ref_logp" not in row or "rejected_ref_logp" not in row:
            missing += 1
    return missing


def dpo_row_count(dataloaders: DataLoaderBundle) -> int:
    """Return the number of DPO rows across all available splits."""

    total = 0
    for split_loader in dataloaders.splits.values():
        total += sum(1 for row in split_loader.dataset.rows if row.get("loss_kind") == "dpo_target")
    return total


def write_ref_logprob_split_cache(
    cache_dir: Path,
    split_loader: SplitDataLoader,
    rows: list[dict[str, Any]],
    *,
    base_manifest: dict[str, Any] | None = None,
) -> None:
    """Write one split cache and update the cache manifest."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(split_cache_path(cache_dir, split_loader.split), index=False)
    manifest = dict(base_manifest or load_manifest(cache_dir) or {})
    pretokenized = dict(manifest.get("pretokenized") or {})
    rows_summary = dict(manifest.get("rows") or {})
    pretokenized[split_loader.split] = file_sha256(split_loader.path)
    rows_summary[split_loader.split] = len(rows)
    manifest.update(
        {
            "schema_version": REF_LOGPROB_CACHE_SCHEMA_VERSION,
            "pretokenized": pretokenized,
            "rows": rows_summary,
        }
    )
    manifest_path(cache_dir).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
