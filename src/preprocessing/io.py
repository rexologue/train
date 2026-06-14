from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import TrainingConfig, file_sha256, stable_json_dumps


LOSS_KINDS = {"sft_target", "sft_tool", "dpo_target"}


class ParquetSchemaError(ValueError):
    """Raised when raw parquet rows do not match the `data` + `type|target` contract."""


@dataclass(frozen=True)
class PretokSplitResult:
    """Prepared pretokenized split location plus manifest summary."""

    split: str
    raw_path: Path
    output_dir: Path
    pretok_path: Path
    manifest_path: Path
    reused: bool
    manifest: dict[str, Any]


def resolve_split_paths(config: TrainingConfig, splits: list[str]) -> list[tuple[str, Path]]:
    """Resolve configured split parquet paths, skipping optional missing test split."""

    raw_config = config.preprocessing["raw"]
    resolved: list[tuple[str, Path]] = []
    for split in splits:
        raw_value = raw_config.get(f"{split}_path")
        if raw_value is None:
            if split == "test" and not raw_config.get("test_required", False):
                continue
            raise FileNotFoundError(f"data.raw.{split}_path is not configured")
        raw_path = Path(raw_value)
        if not raw_path.exists():
            if split == "test" and not raw_config.get("test_required", False):
                continue
            raise FileNotFoundError(f"{split} raw parquet not found: {raw_path}")
        resolved.append((split, raw_path))
    return resolved


def read_raw_dataframe(raw_path: Path) -> Any:
    """Read raw parquet and validate the authoritative `data` + `type|target` wrapper.

    New parquet files should use `type`. Existing files may still use `target`.
    If both columns are present, `type` wins, but any row-level value conflict is
    a schema error because it makes `loss_kind` ambiguous.
    """

    import pandas as pd

    frame = pd.read_parquet(raw_path)
    columns = set(frame.columns)
    has_type = "type" in columns
    has_target = "target" in columns
    if "data" not in columns or not (has_type or has_target):
        raise ParquetSchemaError(f"{raw_path}: raw parquet must contain data and one of type/target")
    if has_type and has_target:
        conflicts = frame[frame["type"] != frame["target"]]
        if len(conflicts):
            first = int(conflicts.index[0])
            raise ParquetSchemaError(f"{raw_path}: row {first} has conflicting type and target values")
    return frame


def dataframe_to_rows(frame: Any) -> list[dict[str, Any]]:
    """Decode dataframe rows into `{row_index, loss_kind, payload}` dictionaries."""

    kind_column = "type" if "type" in frame.columns else "target"
    rows: list[dict[str, Any]] = []
    for row_index, record in enumerate(frame.to_dict(orient="records")):
        data = record["data"]
        if not isinstance(data, str):
            raise ParquetSchemaError(f"row {row_index}: data must be a JSON string")
        loss_kind = record[kind_column]
        if loss_kind not in LOSS_KINDS:
            raise ParquetSchemaError(f"row {row_index}: unsupported {kind_column}={loss_kind!r}")
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ParquetSchemaError(f"row {row_index}: data JSON must decode to an object")
        rows.append({"row_index": row_index, "loss_kind": loss_kind, "payload": payload, "kind_column": kind_column})
    return rows


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Compatibility helper: read raw parquet into decoded payload rows with `loss_kind`."""

    rows: list[dict[str, Any]] = []
    for row in dataframe_to_rows(read_raw_dataframe(Path(path))):
        payload = dict(row["payload"])
        payload["loss_kind"] = row["loss_kind"]
        metadata = dict(payload.get("metadata") or {})
        metadata["parquet_row_index"] = row["row_index"]
        metadata["parquet_type_column"] = row["kind_column"]
        payload["metadata"] = metadata
        rows.append(payload)
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to parquet."""

    import pandas as pd

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write a deterministic JSON manifest."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(stable_json_dumps(manifest) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows, returning an empty list for missing files."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cache_root(config: TrainingConfig) -> Path:
    """Return the pretokenized cache below the project output root."""

    return config.pretokenized_dir


def split_parquet_path(root: Path, split: str) -> Path:
    """Return `{root}/{split}.parquet`."""

    return root / f"{split}.parquet"


def manifest_path(root: Path) -> Path:
    """Return `{root}/manifest.json`."""

    return root / "manifest.json"


def debug_path(root: Path) -> Path:
    """Return `{root}/debug.jsonl`."""

    return root / "debug.jsonl"


def load_manifest(root: Path) -> dict[str, Any] | None:
    """Load cache manifest if present and parseable."""

    path = manifest_path(root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def split_cache_is_valid(
    root: Path,
    split: str,
    raw_hash: str,
    preprocessing_signature: str | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    """Return whether one cached split matches its raw data, processing contract, and content hash."""

    manifest = load_manifest(root)
    if manifest is None:
        return False, None
    split_entry = (manifest.get("splits") or {}).get(split)
    if isinstance(split_entry, dict):
        split_hash = split_entry.get("input_sha256")
    else:
        split_hash = split_entry
    if not isinstance(split_hash, str):
        return False, manifest
    pretok_path = split_parquet_path(root, split)
    expected_pretok_hash = (manifest.get("pretokenized") or {}).get(split)
    if not isinstance(expected_pretok_hash, str) or not pretok_path.exists():
        return False, manifest
    if file_sha256(pretok_path) != expected_pretok_hash:
        return False, manifest
    if preprocessing_signature is not None:
        signatures = manifest.get("preprocessing_signatures")
        cached_signature = signatures.get(split) if isinstance(signatures, dict) else None
        if cached_signature != preprocessing_signature:
            return False, manifest
    valid = split_hash == raw_hash
    return valid, manifest


def load_pretokenized_split_results(config: TrainingConfig, splits: list[str]) -> list[PretokSplitResult]:
    """Load prepared split result descriptors from an existing cache manifest."""

    root = cache_root(config)
    manifest = load_manifest(root)
    if manifest is None:
        raise FileNotFoundError(f"pretokenized manifest not found: {manifest_path(root)}")

    raw_paths = dict(resolve_split_paths(config, splits))
    results: list[PretokSplitResult] = []
    for split, raw_path in raw_paths.items():
        pretok_path = split_parquet_path(root, split)
        if not pretok_path.exists():
            raise FileNotFoundError(f"pretokenized split not found after preprocessing: {pretok_path}")
        split_manifest = split_manifest_from_cache_manifest(manifest, split)
        results.append(
            PretokSplitResult(
                split=split,
                raw_path=raw_path,
                output_dir=root,
                pretok_path=pretok_path,
                manifest_path=manifest_path(root),
                reused=True,
                manifest=split_manifest,
            )
        )
    return results


def split_manifest_from_cache_manifest(manifest: dict[str, Any], split: str) -> dict[str, Any]:
    rows = manifest.get("rows") if isinstance(manifest.get("rows"), dict) else {}
    row_summary = rows.get(split) if isinstance(rows, dict) else None
    row_summary = row_summary if isinstance(row_summary, dict) else {}
    return {
        "input_sha256": (manifest.get("splits") or {}).get(split),
        "pretokenized_sha256": (manifest.get("pretokenized") or {}).get(split),
        "num_raw_rows": int(row_summary.get("raw") or 0),
        "num_rows": int(row_summary.get("processed") or 0),
        "num_rejected_rows": int(row_summary.get("rejected") or 0),
        "rejected_counts": ((manifest.get("rejections") or {}).get(split) or {}),
    }


def sample_debug_rows(debug_rows: list[dict[str, Any]], *, examples_per_loss_kind: int) -> list[dict[str, Any]]:
    """Keep at most N debug examples per `(split, loss_kind)` pair."""

    counts: Counter[tuple[str, str]] = Counter()
    sampled: list[dict[str, Any]] = []
    for row in debug_rows:
        key = (str(row.get("split")), str(row.get("loss_kind")))
        if counts[key] >= examples_per_loss_kind:
            continue
        sampled.append(row)
        counts[key] += 1
    return sampled


def write_split_cache(
    root: Path,
    split: str,
    rows: list[dict[str, Any]],
    debug_rows: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    *,
    base_manifest: dict[str, Any],
    examples_per_loss_kind: int,
) -> dict[str, Any]:
    """Persist one split parquet and update the flat split-hash manifest.

    The output contract is intentionally flat: `{root}/{split}.parquet`,
    `{root}/debug.jsonl`, and `{root}/manifest.json`. Rewriting one split keeps
    debug samples from other splits and replaces only samples for this split.
    Cache validity is deliberately split-local: changing `valid.parquet`
    invalidates only `valid`, because `manifest["splits"][split]` stores the
    raw input hash for that exact split.
    """

    import pandas as pd

    root.mkdir(parents=True, exist_ok=True)
    pretok_path = split_parquet_path(root, split)
    pretok_tmp = pretok_path.with_name(f"{pretok_path.stem}.tmp{pretok_path.suffix}")
    pd.DataFrame(rows).to_parquet(pretok_tmp, index=False)
    pretok_tmp.replace(pretok_path)

    pretok_hash = file_sha256(pretok_path)

    splits = dict((base_manifest.get("splits") or {}))
    splits[split] = split_manifest["input_sha256"]
    pretokenized = dict((base_manifest.get("pretokenized") or {}))
    pretokenized[split] = pretok_hash
    preprocessing_signatures = dict((base_manifest.get("preprocessing_signatures") or {}))
    if split_manifest.get("preprocessing_signature"):
        preprocessing_signatures[split] = split_manifest["preprocessing_signature"]
    rows_summary = dict((base_manifest.get("rows") or {}))
    rows_summary[split] = {
        "raw": split_manifest["num_raw_rows"],
        "processed": split_manifest["num_rows"],
        "rejected": split_manifest["num_rejected_rows"],
    }
    rejections = dict((base_manifest.get("rejections") or {}))
    if split_manifest.get("rejected_counts"):
        rejections[split] = split_manifest["rejected_counts"]
    else:
        rejections.pop(split, None)
    outputs = {name: str(split_parquet_path(root, name)) for name in ("train", "valid", "test") if name in splits}
    outputs["debug"] = str(debug_path(root))
    outputs["manifest"] = str(manifest_path(root))

    manifest = {
        "splits": splits,
        "pretokenized": pretokenized,
        "preprocessing_signatures": preprocessing_signatures,
        "rows": rows_summary,
        "outputs": outputs,
    }
    if rejections:
        manifest["rejections"] = rejections

    previous_debug_rows = [row for row in read_jsonl(debug_path(root)) if row.get("split") != split]
    current_debug_rows = sample_debug_rows(debug_rows, examples_per_loss_kind=examples_per_loss_kind)
    all_debug_rows = previous_debug_rows + current_debug_rows
    write_jsonl(debug_path(root), all_debug_rows)
    manifest["debug"] = {
        "path": str(debug_path(root)),
        "examples_per_loss_kind_per_split": examples_per_loss_kind,
        "num_rows": len(all_debug_rows),
    }
    manifest_tmp = manifest_path(root).with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest_tmp.replace(manifest_path(root))
    return manifest
