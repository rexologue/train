from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import Config
from utils.hashing import file_sha256, stable_hash
from utils.logging import get_logger


# Bump when the on-disk cache layout or the meaning of an entry changes.
REF_CACHE_VERSION = 1

REF_CACHE_PARQUET_NAME = "ref_logps.parquet"
REF_CACHE_MANIFEST_NAME = "ref_logps.manifest.json"


@dataclass(frozen=True, slots=True)
class RefCachePaths:
    """Filesystem locations of the reference-logp cache artifacts."""

    parquet: Path
    manifest: Path


def ref_cache_paths(config: Config) -> RefCachePaths:
    """Return the ref-logp cache locations next to the pretokenized splits."""

    root = config.pretokenized_dir
    return RefCachePaths(parquet=root / REF_CACHE_PARQUET_NAME, manifest=root / REF_CACHE_MANIFEST_NAME)


def reference_signature(config: Config, model_source: Any) -> str:
    """Return a stable identity for the *reference* forward this cache encodes.

    A cached completion logp is a pure function of the base model weights and
    the exact pretokenized completion tokens. The token ids are addressed by the
    parquet ``render_hash`` key, but the *weights* and numerics are not, so they
    are pinned here. A different model payload, precision, attention kernel, or
    MoE expert kernel produces different reference logps for the same tokens, so
    any of those changing must invalidate the whole cache.
    """

    model_identity = (
        getattr(model_source, "source_dir_hash", None)
        or getattr(model_source, "expected_payload_hash", None)
        or getattr(model_source, "effective_model_id", None)
    )
    payload = {
        "version": REF_CACHE_VERSION,
        "model_identity": model_identity,
        "precision": config.model.precision,
        "attn_implementation": config.model.attn_implementation,
        "experts_implementation": config.model.experts_implementation,
    }
    return stable_hash(payload)


class RefLogpCache:
    """In-memory lookup of precomputed reference completion logps by render hash.

    The cache is consulted per batch. ``lookup`` returns ``None`` when *any*
    requested hash is absent, which the trainer treats as a whole-batch miss and
    falls back to the on-the-fly PEFT-reference forward. Whole-batch granularity
    keeps the loss math homogeneous: a batch is either fully cache-served or
    fully on-the-fly, never a mix.
    """

    def __init__(self, entries: dict[str, float], *, signature: str):
        self._entries = entries
        self.signature = signature

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, render_hash: Any) -> bool:
        return isinstance(render_hash, str) and render_hash in self._entries

    def lookup(self, render_hashes: list[Any]) -> list[float] | None:
        """Return logps for every hash, or ``None`` if any is missing/invalid."""

        resolved: list[float] = []
        for render_hash in render_hashes:
            if not isinstance(render_hash, str):
                return None
            value = self._entries.get(render_hash)
            if value is None:
                return None
            resolved.append(float(value))
        return resolved


def read_cache_manifest(paths: RefCachePaths) -> dict[str, Any] | None:
    """Load the ref-cache manifest if present and parseable."""

    if not paths.manifest.exists():
        return None
    try:
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return manifest if isinstance(manifest, dict) else None


def read_cache_entries(paths: RefCachePaths) -> dict[str, float]:
    """Read the render_hash -> ref_logp map from the cache parquet."""

    if not paths.parquet.exists():
        return {}
    frame = pd.read_parquet(paths.parquet)
    if "render_hash" not in frame.columns or "ref_logp" not in frame.columns:
        raise ValueError(f"ref-logp cache parquet missing required columns: {paths.parquet}")
    entries: dict[str, float] = {}
    for render_hash, ref_logp in zip(frame["render_hash"].tolist(), frame["ref_logp"].tolist()):
        entries[str(render_hash)] = float(ref_logp)
    return entries


def load_ref_logp_cache(config: Config, *, expected_signature: str) -> RefLogpCache | None:
    """Load a usable ref-logp cache, or ``None`` to fall back to on-the-fly.

    Returns ``None`` (never raises) for every "no usable cache" condition: no
    manifest, a signature built against a different model/precision, a missing
    or corrupted parquet. The caller logs the reason and proceeds with the
    on-the-fly PEFT reference exactly as before.
    """

    logger = get_logger(__name__)
    paths = ref_cache_paths(config)
    manifest = read_cache_manifest(paths)
    if manifest is None:
        return None

    cached_signature = manifest.get("signature")
    if cached_signature != expected_signature:
        logger.warning(
            "ignoring ref-logp cache: signature mismatch cached=%s expected=%s (model/precision changed); "
            "re-run precompute to refresh",
            cached_signature,
            expected_signature,
        )
        return None

    if not paths.parquet.exists():
        logger.warning("ignoring ref-logp cache: parquet missing at %s", paths.parquet)
        return None

    expected_parquet_hash = manifest.get("parquet_sha256")
    if isinstance(expected_parquet_hash, str) and file_sha256(paths.parquet) != expected_parquet_hash:
        logger.warning("ignoring ref-logp cache: parquet checksum mismatch at %s", paths.parquet)
        return None

    entries = read_cache_entries(paths)
    return RefLogpCache(entries, signature=expected_signature)


def load_reusable_entries(config: Config, *, expected_signature: str) -> dict[str, float]:
    """Return existing cache entries reusable for an incremental precompute.

    Only entries produced under a matching signature are reusable; if the model
    or numerics changed the whole cache is recomputed from scratch.
    """

    paths = ref_cache_paths(config)
    manifest = read_cache_manifest(paths)
    if manifest is None or manifest.get("signature") != expected_signature:
        return {}
    if isinstance(manifest.get("parquet_sha256"), str) and paths.parquet.exists():
        if file_sha256(paths.parquet) != manifest["parquet_sha256"]:
            return {}
    return read_cache_entries(paths)


def write_ref_logp_cache(
    config: Config,
    *,
    entries: dict[str, float],
    signature: str,
    model_source: Any,
    splits: list[str],
) -> RefCachePaths:
    """Atomically persist the ref-logp cache parquet and its manifest."""

    logger = get_logger(__name__)
    paths = ref_cache_paths(config)
    paths.parquet.parent.mkdir(parents=True, exist_ok=True)

    # Sort by hash so the parquet (and therefore its checksum) is deterministic.
    ordered = sorted(entries.items())
    frame = pd.DataFrame(ordered, columns=["render_hash", "ref_logp"])

    parquet_tmp = paths.parquet.with_name(f"{paths.parquet.stem}.tmp{paths.parquet.suffix}")
    logger.info("writing ref-logp cache parquet: entries=%s path=%s", len(ordered), paths.parquet)
    frame.to_parquet(parquet_tmp, index=False)
    parquet_tmp.replace(paths.parquet)

    parquet_hash = file_sha256(paths.parquet)
    manifest = {
        "version": REF_CACHE_VERSION,
        "signature": signature,
        "num_entries": len(ordered),
        "splits": sorted(splits),
        "parquet": paths.parquet.name,
        "parquet_sha256": parquet_hash,
        "model_source": {
            "effective_model_id": getattr(model_source, "effective_model_id", None),
            "ref": getattr(model_source, "ref", None),
            "source_dir_hash": getattr(model_source, "source_dir_hash", None),
            "resolved_version": getattr(model_source, "resolved_version", None),
        },
    }
    manifest_tmp = paths.manifest.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_tmp.replace(paths.manifest)
    logger.info("wrote ref-logp cache manifest: path=%s signature=%s", paths.manifest, signature)
    return paths
