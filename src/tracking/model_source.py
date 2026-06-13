from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import TrainingConfig


@dataclass(frozen=True)
class ModelSourceResolution:
    """Effective model source selected for this run."""

    kind: str
    effective_model_id: str
    ref: str | None = None
    model_name: str | None = None
    alias: str | None = None
    version: str | None = None
    local_dir: str | None = None
    pulled: bool = False
    used_local: bool = False
    verified_local_hash: bool = False
    verified_remote_ref: bool = False
    resolved_version: str | None = None
    source_dir_hash: str | None = None
    local_payload_hash: str | None = None
    metadata_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_model_source(config: TrainingConfig, *, tracking_uri: str | None = None) -> ModelSourceResolution:
    """Resolve a local/HF model id or pull a registry model only when local cache is empty."""

    model_config = config.section("model")
    source_config = model_config.get("source")
    if not isinstance(source_config, dict) or source_config.get("kind", "local_or_hf") == "local_or_hf":
        local_dir = source_config.get("local_dir") if isinstance(source_config, dict) else None
        if local_dir:
            resolved_local = str(Path(str(local_dir)).expanduser().resolve())
            return ModelSourceResolution(
                kind="local_or_hf",
                effective_model_id=resolved_local,
                local_dir=resolved_local,
                used_local=True,
            )
        base_model_id = str(model_config["base_model_id"])
        return ModelSourceResolution(kind="local_or_hf", effective_model_id=base_model_id)

    if source_config.get("kind") != "registry":
        raise ValueError("model.source.kind must be local_or_hf or registry")

    model_name = str(source_config["model_name"])
    alias = source_config.get("alias")
    version = source_config.get("version")
    ref = build_registry_ref(model_name, alias=alias, version=version)
    local_dir = Path(str(source_config["local_dir"])).expanduser().resolve()
    pull_policy = str(source_config.get("pull_policy", "if_local_empty"))
    if pull_policy != "if_local_empty":
        raise ValueError("only model.source.pull_policy=if_local_empty is supported")

    verify_local_hash = bool(source_config.get("verify_local_hash", True))
    verify_remote_ref = bool(source_config.get("verify_remote_ref", False))
    require_registry_metadata = bool(source_config.get("require_registry_metadata", True))
    effective_tracking_uri = tracking_uri or str(config.section("mlflow").get("tracking_uri"))

    if directory_has_payload(local_dir):
        return use_existing_registry_model(
            local_dir=local_dir,
            model_name=model_name,
            alias=str(alias) if alias is not None else None,
            version=str(version) if version is not None else None,
            ref=ref,
            tracking_uri=effective_tracking_uri,
            verify_local_hash=verify_local_hash,
            verify_remote_ref=verify_remote_ref,
            require_registry_metadata=require_registry_metadata,
        )

    return pull_registry_model_if_missing(
        local_dir=local_dir,
        model_name=model_name,
        alias=str(alias) if alias is not None else None,
        version=str(version) if version is not None else None,
        ref=ref,
        tracking_uri=effective_tracking_uri,
        require_registry_metadata=require_registry_metadata,
    )


def build_registry_ref(model_name: str, *, alias: Any, version: Any) -> str:
    """Build a models:/ URI from mutually exclusive alias/version fields."""

    has_alias = alias not in (None, "")
    has_version = version not in (None, "")
    if has_alias == has_version:
        raise ValueError("model.source must configure exactly one of alias or version")
    if has_alias:
        return f"models:/{model_name}@{alias}"
    return f"models:/{model_name}/{version}"


def directory_has_payload(path: Path) -> bool:
    """Return whether a configured local model directory already has content."""

    if not path.exists():
        return False
    if not path.is_dir():
        raise ValueError(f"model.source.local_dir must be a directory: {path}")
    return any(path.iterdir())


def use_existing_registry_model(
    *,
    local_dir: Path,
    model_name: str,
    alias: str | None,
    version: str | None,
    ref: str,
    tracking_uri: str,
    verify_local_hash: bool,
    verify_remote_ref: bool,
    require_registry_metadata: bool,
) -> ModelSourceResolution:
    """Use a non-empty local model directory and optionally verify its sidecar/hash."""

    sidecar_path = registry_metadata_path(local_dir)
    sidecar = read_registry_metadata(sidecar_path)
    if require_registry_metadata and not sidecar:
        raise FileNotFoundError(f"registry model sidecar is required for local cache verification: {sidecar_path}")

    local_hash: str | None = None
    source_hash = string_or_none(sidecar.get("source_dir_hash")) if sidecar else None
    resolved_version = string_or_none(sidecar.get("resolved_version")) if sidecar else None
    verified_local_hash = False
    verified_remote_ref = False

    if verify_local_hash:
        local_hash = _hash_directory(local_dir)
        if source_hash and local_hash != source_hash:
            raise ValueError(f"local model hash mismatch: expected {source_hash}, got {local_hash}")
        verified_local_hash = bool(source_hash)

    if verify_remote_ref:
        info = _get_model_info(ref, tracking_uri)
        remote_hash = extract_source_hash(info)
        remote_version = extract_version(info)
        if source_hash and remote_hash and source_hash != remote_hash:
            raise ValueError(f"local model cache does not match current registry ref: local={source_hash} remote={remote_hash}")
        if resolved_version and remote_version and resolved_version != remote_version:
            raise ValueError(f"local model cache version does not match current registry ref: local={resolved_version} remote={remote_version}")
        verified_remote_ref = bool(remote_hash or remote_version)

    return ModelSourceResolution(
        kind="registry",
        effective_model_id=str(local_dir),
        ref=ref,
        model_name=model_name,
        alias=alias,
        version=version,
        local_dir=str(local_dir),
        pulled=False,
        used_local=True,
        verified_local_hash=verified_local_hash,
        verified_remote_ref=verified_remote_ref,
        resolved_version=resolved_version,
        source_dir_hash=source_hash,
        local_payload_hash=local_hash,
        metadata_path=str(sidecar_path),
    )


def pull_registry_model_if_missing(
    *,
    local_dir: Path,
    model_name: str,
    alias: str | None,
    version: str | None,
    ref: str,
    tracking_uri: str,
    require_registry_metadata: bool,
) -> ModelSourceResolution:
    """Pull a registry model into an empty local directory and write a verification sidecar."""

    info = _get_model_info(ref, tracking_uri)
    source_hash = extract_source_hash(info)
    resolved_version = extract_version(info)
    if require_registry_metadata and not source_hash:
        raise ValueError(f"registry model {ref} has no modelctl.source_dir_hash metadata")

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if local_dir.exists() and directory_has_payload(local_dir):
        raise FileExistsError(f"model.source.local_dir became non-empty before pull: {local_dir}")

    with tempfile.TemporaryDirectory(prefix=f".{local_dir.name}.pull.", dir=str(local_dir.parent)) as tmp_root:
        temp_payload = Path(tmp_root) / "payload"
        _pull_model(ref, temp_payload, tracking_uri)
        if not directory_has_payload(temp_payload):
            raise ValueError(f"modelctl pull produced an empty payload for {ref}")
        local_hash = _hash_directory(temp_payload)
        if source_hash and local_hash != source_hash:
            raise ValueError(f"pulled model hash mismatch: expected {source_hash}, got {local_hash}")
        if local_dir.exists():
            local_dir.rmdir()
        shutil.move(str(temp_payload), str(local_dir))

    sidecar_path = registry_metadata_path(local_dir)
    sidecar = {
        "model_name": model_name,
        "ref": ref,
        "alias": alias,
        "version": version,
        "resolved_version": resolved_version,
        "source_dir_hash": source_hash,
        "local_payload_hash": local_hash,
        "tracking_uri": tracking_uri,
        "pulled_at": datetime.now(timezone.utc).isoformat(),
    }
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ModelSourceResolution(
        kind="registry",
        effective_model_id=str(local_dir),
        ref=ref,
        model_name=model_name,
        alias=alias,
        version=version,
        local_dir=str(local_dir),
        pulled=True,
        used_local=False,
        verified_local_hash=True,
        verified_remote_ref=True,
        resolved_version=resolved_version,
        source_dir_hash=source_hash,
        local_payload_hash=local_hash,
        metadata_path=str(sidecar_path),
    )


def registry_metadata_path(local_dir: Path) -> Path:
    """Return the sidecar path stored outside the model payload directory."""

    return local_dir.with_name(f"{local_dir.name}.estadel_registry.json")


def read_registry_metadata(path: Path) -> dict[str, Any]:
    """Read a registry sidecar if present."""

    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"registry sidecar must contain a JSON object: {path}")
    return data


def extract_source_hash(info: dict[str, Any]) -> str | None:
    """Extract modelctl's source directory hash from model info."""

    direct = string_or_none(info.get("source_dir_hash"))
    if direct:
        return direct
    tags = info.get("tags")
    if isinstance(tags, dict):
        return string_or_none(tags.get("modelctl.source_dir_hash"))
    return None


def extract_version(info: dict[str, Any]) -> str | None:
    """Extract resolved model version from model info."""

    return string_or_none(info.get("version"))


def string_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _get_model_info(ref: str, tracking_uri: str) -> dict[str, Any]:
    from modelctl.core import get_model_info

    return get_model_info(ref, tracking_uri=tracking_uri)


def _pull_model(ref: str, output_dir: Path, tracking_uri: str) -> Any:
    from modelctl.core import pull_model

    return pull_model(ref, output_dir, tracking_uri=tracking_uri)


def _hash_directory(path: Path) -> str:
    from modelctl.core import hash_directory

    return hash_directory(path)
