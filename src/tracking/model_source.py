from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from config import Config
from registry.modelctl_client import ModelctlClient, ModelctlInfo, ModelctlPullResult, ModelctlVerifyResult


@dataclass(frozen=True)
class ModelSourceResolution:
    """Effective model source selected for this run."""

    effective_model_id: str
    ref: str | None = None
    model_name: str | None = None
    alias: str | None = None
    local_dir: str | None = None
    pulled: bool = False
    used_local: bool = False
    verified_local_hash: bool = False
    verified_remote_ref: bool = False
    resolved_version: str | None = None
    expected_payload_hash: str | None = None
    local_payload_hash: str | None = None
    source_dir_hash: str | None = None
    metadata_path: str | None = None


    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def resolve_model_source(
    config: Config,
    *,
    tracking_uri: str | None = None,
    client: ModelctlClient | None = None,
) -> ModelSourceResolution:
    """Resolve the configured registry alias into a verified local model cache."""

    model_config = config.model
    model_name = model_config.name
    alias = model_config.alias
    ref = build_registry_ref(model_name, alias)
    local_dir = model_config.cache_dir.expanduser().resolve()
    effective_tracking_uri = tracking_uri or config.mlflow.tracking_uri
    modelctl = client or ModelctlClient(tracking_uri=effective_tracking_uri)

    info = modelctl.info(ref)
    if directory_has_payload(local_dir):
        verification = modelctl.verify(ref, local_dir)

        if verification.matches:
            return use_verified_local_model(
                local_dir=local_dir,
                model_name=model_name,
                alias=alias,
                info=info,
                verification=verification,
            )

        return pull_registry_model(
            local_dir=local_dir,
            model_name=model_name,
            alias=alias,
            info=info,
            modelctl=modelctl,
            overwrite=True,
        )

    return pull_registry_model(
        local_dir=local_dir,
        model_name=model_name,
        alias=alias,
        info=info,
        modelctl=modelctl,
        overwrite=local_dir.exists(),
    )


def build_registry_ref(model_name: str, alias: str) -> str:
    """Build the only supported source reference: a registry alias."""

    return f"models:/{model_name}@{alias}"


def directory_has_payload(path: Path) -> bool:
    """Return whether a configured local model directory already has content."""

    if not path.exists():
        return False
    if not path.is_dir():
        raise ValueError(f"model.cache_dir must be a directory: {path}")

    return any(path.iterdir())


def use_verified_local_model(
    *,
    local_dir: Path,
    model_name: str,
    alias: str | None,
    info: ModelctlInfo,
    verification: ModelctlVerifyResult,
) -> ModelSourceResolution:
    """Use a local model cache that modelctl verified against the registry."""

    validate_model_payload(local_dir)
    sidecar_path = registry_metadata_path(local_dir)
    sidecar = build_registry_sidecar(
        model_name=model_name,
        alias=alias,
        ref=verification.ref,
        info=info,
        verification=verification,
        pull=None,
    )
    write_registry_metadata(sidecar_path, sidecar)

    return ModelSourceResolution(
        effective_model_id=str(local_dir),
        ref=verification.ref,
        model_name=model_name,
        alias=alias,
        local_dir=str(local_dir),
        pulled=False,
        used_local=True,
        verified_local_hash=True,
        verified_remote_ref=True,
        resolved_version=info.version,
        expected_payload_hash=verification.expected_payload_hash,
        local_payload_hash=verification.actual_payload_hash,
        source_dir_hash=verification.expected_payload_hash,
        metadata_path=str(sidecar_path),
    )


def pull_registry_model(
    *,
    local_dir: Path,
    model_name: str,
    alias: str | None,
    info: ModelctlInfo,
    modelctl: ModelctlClient,
    overwrite: bool,
) -> ModelSourceResolution:
    """Pull a registry model into the configured local cache directory."""

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    pull = modelctl.pull(info.ref, local_dir, overwrite=overwrite)
    if not directory_has_payload(local_dir):
        raise ValueError(f"modelctl pull produced an empty payload for {info.ref}")
    validate_model_payload(local_dir)
    verification = modelctl.verify(info.ref, local_dir)
    if not verification.matches:
        raise ValueError(
            "pulled model did not verify against registry: "
            f"expected={verification.expected_payload_hash} actual={verification.actual_payload_hash}"
        )

    sidecar_path = registry_metadata_path(local_dir)
    sidecar = build_registry_sidecar(
        model_name=model_name,
        alias=alias,
        ref=info.ref,
        info=info,
        verification=verification,
        pull=pull,
    )
    write_registry_metadata(sidecar_path, sidecar)
    return ModelSourceResolution(
        effective_model_id=str(local_dir),
        ref=info.ref,
        model_name=model_name,
        alias=alias,
        local_dir=str(local_dir),
        pulled=True,
        used_local=False,
        verified_local_hash=True,
        verified_remote_ref=True,
        resolved_version=info.version,
        expected_payload_hash=verification.expected_payload_hash,
        local_payload_hash=verification.actual_payload_hash,
        source_dir_hash=verification.expected_payload_hash,
        metadata_path=str(sidecar_path),
    )


def registry_metadata_path(local_dir: Path) -> Path:
    """Return the sidecar path stored outside the model payload directory."""

    return local_dir.with_name(f"{local_dir.name}.sft_dpo_registry.json")


def build_registry_sidecar(
    *,
    model_name: str,
    alias: str | None,
    ref: str,
    info: ModelctlInfo,
    verification: ModelctlVerifyResult,
    pull: ModelctlPullResult | None,
) -> dict[str, Any]:
    """Build a local audit sidecar for the selected model cache."""

    return {
        "model_name": model_name,
        "ref": ref,
        "alias": alias,
        "resolved_version": info.version,
        "expected_payload_hash": verification.expected_payload_hash,
        "local_payload_hash": verification.actual_payload_hash,
        "source_dir_hash": verification.expected_payload_hash,
        "tracking_uri": pull.tracking_uri if pull is not None else None,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "modelctl_info": info.raw,
        "modelctl_verify": verification.raw,
        "modelctl_pull": pull.raw if pull is not None else None,
    }


def write_registry_metadata(path: Path, sidecar: dict[str, Any]) -> None:
    """Write model source audit metadata next to the local model cache."""

    path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_model_source_resolution_from_cache(config: Config) -> ModelSourceResolution:
    """Load the main-process model source resolution from its local sidecar."""

    model_config = config.model
    local_dir = model_config.cache_dir.expanduser().resolve()
    sidecar_path = registry_metadata_path(local_dir)
    if not sidecar_path.exists():
        raise FileNotFoundError(f"model source sidecar not found after main-process resolution: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict):
        raise ValueError(f"model source sidecar must be a JSON object: {sidecar_path}")
    expected_hash = string_or_none(sidecar.get("expected_payload_hash") or sidecar.get("source_dir_hash"))
    local_hash = string_or_none(sidecar.get("local_payload_hash"))
    return ModelSourceResolution(
        effective_model_id=str(local_dir),
        ref=string_or_none(sidecar.get("ref")),
        model_name=string_or_none(sidecar.get("model_name") or model_config.name),
        alias=string_or_none(sidecar.get("alias") or model_config.alias),
        local_dir=str(local_dir),
        pulled=False,
        used_local=True,
        verified_local_hash=bool(local_hash),
        verified_remote_ref=bool(expected_hash),
        resolved_version=string_or_none(sidecar.get("resolved_version")),
        expected_payload_hash=expected_hash,
        local_payload_hash=local_hash,
        source_dir_hash=string_or_none(sidecar.get("source_dir_hash") or expected_hash),
        metadata_path=str(sidecar_path),
    )


def string_or_none(value: Any) -> str | None:
    """Convert empty sidecar values to None and non-empty values to strings."""

    if value in (None, ""):
        return None
    return str(value)


def validate_model_payload(path: Path) -> None:
    """Reject incomplete Transformers payloads before expensive model loading."""

    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"model payload directory does not exist: {path}")

    found_weight_index = False
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name

        if not index_path.exists():
            continue

        found_weight_index = True
        
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"model weight index is not valid JSON: {index_path}") from exc

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"model weight index has no weight_map: {index_path}")

        referenced_files = sorted({str(value) for value in weight_map.values()})
        missing = [name for name in referenced_files if not (path / name).is_file()]

        if missing:
            preview = missing[:5]
            suffix = "" if len(missing) <= len(preview) else f" (+{len(missing) - len(preview)} more)"
            raise FileNotFoundError(f"incomplete model payload {path}: missing weight shards {preview}{suffix}")

    if (path / "config.json").exists() and not found_weight_index:
        weight_files = [*path.glob("*.safetensors"), *path.glob("pytorch_model*.bin")]
        if not any(candidate.is_file() for candidate in weight_files):
            raise FileNotFoundError(f"incomplete model payload {path}: no Transformers weight files found")
