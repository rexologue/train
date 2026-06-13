"""Core implementation for the ``modelctl`` command line utility.

The module intentionally keeps MLflow interactions explicit and boring. Each
registration operation creates a short technical run in a dedicated experiment,
logs a model artifact, creates a Model Registry version, attaches aliases, and
stores metadata in both searchable tags and JSON artifacts.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .generic_pyfunc import GenericDirectoryPyFunc
from .tags import flatten_for_mlflow_tags

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5000
DEFAULT_EXPERIMENT_NAME = "__model_registry_uploads__"
DEFAULT_MODEL_ARTIFACT_NAME = "model"
DEFAULT_SCHEMA_VERSION = "1.0"

ModelKind = Literal["generic", "hf", "pytorch"]


@dataclass(frozen=True)
class RegisterResult:
    """Result returned after a successful model registration."""

    name: str
    version: str
    aliases: list[str]
    kind: str
    run_id: str
    model_uri: str
    source_uri: str
    source_dir_hash: str
    tracking_uri: str


@dataclass(frozen=True)
class PullResult:
    """Result returned after a successful model pull."""

    ref: str
    model_uri: str
    downloaded_path: str | None
    output_path: str
    payload_only: bool


@dataclass(frozen=True)
class ModelVersionSummary:
    """Small printable summary for one MLflow model version."""

    name: str
    version: str
    aliases: list[str]
    status: str | None
    run_id: str | None
    source: str | None
    kind: str | None
    source_dir_hash: str | None
    created_at: str | None


def configure_mlflow(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, tracking_uri: str | None = None) -> str:
    """Configure the MLflow Tracking URI used by the utility.

    Parameters
    ----------
    host:
        Tracking server host. Ignored when ``tracking_uri`` is passed.
    port:
        Tracking server port. Ignored when ``tracking_uri`` is passed.
    tracking_uri:
        Full MLflow tracking URI. This is useful for HTTPS endpoints, custom
        paths, Databricks, or local SQLite/file stores.

    Returns
    -------
    str
        The effective tracking URI.

    Notes
    -----
    Authentication is intentionally not handled in code. MLflow already reads
    ``MLFLOW_TRACKING_USERNAME`` and ``MLFLOW_TRACKING_PASSWORD`` from the
    environment for HTTP Basic authentication.
    """

    effective_uri = tracking_uri or f"http://{host}:{port}"
    mlflow.set_tracking_uri(effective_uri)
    return effective_uri


def register_model_directory(
    source_dir: str | Path,
    name: str,
    *,
    kind: ModelKind = "generic",
    aliases: Iterable[str] | None = None,
    general_tags: dict[str, Any] | None = None,
    training_tags: dict[str, Any] | None = None,
    description: str | None = None,
    hf_task: str | None = None,
    pytorch_file: str | Path | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tracking_uri: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> RegisterResult:
    """Register a new MLflow Model Registry version from a local directory.

    Parameters
    ----------
    source_dir:
        Local model directory. For ``kind='generic'`` this can contain anything.
        For ``kind='hf'`` it must be a Hugging Face Transformers-compatible local
        checkpoint directory, usually containing ``config.json``. For
        ``kind='pytorch'`` it must contain a TorchScript file or you must pass
        ``pytorch_file``.
    name:
        Registered model name in MLflow.
    kind:
        Registration mode: ``generic``, ``hf`` or ``pytorch``. Generic mode is
        always directory-based and wraps the payload in a modelctl PyFunc model.
    aliases:
        Aliases to point at the newly created version. When omitted, the first
        version receives ``baseline`` and ``champion``; later versions receive
        ``candidate``.
    general_tags:
        Optional free-form metadata not tied to training. Full content is logged
        as JSON artifact and a flattened searchable projection is written to
        model version tags under the ``general.`` prefix.
    training_tags:
        Optional free-form metadata tied to training, datasets, metrics, code,
        environment, or experiment settings. Full content is logged as JSON and
        flattened under the ``training.`` prefix.
    description:
        Optional description for the created model version.
    hf_task:
        Optional Transformers task passed to ``mlflow.transformers.log_model``.
    pytorch_file:
        Optional TorchScript file for native PyTorch flavor registration.
    host, port, tracking_uri:
        MLflow connection settings. Defaults to ``http://localhost:5000``.
    experiment_name:
        Dedicated technical experiment for registry upload runs.

    Returns
    -------
    RegisterResult
        Registration metadata including model version, aliases and source URI.
    """

    source_path = Path(source_dir).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")
    if not source_path.is_dir():
        raise ValueError(f"Source path must be a directory: {source_path}")
    if not name.strip():
        raise ValueError("Registered model name cannot be empty")

    effective_uri = configure_mlflow(host=host, port=port, tracking_uri=tracking_uri)
    client = MlflowClient()
    mlflow.set_experiment(experiment_name)

    general_tags = general_tags or {}
    training_tags = training_tags or {}
    source_hash = hash_directory(source_path)
    created_at = utc_now_iso()

    ensure_registered_model(client, name)
    selected_aliases = list(aliases) if aliases is not None else default_aliases_for_next_version(client, name)

    manifest = build_manifest(
        model_name=name,
        kind=kind,
        source_path=source_path,
        source_dir_hash=source_hash,
        created_at=created_at,
        general_tags=general_tags,
        training_tags=training_tags,
        hf_task=hf_task,
        pytorch_file=pytorch_file,
    )
    version_tags = build_version_tags(
        kind=kind,
        source_hash=source_hash,
        created_at=created_at,
        general_tags=general_tags,
        training_tags=training_tags,
    )

    run_name = f"register:{name}:{kind}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.set_tags(build_run_tags(name=name, kind=kind, source_hash=source_hash))
        mlflow.log_dict(general_tags, "modelctl_metadata/general_tags.json")
        mlflow.log_dict(training_tags, "modelctl_metadata/training_tags.json")
        mlflow.log_dict(manifest, "modelctl_metadata/manifest.json")
        mlflow.log_params({"model_name": name, "kind": kind, "source_dir_hash": source_hash})

        model_info = log_model_by_kind(
            source_path=source_path,
            kind=kind,
            manifest=manifest,
            general_tags=general_tags,
            training_tags=training_tags,
            hf_task=hf_task,
            pytorch_file=pytorch_file,
        )

        source_uri = f"runs:/{run_id}/{DEFAULT_MODEL_ARTIFACT_NAME}"
        model_version = client.create_model_version(
            name=name,
            source=source_uri,
            run_id=run_id,
            tags=version_tags,
            description=description,
        )

    # Be explicit: some MLflow 3.x stores normalize logged model sources to
    # ``models:/m-...`` and not every backend/search path reliably exposes tags
    # that were passed to ``create_model_version`` immediately. Re-setting them
    # through the dedicated API keeps the registry metadata stable.
    for key, value in version_tags.items():
        client.set_model_version_tag(name=name, version=str(model_version.version), key=key, value=value)

    for alias in selected_aliases:
        client.set_registered_model_alias(name=name, alias=alias, version=str(model_version.version))

    model_uri = f"models:/{name}/{model_version.version}"
    return RegisterResult(
        name=name,
        version=str(model_version.version),
        aliases=selected_aliases,
        kind=kind,
        run_id=run_id,
        model_uri=model_uri,
        source_uri=source_uri,
        source_dir_hash=source_hash,
        tracking_uri=effective_uri,
    )


def promote_alias(
    name: str,
    version: str,
    alias: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tracking_uri: str | None = None,
) -> dict[str, str]:
    """Point an alias at an existing model version.

    This is the promotion primitive. For example, promoting version ``12`` to
    ``champion`` means consumers using ``models:/name@champion`` will start
    resolving to version ``12``.
    """

    configure_mlflow(host=host, port=port, tracking_uri=tracking_uri)
    client = MlflowClient()
    client.set_registered_model_alias(name=name, alias=alias, version=str(version))
    return {"name": name, "version": str(version), "alias": alias}


def pull_model(
    ref: str,
    output_dir: str | Path,
    *,
    payload_only: bool = True,
    overwrite: bool = False,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tracking_uri: str | None = None,
) -> PullResult:
    """Download a model version or alias into a local directory.

    Parameters
    ----------
    ref:
        Model reference. Supported forms are ``models:/name@alias``,
        ``models:/name/12``, ``name@alias`` and ``name:12``.
    output_dir:
        Destination directory.
    payload_only:
        If true, generic model packages are unpacked to the original payload
        directory instead of copying the whole MLflow model package. Native
        models are copied as full MLflow model packages.
    overwrite:
        Delete destination directory first if it already exists.
    host, port, tracking_uri:
        MLflow connection settings.

    Returns
    -------
    PullResult
        Information about downloaded and final output paths.
    """

    configure_mlflow(host=host, port=port, tracking_uri=tracking_uri)
    model_uri = normalize_model_ref(ref)
    output_path = Path(output_dir).expanduser().resolve()

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists. Use --overwrite: {output_path}")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    with tempfile.TemporaryDirectory(prefix="modelctl_pull_") as temp_dir:
        downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=temp_dir)).resolve()
        source_to_copy = choose_pull_source(downloaded, payload_only=payload_only)
        copy_path(source_to_copy, output_path)

    return PullResult(
        ref=ref,
        model_uri=model_uri,
        downloaded_path=None,
        output_path=str(output_path),
        payload_only=payload_only,
    )


def list_model_versions(
    name: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tracking_uri: str | None = None,
) -> list[ModelVersionSummary]:
    """Return all versions of a registered model sorted newest first.

    ``MlflowClient.search_model_versions`` is convenient for discovering
    versions, but in some MLflow versions/backends its returned entities can be
    partially populated: aliases and tags may be empty even when they exist in
    the registry. The utility therefore uses search only to get version numbers
    and then fetches every version through ``get_model_version``. Aliases are
    also reconstructed from the registered model's alias map.
    """

    configure_mlflow(host=host, port=port, tracking_uri=tracking_uri)
    client = MlflowClient()
    versions = list(client.search_model_versions(f"name='{name}'"))
    versions.sort(key=lambda item: int(item.version), reverse=True)
    aliases_by_version = collect_aliases_by_version(client, name)

    summaries: list[ModelVersionSummary] = []
    for item in versions:
        full_version = fetch_model_version(client, name, str(item.version))
        summaries.append(summarize_model_version(full_version, aliases_by_version=aliases_by_version))
    return summaries


def get_model_info(
    ref: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Return registry information for a model reference.

    ``ref`` can be ``name@alias`` or ``name:version``. The returned dictionary is
    JSON-serializable and suitable for printing in CLI output.
    """

    configure_mlflow(host=host, port=port, tracking_uri=tracking_uri)
    client = MlflowClient()
    name, version_or_alias, ref_kind = split_registry_ref(ref)
    if ref_kind == "alias":
        mv = client.get_model_version_by_alias(name, version_or_alias)
    else:
        mv = client.get_model_version(name, version_or_alias)
    aliases_by_version = collect_aliases_by_version(client, name)
    summary = summarize_model_version(mv, aliases_by_version=aliases_by_version)
    return asdict(summary) | {"tags": dict(mv.tags or {})}


def log_model_by_kind(
    *,
    source_path: Path,
    kind: ModelKind,
    manifest: dict[str, Any],
    general_tags: dict[str, Any],
    training_tags: dict[str, Any],
    hf_task: str | None,
    pytorch_file: str | Path | None,
) -> Any:
    """Log a model artifact into the active MLflow run according to ``kind``."""

    if kind == "generic":
        return log_generic_model(source_path, manifest, general_tags, training_tags)
    if kind == "hf":
        return log_hf_model(source_path, manifest, hf_task)
    if kind == "pytorch":
        return log_pytorch_model(source_path, manifest, pytorch_file)
    raise ValueError(f"Unsupported kind: {kind}")


def log_generic_model(source_path: Path, manifest: dict[str, Any], general_tags: dict[str, Any], training_tags: dict[str, Any]) -> Any:
    """Log an arbitrary directory as a modelctl generic PyFunc model."""

    with tempfile.TemporaryDirectory(prefix="modelctl_pkg_") as temp_dir:
        package_dir = Path(temp_dir) / "package"
        payload_dir = package_dir / "payload"
        metadata_dir = package_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        copy_path(source_path, payload_dir)
        write_json(package_dir / "manifest.json", manifest)
        write_json(metadata_dir / "general_tags.json", general_tags)
        write_json(metadata_dir / "training_tags.json", training_tags)

        return call_log_model(
            mlflow.pyfunc.log_model,
            name=DEFAULT_MODEL_ARTIFACT_NAME,
            python_model=GenericDirectoryPyFunc(),
            artifacts={"package": str(package_dir)},
            metadata={"modelctl_kind": "generic", "modelctl_schema_version": DEFAULT_SCHEMA_VERSION},
            pip_requirements=["mlflow", "pandas"],
        )


def log_hf_model(source_path: Path, manifest: dict[str, Any], hf_task: str | None) -> Any:
    """Log a Hugging Face Transformers directory using MLflow's native flavor."""

    import mlflow.transformers

    kwargs: dict[str, Any] = {
        "transformers_model": str(source_path),
        "metadata": {"modelctl_kind": "hf", "modelctl_schema_version": DEFAULT_SCHEMA_VERSION},
    }
    if hf_task:
        kwargs["task"] = hf_task
    return call_log_model(mlflow.transformers.log_model, name=DEFAULT_MODEL_ARTIFACT_NAME, **kwargs)


def log_pytorch_model(source_path: Path, manifest: dict[str, Any], pytorch_file: str | Path | None) -> Any:
    """Log a TorchScript model using MLflow's native PyTorch flavor.

    Native MLflow PyTorch logging needs an actual ``torch.nn.Module`` or a
    scripted/traced model. A plain checkpoint folder is not enough because the
    Python class definition is not recoverable from weights alone. This function
    therefore supports TorchScript artifacts, which can be loaded with
    ``torch.jit.load``.
    """

    import torch
    import mlflow.pytorch

    model_file = resolve_pytorch_file(source_path, pytorch_file)
    model = torch.jit.load(str(model_file), map_location="cpu")
    return call_log_model(
        mlflow.pytorch.log_model,
        name=DEFAULT_MODEL_ARTIFACT_NAME,
        pytorch_model=model,
        metadata={"modelctl_kind": "pytorch", "modelctl_schema_version": DEFAULT_SCHEMA_VERSION},
    )


def call_log_model(func: Any, *, name: str, **kwargs: Any) -> Any:
    """Call an MLflow ``log_model`` function using ``name`` or ``artifact_path``.

    Recent MLflow versions prefer ``name`` in examples while older code often
    used ``artifact_path``. This helper makes the utility less sensitive to the
    installed MLflow version.
    """

    signature = inspect.signature(func)
    if "name" in signature.parameters:
        return func(name=name, **kwargs)
    return func(artifact_path=name, **kwargs)


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    """Create a registered model if it does not already exist."""

    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)


def default_aliases_for_next_version(client: MlflowClient, name: str) -> list[str]:
    """Choose aliases when the user did not pass ``--alias``.

    The first version becomes both ``baseline`` and ``champion``. Later versions
    become ``candidate`` to avoid accidentally moving production consumers.
    """

    versions = list(client.search_model_versions(f"name='{name}'"))
    if not versions:
        return ["baseline", "champion"]
    return ["candidate"]


def build_manifest(
    *,
    model_name: str,
    kind: str,
    source_path: Path,
    source_dir_hash: str,
    created_at: str,
    general_tags: dict[str, Any],
    training_tags: dict[str, Any],
    hf_task: str | None,
    pytorch_file: str | Path | None,
) -> dict[str, Any]:
    """Build a stable manifest stored next to every registered payload."""

    return {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "created_by": "modelctl",
        "created_at": created_at,
        "model_name": model_name,
        "kind": kind,
        "source_basename": source_path.name,
        "source_dir_hash": source_dir_hash,
        "payload_path": "payload",
        "general_tags_path": "metadata/general_tags.json",
        "training_tags_path": "metadata/training_tags.json",
        "general_tags": general_tags,
        "training_tags": training_tags,
        "hf_task": hf_task,
        "pytorch_file": str(pytorch_file) if pytorch_file else None,
    }


def build_run_tags(name: str, kind: str, source_hash: str) -> dict[str, str]:
    """Build tags for the technical MLflow run created by modelctl."""

    return {
        "modelctl.managed": "true",
        "modelctl.operation": "register",
        "modelctl.registry_only": "true",
        "modelctl.model_name": name,
        "modelctl.kind": kind,
        "modelctl.source_dir_hash": source_hash,
    }


def build_version_tags(
    *,
    kind: str,
    source_hash: str,
    created_at: str,
    general_tags: dict[str, Any],
    training_tags: dict[str, Any],
) -> dict[str, str]:
    """Build searchable MLflow Model Version tags."""

    tags = {
        "modelctl.managed": "true",
        "modelctl.schema_version": DEFAULT_SCHEMA_VERSION,
        "modelctl.kind": kind,
        "modelctl.source_dir_hash": source_hash,
        "modelctl.created_at": created_at,
    }
    tags.update(flatten_for_mlflow_tags("general", general_tags))
    tags.update(flatten_for_mlflow_tags("training", training_tags))
    return tags


def hash_directory(path: Path) -> str:
    """Compute a stable SHA256 hash for all files in a directory.

    The hash includes relative file paths and file bytes. Directory mtimes,
    owners and permissions are intentionally ignored, which makes the digest more
    stable across machines, NFS mounts and containerized environments.
    """

    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_path = file_path.relative_to(path).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def resolve_pytorch_file(source_path: Path, pytorch_file: str | Path | None) -> Path:
    """Resolve a TorchScript file from explicit CLI input or common names."""

    if pytorch_file is not None:
        candidate = Path(pytorch_file).expanduser()
        if not candidate.is_absolute():
            candidate = source_path / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"TorchScript file does not exist: {candidate}")
        return candidate

    candidates = [
        "model.pt",
        "model.ts",
        "model.torchscript",
        "torchscript.pt",
        "traced_model.pt",
        "scripted_model.pt",
    ]
    for candidate_name in candidates:
        candidate = source_path / candidate_name
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Native kind='pytorch' requires a TorchScript file. Pass --pytorch-file or use kind='generic'."
    )


def normalize_model_ref(ref: str) -> str:
    """Normalize user-friendly model references into MLflow model URIs."""

    if ref.startswith("models:/"):
        return ref
    name, value, ref_kind = split_registry_ref(ref)
    if ref_kind == "alias":
        return f"models:/{name}@{value}"
    return f"models:/{name}/{value}"


def split_registry_ref(ref: str) -> tuple[str, str, Literal["alias", "version"]]:
    """Split ``name@alias`` or ``name:version`` into components."""

    if ref.startswith("models:/"):
        stripped = ref.removeprefix("models:/")
        if "@" in stripped:
            name, alias = stripped.split("@", 1)
            return name, alias.split("/", 1)[0], "alias"
        parts = stripped.split("/", 1)
        if len(parts) == 2:
            return parts[0], parts[1].split("/", 1)[0], "version"
        raise ValueError(f"Unsupported model URI: {ref}")

    if "@" in ref:
        name, alias = ref.split("@", 1)
        return name, alias, "alias"
    if ":" in ref:
        name, version = ref.rsplit(":", 1)
        return name, version, "version"
    raise ValueError("Model ref must be name@alias, name:version or models:/... URI")


def choose_pull_source(downloaded: Path, *, payload_only: bool) -> Path:
    """Choose what should be copied from a downloaded MLflow model package."""

    if not payload_only:
        return downloaded

    generic_payload = downloaded / "artifacts" / "package" / "payload"
    if generic_payload.exists():
        return generic_payload

    # Be tolerant to MLflow layout changes and look for the modelctl package.
    for manifest_path in downloaded.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("created_by") == "modelctl" and manifest.get("kind") == "generic":
            payload_path = manifest_path.parent / "payload"
            if payload_path.exists():
                return payload_path

    return downloaded


def copy_path(source: Path, destination: Path) -> None:
    """Copy a file or directory to ``destination``."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dictionary as pretty UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_model_version(client: MlflowClient, name: str, version: str) -> Any:
    """Fetch one fully populated model version from the registry.

    The search endpoint is used only for discovery. This function is used before
    printing details because it returns a more complete entity on MLflow servers
    where search results omit tags or aliases.
    """

    return client.get_model_version(name=name, version=str(version))


def collect_aliases_by_version(client: MlflowClient, name: str) -> dict[str, list[str]]:
    """Return a reverse mapping ``version -> [aliases]`` for a model.

    MLflow stores aliases on the registered model as a mapping ``alias ->
    version`` and also exposes aliases on individual model versions. Reading the
    registered model gives a reliable, cheap source for ``modelctl list``.
    """

    try:
        registered_model = client.get_registered_model(name)
    except MlflowException:
        return {}

    alias_map = getattr(registered_model, "aliases", {}) or {}
    aliases_by_version: dict[str, list[str]] = {}
    for alias, version in dict(alias_map).items():
        aliases_by_version.setdefault(str(version), []).append(str(alias))

    for aliases in aliases_by_version.values():
        aliases.sort()
    return aliases_by_version


def summarize_model_version(mv: Any, *, aliases_by_version: dict[str, list[str]] | None = None) -> ModelVersionSummary:
    """Convert an MLflow ModelVersion entity into a small summary."""

    tags = dict(mv.tags or {})
    version = str(mv.version)
    version_aliases = list(getattr(mv, "aliases", []) or [])
    if aliases_by_version is not None:
        version_aliases = aliases_by_version.get(version, version_aliases)

    return ModelVersionSummary(
        name=str(mv.name),
        version=version,
        aliases=version_aliases,
        status=str(getattr(mv, "status", "")) or None,
        run_id=getattr(mv, "run_id", None),
        source=getattr(mv, "source", None),
        kind=tags.get("modelctl.kind"),
        source_dir_hash=tags.get("modelctl.source_dir_hash"),
        created_at=tags.get("modelctl.created_at") or timestamp_ms_to_iso(getattr(mv, "creation_timestamp", None)),
    )


def timestamp_ms_to_iso(timestamp_ms: int | None) -> str | None:
    """Convert an MLflow millisecond timestamp to ISO-8601 UTC text."""

    if timestamp_ms is None:
        return None
    try:
        return datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
