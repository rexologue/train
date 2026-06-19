from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

from registry.tags import validate_training_aliases


@dataclass(frozen=True)
class ModelctlCommandFailure(RuntimeError):
    """Raised when a modelctl command exits unsuccessfully."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def __str__(self) -> str:
        command_text = " ".join(self.command)
        stderr = self.stderr.strip()
        stdout = self.stdout.strip()
        details = stderr or stdout or "no output"
        return f"modelctl command failed rc={self.returncode}: {command_text}: {details}"


@dataclass(frozen=True)
class ModelctlInfo:
    """Parsed output of `modelctl info`."""

    ref: str
    name: str | None
    version: str | None
    aliases: list[str]
    payload_hash: str | None
    tags: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class ModelctlVerifyResult:
    """Parsed output of `modelctl verify`."""

    ref: str
    model_uri: str | None
    path: Path
    expected_payload_hash: str | None
    actual_payload_hash: str | None
    matches: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class ModelctlPullResult:
    """Parsed output of `modelctl pull`."""

    ref: str
    model_uri: str | None
    output_path: Path
    full_package: bool
    payload_hash: str | None
    verified: bool
    replaced_existing: bool
    tracking_uri: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class ModelctlRegisterRequest:
    """A candidate registration request for modelctl."""

    model_name: str
    source_dir: Path
    aliases: tuple[str, ...]
    general_tags_json: Path | None = None
    training_tags_json: Path | None = None
    description: str | None = None


@dataclass(frozen=True)
class ModelctlRegisterResult:
    """Parsed output of `modelctl register`."""

    name: str
    version: str | None
    aliases: list[str]
    run_id: str | None
    model_uri: str | None
    source_uri: str | None
    payload_hash: str | None
    tracking_uri: str | None
    raw: dict[str, Any]


class ModelctlClient:
    """Subprocess-only facade over the external modelctl CLI."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        executable: str = "modelctl",
        timeout_seconds: float = 300,
    ):
        self.tracking_uri = tracking_uri
        self.executable = executable
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


    def info(self, ref: str) -> ModelctlInfo:
        """Return registry metadata for one model reference."""

        payload = self._run_json(["info", ref])
        tags = payload.get("tags")
        aliases = payload.get("aliases")
        return ModelctlInfo(
            ref=ref,
            name=string_or_none(payload.get("name")),
            version=string_or_none(payload.get("version")),
            aliases=[str(item) for item in aliases] if isinstance(aliases, list) else [],
            payload_hash=extract_payload_hash(payload),
            tags=tags if isinstance(tags, dict) else {},
            raw=payload,
        )


    def verify(self, ref: str, path: str | Path) -> ModelctlVerifyResult:
        """Compare a local model directory with a registry ref."""

        payload = self._run_json(["verify", ref, str(path)], allowed_returncodes={0, 2})
        return ModelctlVerifyResult(
            ref=ref,
            model_uri=string_or_none(payload.get("model_uri")),
            path=Path(str(payload.get("path") or path)),
            expected_payload_hash=string_or_none(payload.get("expected_payload_hash")),
            actual_payload_hash=string_or_none(payload.get("actual_payload_hash")),
            matches=bool(payload.get("matches")),
            raw=payload,
        )


    def pull(self, ref: str, output_dir: str | Path, *, overwrite: bool = False) -> ModelctlPullResult:
        """Download a model payload directory with modelctl's built-in verification."""

        args = ["pull", ref, str(output_dir)]
        if overwrite:
            args.append("--overwrite")
        payload = self._run_json(args)
        return ModelctlPullResult(
            ref=ref,
            model_uri=string_or_none(payload.get("model_uri")),
            output_path=Path(str(payload.get("output_path") or output_dir)),
            full_package=bool(payload.get("full_package")),
            payload_hash=string_or_none(payload.get("payload_hash")),
            verified=bool(payload.get("verified")),
            replaced_existing=bool(payload.get("replaced_existing")),
            tracking_uri=self.tracking_uri,
            raw=payload,
        )


    def register(self, request: ModelctlRegisterRequest) -> ModelctlRegisterResult:
        """Register one model payload directory as an explicit candidate."""

        aliases = list(request.aliases)
        validate_training_aliases(aliases)
        args = ["register", str(request.source_dir), request.model_name]
        for alias in aliases:
            args.extend(["--alias", alias])
        if request.general_tags_json is not None:
            args.extend(["--general-tags-json", str(request.general_tags_json)])
        if request.training_tags_json is not None:
            args.extend(["--training-tags-json", str(request.training_tags_json)])
        if request.description is not None:
            args.extend(["--description", request.description])
        payload = self._run_json(args)
        result_aliases = payload.get("aliases")
        return ModelctlRegisterResult(
            name=str(payload.get("name") or request.model_name),
            version=string_or_none(payload.get("version")),
            aliases=[str(item) for item in result_aliases] if isinstance(result_aliases, list) else aliases,
            run_id=string_or_none(payload.get("run_id")),
            model_uri=string_or_none(payload.get("model_uri")),
            source_uri=string_or_none(payload.get("source_uri")),
            payload_hash=string_or_none(payload.get("payload_hash")),
            tracking_uri=string_or_none(payload.get("tracking_uri")),
            raw=payload,
        )


    def _run_json(
        self,
        command_args: list[str],
        *,
        allowed_returncodes: set[int] | None = None,
    ) -> dict[str, Any]:
        """Run one fixed modelctl command and parse its JSON stdout."""

        allowed = allowed_returncodes or {0}
        command = self._command(command_args)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if completed.returncode not in allowed:
            raise ModelctlCommandFailure(
                command=tuple(command),
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        stdout = completed.stdout.strip()
        if not stdout:
            return {}
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"modelctl returned non-JSON stdout for {' '.join(command)}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"modelctl returned non-object JSON for {' '.join(command)}")
        return payload


    def _command(self, command_args: list[str]) -> list[str]:
        """Build a complete modelctl command line."""

        args = [self.executable, *command_args]
        if self.tracking_uri:
            args.extend(["--tracking-uri", self.tracking_uri])
        return args


def extract_payload_hash(payload: dict[str, Any]) -> str | None:
    """Extract the modelctl payload hash from info-like JSON."""

    direct = string_or_none(payload.get("payload_hash"))
    if direct:
        return direct
    tags = payload.get("tags")
    if isinstance(tags, dict):
        return string_or_none(tags.get("modelctl.payload_hash"))
    return None


def string_or_none(value: Any) -> str | None:
    """Convert empty values to None and non-empty values to strings."""

    if value in (None, ""):
        return None
    return str(value)
