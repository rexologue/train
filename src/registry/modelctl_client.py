from __future__ import annotations

from registry.tags import validate_training_aliases


def build_modelctl_register_args(
    modelctl_path: str,
    model_name: str,
    artifact_path: str,
    aliases: list[str],
    *,
    kind: str = "generic",
    tracking_uri: str | None = None,
    training_tags_json: str | None = None,
    general_tags_json: str | None = None,
) -> list[str]:
    """Build args for the vendored modelctl register CLI.

    The real modelctl contract is positional: `register SOURCE_DIR NAME`.
    Training code must always pass explicit candidate aliases, because modelctl
    defaults the first unaliased version to baseline/champion.
    """

    validate_training_aliases(aliases)
    args = [modelctl_path, "register", artifact_path, model_name, "--kind", kind]
    for alias in aliases:
        args.extend(["--alias", alias])
    if tracking_uri:
        args.extend(["--tracking-uri", tracking_uri])
    if general_tags_json:
        args.extend(["--general-tags-json", general_tags_json])
    if training_tags_json:
        args.extend(["--training-tags-json", training_tags_json])
    return args


def build_modelctl_pull_args(modelctl_path: str, ref: str, output_dir: str, *, tracking_uri: str | None = None) -> list[str]:
    """Build args for `modelctl pull REF OUTPUT_DIR` without overwrite by default."""

    args = [modelctl_path, "pull", ref, output_dir]
    if tracking_uri:
        args.extend(["--tracking-uri", tracking_uri])
    return args


def build_modelctl_info_args(modelctl_path: str, ref: str, *, tracking_uri: str | None = None) -> list[str]:
    """Build args for `modelctl info REF`."""

    args = [modelctl_path, "info", ref]
    if tracking_uri:
        args.extend(["--tracking-uri", tracking_uri])
    return args
