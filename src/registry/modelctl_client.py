from __future__ import annotations

from registry.tags import validate_training_aliases


def build_modelctl_register_args(
    model_name: str,
    artifact_path: str,
    aliases: list[str],
    *,
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
    args = ["modelctl", "register", artifact_path, model_name, "--kind", "generic"]
    for alias in aliases:
        args.extend(["--alias", alias])
    if tracking_uri:
        args.extend(["--tracking-uri", tracking_uri])
    if general_tags_json:
        args.extend(["--general-tags-json", general_tags_json])
    if training_tags_json:
        args.extend(["--training-tags-json", training_tags_json])
    return args
