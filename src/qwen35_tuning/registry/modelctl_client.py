from __future__ import annotations

from qwen35_tuning.registry.tags import validate_training_aliases


def build_modelctl_register_args(modelctl_path: str, model_name: str, artifact_path: str, aliases: list[str]) -> list[str]:
    validate_training_aliases(aliases)
    args = [modelctl_path, "register", "--model", model_name, "--path", artifact_path]
    for alias in aliases:
        args.extend(["--alias", alias])
    return args

