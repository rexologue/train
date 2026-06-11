from __future__ import annotations


FORBIDDEN_TRAINING_ALIASES = {"baseline", "champion"}


def candidate_alias(template: str, candidate_index: int) -> str:
    alias = template.format(candidate_index=candidate_index)
    if alias in FORBIDDEN_TRAINING_ALIASES:
        raise ValueError(f"training must not register alias {alias!r}")
    if not alias.startswith("candidate-"):
        raise ValueError("training candidate alias must start with candidate-")
    return alias


def validate_training_aliases(aliases: list[str]) -> None:
    forbidden = sorted(FORBIDDEN_TRAINING_ALIASES & set(aliases))
    if forbidden:
        raise ValueError(f"training registration cannot assign production aliases: {forbidden}")

