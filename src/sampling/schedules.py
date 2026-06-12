from __future__ import annotations


def normalize_sampler_weights(weights: dict[str, float] | None, observed_counts: dict[str, int]) -> dict[str, float]:
    if weights is None:
        total = sum(observed_counts.values())
        return {key: value / total for key, value in observed_counts.items()} if total else {}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("sampler weights must sum to a positive value")
    return {key: value / total_weight for key, value in weights.items()}

