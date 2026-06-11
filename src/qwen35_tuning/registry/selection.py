from __future__ import annotations


def better_metric(current: float | None, candidate: float, greater_is_better: bool) -> bool:
    if current is None:
        return True
    return candidate > current if greater_is_better else candidate < current

