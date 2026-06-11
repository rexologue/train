from __future__ import annotations


def accuracy(results: list[bool]) -> float:
    return sum(results) / len(results) if results else 0.0

