from __future__ import annotations

from collections import defaultdict
from typing import Iterator


class RoutedBatchSampler:
    def __init__(self, loss_kinds: list[str], batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        groups: dict[str, list[int]] = defaultdict(list)
        for index, loss_kind in enumerate(loss_kinds):
            groups[loss_kind].append(index)
        self.groups = dict(groups)

    def __iter__(self) -> Iterator[list[int]]:
        for loss_kind in sorted(self.groups):
            indices = self.groups[loss_kind]
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.groups.values())

