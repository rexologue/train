from __future__ import annotations

from collections import defaultdict
import random
from typing import Iterator


class RoutedBatchSampler:
    """Batch sampler that guarantees homogeneous `loss_kind` batches.

    The sampler first preserves row order inside each route, chunks each route
    into fixed-size batches, and only then shuffles the list of completed
    batches. That keeps route frequencies equal to the underlying parquet data
    while preventing any SFT/DPO mix inside a single batch.
    """

    def __init__(
        self,
        loss_kinds: list[str],
        batch_size: int,
        *,
        seed: int = 0,
        drop_last: bool = False,
        shuffle: bool = True,
        replica_group_size: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if replica_group_size <= 0:
            raise ValueError("replica_group_size must be positive")

        self.loss_kinds = list(loss_kinds)
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.replica_group_size = int(replica_group_size)
        self.epoch = 0
        self.num_padded_replica_batches = 0

        groups: dict[str, list[int]] = defaultdict(list)
        for index, loss_kind in enumerate(self.loss_kinds):
            groups[loss_kind].append(index)

        self.groups = dict(groups)
        self.batches = self._build_batches()


    def _build_batches(self) -> list[list[int]]:
        """Create route-local batches before deterministic cross-route shuffle."""

        replica_groups: list[list[list[int]]] = []
        self.num_padded_replica_batches = 0
        rng = random.Random(self.seed + self.epoch)
        for indices in self.groups.values():
            route_indices = list(indices)
            if self.shuffle:
                rng.shuffle(route_indices)
            route_batches: list[list[int]] = []
            for start in range(0, len(route_indices), self.batch_size):
                batch = route_indices[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                route_batches.append(batch)
            replica_groups.extend(self._replica_groups(route_batches))

        if self.shuffle:
            rng.shuffle(replica_groups)

        batches: list[list[int]] = []
        for group in replica_groups:
            batches.extend(group)
        return batches

    def _replica_groups(self, route_batches: list[list[int]]) -> list[list[list[int]]]:
        """Group route-local batches so Accelerate batch sharding keeps ranks aligned."""

        groups: list[list[list[int]]] = []
        width = self.replica_group_size
        for start in range(0, len(route_batches), width):
            group = [list(batch) for batch in route_batches[start : start + width]]
            if len(group) < width:
                if self.drop_last:
                    continue
                if not group:
                    continue
                while len(group) < width:
                    group.append(list(group[-1]))
                    self.num_padded_replica_batches += 1
            groups.append(group)
        return groups


    def set_epoch(self, epoch: int) -> None:
        """Reshuffle route-local and cross-route batch order for a new epoch."""

        self.epoch = int(epoch)
        self.batches = self._build_batches()


    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.batches:
            yield list(batch)


    def __len__(self) -> int:
        return len(self.batches)


    def summary(self) -> dict[str, object]:
        """Return lightweight routing stats for logging and audits."""

        rows_by_loss_kind = {loss_kind: len(indices) for loss_kind, indices in sorted(self.groups.items())}
        batches_by_loss_kind: dict[str, int] = {loss_kind: 0 for loss_kind in rows_by_loss_kind}
        short_batches = 0

        for batch in self.batches:
            if not batch:
                continue

            loss_kind = self.loss_kinds[batch[0]]
            batches_by_loss_kind[loss_kind] = batches_by_loss_kind.get(loss_kind, 0) + 1

            if len(batch) < self.batch_size:
                short_batches += 1

        return {
            "rows_by_loss_kind": rows_by_loss_kind,
            "batches_by_loss_kind": batches_by_loss_kind,
            "num_batches": len(self.batches),
            "num_short_batches": short_batches,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "shuffle": self.shuffle,
            "replica_group_size": self.replica_group_size,
            "num_padded_replica_batches": self.num_padded_replica_batches,
            "seed": self.seed,
            "epoch": self.epoch,
        }
