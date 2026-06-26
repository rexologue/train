from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import Config
from data.collators import RoutedCollator
from data.pretokenized_dataset import PretokenizedDataset
from data.routed_batch_sampler import RoutedBatchSampler
from preprocessing.io import PretokSplitResult


REQUIRED_TRAINING_SPLITS = {"train", "valid"}
SFT_LOSS_KINDS = {"sft_target", "sft_tool"}
DPO_LOSS_KIND = "dpo_target"


@dataclass(frozen=True)
class SplitDataLoader:
    """DataLoader plus the dataset/sampler metadata needed for audits."""

    split: str
    path: Path
    dataset: PretokenizedDataset
    sampler: RoutedBatchSampler
    dataloader: Any
    summary: dict[str, Any]


@dataclass(frozen=True)
class DataLoaderBundle:
    """Container for all split DataLoader instances used by the run."""

    splits: dict[str, SplitDataLoader]

    def __getitem__(self, split: str) -> SplitDataLoader:
        return self.splits[split]


def resolve_pad_token_id(config: Config) -> int:
    """Load tokenizer metadata and return the configured padding token id.

    The tokenizer metadata is reloaded here to avoid hardcoding padding
    behavior before the training objects are constructed.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model.cache_dir),
        use_fast=config.tokenizer.use_fast,
        trust_remote_code=config.model.trust_remote_code,
    )

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    
    raise ValueError("tokenizer must define pad_token_id or eos_token_id for batch padding")


def build_dataloaders(
    config: Config,
    pretok_results: list[PretokSplitResult],
    *,
    pad_token_id: int | None = None,
    num_processes: int = 1,
) -> DataLoaderBundle:
    """Build homogeneous routed DataLoader instances from pretokenized splits.

    `train` and `valid` are required because the next stage is the ordinary
    SFT/DPO training contour. `test` is included only when preprocessing
    produced it. Each split uses the same collator but has an independent
    route sampler over that split's `loss_kind` column.
    """

    results_by_split = {result.split: result for result in pretok_results}
    missing = sorted(REQUIRED_TRAINING_SPLITS - set(results_by_split))
    if missing:
        raise ValueError(f"pretokenized train and valid splits are required; missing={missing}")

    batch_size = config.training.per_device_train_batch_size
    if batch_size <= 0:
        raise ValueError("training.per_device_train_batch_size must be a positive integer")
    replica_group_size = max(int(num_processes), 1)
    drop_last = config.training.drop_last
    seed = config.project.seed
    collator = RoutedCollator(
        pad_token_id=resolve_pad_token_id(config) if pad_token_id is None else int(pad_token_id),
        ignore_index=config.ignore_index,
    )

    split_loaders: dict[str, SplitDataLoader] = {}
    for split in ("train", "valid", "test"):
        result = results_by_split.get(split)
        if result is None:
            continue

        # The Dataset owns parquet normalization; sampler and collator operate
        # only on the normalized row contract.
        dataset = PretokenizedDataset.from_parquet(result.pretok_path, split=split)
        if split in REQUIRED_TRAINING_SPLITS and len(dataset) == 0:
            raise ValueError(f"pretokenized {split} split is empty: {result.pretok_path}")
        
        sampler = RoutedBatchSampler(
            dataset.loss_kinds,
            batch_size,
            seed=seed,
            drop_last=drop_last,
            shuffle=True,
            replica_group_size=replica_group_size,
            gradient_accumulation_steps=(
                config.training.gradient_accumulation_steps if split == "train" else None
            ),
            drop_incomplete_accumulation=(split == "train"),
        )
        if split in REQUIRED_TRAINING_SPLITS and len(sampler) == 0:
            raise ValueError(
                f"{split} routed sampler produced zero batches; check training.drop_last and per-device batch size"
            )
        
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)
        summary = sampler.summary()
        summary.update(
            {
                "split": split,
                "path": str(result.pretok_path),
                "num_rows": len(dataset),
                "loss_kind_counts": dict(sorted(dataset.loss_kind_counts.items())),
            }
        )
        split_loaders[split] = SplitDataLoader(split, result.pretok_path, dataset, sampler, dataloader, summary)

    return DataLoaderBundle(split_loaders)


def validate_training_inputs(config: Config, dataloaders: DataLoaderBundle) -> None:
    """Validate routed training inputs against configured loss routes."""

    configured_routes = set(config.loss_routing.routes)
    for route_name, route in config.loss_routing.routes.items():
        if route_name in SFT_LOSS_KINDS and route.type != "sft_ce":
            raise ValueError(f"{route_name} must use loss type sft_ce")
        if route_name == DPO_LOSS_KIND and route.type != "dpo":
            raise ValueError("dpo_target must use loss type dpo")

    for split in sorted(REQUIRED_TRAINING_SPLITS):
        split_loader = dataloaders[split]
        unsupported_data = set(split_loader.dataset.loss_kinds) - configured_routes
        if unsupported_data:
            raise ValueError(f"{split} contains loss kinds missing from config: {sorted(unsupported_data)}")
        if split_loader.summary["num_short_batches"] and split_loader.summary["batch_size"] > 1:
            raise ValueError(
                f"{split} contains short routed batches with per-device batch size > 1; "
                "Accelerate even-batch padding can mix loss routes. Set training.drop_last=true or use batch size 1."
            )
