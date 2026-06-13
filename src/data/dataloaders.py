from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import TrainingConfig
from data.collators import RoutedCollator
from data.pretokenized_dataset import PretokenizedDataset
from data.routed_batch_sampler import RoutedBatchSampler
from preprocessing.io import PretokSplitResult


REQUIRED_TRAINING_SPLITS = {"train", "valid"}


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
    """Container for all split DataLoader instances built at startup."""

    splits: dict[str, SplitDataLoader]

    def __getitem__(self, split: str) -> SplitDataLoader:
        return self.splits[split]


def resolve_pad_token_id(config: TrainingConfig) -> int:
    """Load tokenizer metadata and return the configured padding token id.

    Preprocessing already loads the tokenizer for rendering/tokenization. The
    startup DataLoader stage currently reloads tokenizer metadata only to avoid
    hardcoding padding behavior before a shared training context exists.
    """

    from transformers import AutoTokenizer

    tokenizer_config = config.section("tokenizer")
    model_config = config.section("model")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["tokenizer_id"],
        revision=tokenizer_config.get("tokenizer_revision"),
        use_fast=bool(tokenizer_config.get("use_fast", True)),
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    raise ValueError("tokenizer must define pad_token_id or eos_token_id for batch padding")


def build_dataloaders(
    config: TrainingConfig,
    pretok_results: list[PretokSplitResult],
    *,
    pad_token_id: int | None = None,
) -> DataLoaderBundle:
    """Build homogeneous routed DataLoader instances from pretokenized splits.

    `train` and `valid` are required because the next stage is the ordinary
    SFT/DPO training contour. `test` is included only when preprocessing
    produced it. Each split uses the same collator but has an independent
    route sampler over that split's `loss_kind` column.
    """

    from torch.utils.data import DataLoader

    results_by_split = {result.split: result for result in pretok_results}
    missing = sorted(REQUIRED_TRAINING_SPLITS - set(results_by_split))
    if missing:
        raise ValueError(f"pretokenized train and valid splits are required; missing={missing}")

    training = config.section("training")
    batch_size = int(training.get("per_device_train_batch_size", 0))
    if batch_size <= 0:
        raise ValueError("training.per_device_train_batch_size must be a positive integer")
    drop_last = bool(training.get("drop_last", False))
    seed = int(config.section("project").get("seed", 0))
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
        sampler = RoutedBatchSampler(dataset.loss_kinds, batch_size, seed=seed, drop_last=drop_last, shuffle=True)
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
