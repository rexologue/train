from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import TrainingConfig, effective_tokenizer_id
from data.collators import RoutedCollator
from data.pretokenized_dataset import PretokenizedDataset
from data.routed_batch_sampler import RoutedBatchSampler
from preprocessing.io import PretokSplitResult


REQUIRED_TRAINING_SPLITS = {"train", "valid"}
SFT_LOSS_KINDS = {"sft_target", "sft_tool"}


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
        effective_tokenizer_id(config),
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
        if split in REQUIRED_TRAINING_SPLITS and len(dataset) == 0:
            raise ValueError(f"pretokenized {split} split is empty: {result.pretok_path}")
        sampler = RoutedBatchSampler(dataset.loss_kinds, batch_size, seed=seed, drop_last=drop_last, shuffle=True)
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


def validate_sft_only_training_inputs(config: TrainingConfig, dataloaders: DataLoaderBundle) -> None:
    """Reject routes and sampler shapes that the current SFT-only trainer cannot run safely."""

    configured_routes = set(config.section("loss_routing").get("routes") or {})
    unsupported_routes = configured_routes - SFT_LOSS_KINDS
    if unsupported_routes:
        raise ValueError(f"SFT-only training does not support configured routes: {sorted(unsupported_routes)}")

    for split in sorted(REQUIRED_TRAINING_SPLITS):
        split_loader = dataloaders[split]
        unsupported_data = set(split_loader.dataset.loss_kinds) - SFT_LOSS_KINDS
        if unsupported_data:
            raise ValueError(f"SFT-only training does not support {split} loss kinds: {sorted(unsupported_data)}")
        if split_loader.summary["num_short_batches"] and split_loader.summary["batch_size"] > 1:
            raise ValueError(
                f"{split} contains short routed batches with per-device batch size > 1; "
                "Accelerate even-batch padding can mix loss routes. Set training.drop_last=true or use batch size 1."
            )
