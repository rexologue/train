from __future__ import annotations

import argparse
import json

from config import load_config
from data.dataloaders import build_dataloaders
from data.inspection import inspect_random_batch
from preprocessing.pipeline import prepare_pretokenized_splits
from tracking import ExperimentTracker
from utils.logging import get_logger


def main() -> None:
    """Top-level training orchestrator.

    The project is still at the startup preprocessing stage: this entrypoint
    reads YAML config, prepares/reuses pretokenized splits, and then stops
    before the future model-training loop.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.preprocess.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"], choices=["train", "valid", "test"])
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--inspect-random-batch", action="store_true")
    parser.add_argument("--inspect-split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--inspect-token-limit", type=int, default=32)

    args = parser.parse_args()

    logger = get_logger("train")
    logger.info("loading config: %s", args.config)
    config = load_config(args.config)
    logger.info("config loaded: project=%s run_name=%s", config.section("project")["name"], config.section("project").get("run_name"))

    with ExperimentTracker.from_config(config) as tracker:
        model_source = tracker.resolve_model_source()
        logger.info(
            "model source resolved: kind=%s effective_model_id=%s ref=%s pulled=%s used_local=%s",
            model_source.kind,
            model_source.effective_model_id,
            model_source.ref,
            model_source.pulled,
            model_source.used_local,
        )
        tracker.log_run_start(config_path=args.config)
        tracker.log_lineage()

        logger.info("preparing pretokenized splits: %s", ",".join(args.splits))
        results = prepare_pretokenized_splits(config, args.splits, force=args.force_preprocess)
        for result in results:
            logger.info(
                "split ready: split=%s reused=%s rows=%s rejected=%s pretok=%s manifest=%s",
                result.split,
                result.reused,
                result.manifest.get("num_rows"),
                result.manifest.get("num_rejected_rows"),
                result.pretok_path,
                result.manifest_path,
            )
        tracker.log_preprocessing_results(results)

        logger.info("building routed dataloaders")
        dataloaders = build_dataloaders(config, results)
        for split, split_loader in dataloaders.splits.items():
            summary = split_loader.summary
            logger.info(
                "dataloader ready: split=%s rows=%s batches=%s short_batches=%s loss_kinds=%s path=%s",
                split,
                summary["num_rows"],
                summary["num_batches"],
                summary["num_short_batches"],
                summary["loss_kind_counts"],
                summary["path"],
            )
        tracker.log_dataloaders(dataloaders)

        if args.inspect_random_batch:
            report = inspect_random_batch(
                dataloaders,
                split=args.inspect_split,
                seed=int(config.section("project").get("seed", 0)),
                token_limit=args.inspect_token_limit,
            )
            logger.info("random dataloader batch inspection:\n%s", json.dumps(report, ensure_ascii=False, indent=2))

    logger.info("startup preprocessing and dataloader build complete; model training is the next pipeline stage")


if __name__ == "__main__":
    main()
