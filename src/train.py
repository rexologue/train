from __future__ import annotations

import argparse

from config import load_config
from preprocessing.pipeline import prepare_pretokenized_splits
from utils.logging import get_logger


def main() -> None:
    """Top-level training orchestrator.

    The project is still at the startup preprocessing stage: this entrypoint
    reads YAML config, prepares/reuses pretokenized splits, and then stops
    before the future model-training loop.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.preprocess.yaml")
    parser.add_argument("--splits", nargs="+", default=["valid"], choices=["train", "valid", "test"])
    parser.add_argument("--force-preprocess", action="store_true")

    args = parser.parse_args()

    logger = get_logger("train")
    logger.info("loading config: %s", args.config)
    config = load_config(args.config)
    logger.info("config loaded: project=%s run_name=%s", config.section("project")["name"], config.section("project").get("run_name"))

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

    logger.info("startup preprocessing complete; model training is the next pipeline stage")


if __name__ == "__main__":
    main()
