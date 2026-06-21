from __future__ import annotations

import argparse

from config import load_config
from preprocessing.pipeline import prepare_pretokenized_splits
from tracking.model_source import resolve_model_source
from utils.logging import configure_logging, get_logger
from utils.seed import set_seed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> None:
    """Prepare local model and preprocessing artifacts before distributed training."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild pretokenized split caches even when raw hashes and preprocessing signatures match",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        help="override preprocessing.workers.num_workers for this prepare run",
    )
    parser.add_argument(
        "--worker-chunk-size",
        type=positive_int,
        default=None,
        help="override preprocessing.workers.chunk_size for this prepare run",
    )
    args = parser.parse_args()

    configure_logging(is_main_process=True)
    logger = get_logger("prepare")
    logger.info("loading config: %s", args.config)
    config = load_config(args.config)
    logger.info("config loaded: project=%s run_name=%s", config.project.name, config.project.run_name)
    set_seed(config.project.seed)

    model_source = resolve_model_source(config, tracking_uri=config.mlflow.tracking_uri)
    logger.info(
        "registry model resolved: effective_model_id=%s ref=%s pulled=%s used_local=%s",
        model_source.effective_model_id,
        model_source.ref,
        model_source.pulled,
        model_source.used_local,
    )

    logger.info(
        "building training data cache force=%s workers=%s chunk_size=%s",
        args.force,
        args.workers if args.workers is not None else config.preprocessing.workers.num_workers,
        args.worker_chunk_size if args.worker_chunk_size is not None else config.preprocessing.workers.chunk_size,
    )
    results = prepare_pretokenized_splits(
        config,
        ["train", "valid", "test"],
        model_source=model_source,
        force_refresh=args.force,
        num_workers=args.workers,
        worker_chunk_size=args.worker_chunk_size,
    )

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

    logger.info("prepare complete")


if __name__ == "__main__":
    main()
