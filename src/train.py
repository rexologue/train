from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import nullcontext
from numbers import Number
from pathlib import Path

from config import load_config
from checkpointing import (
    adapter_dir,
    build_resume_hashes,
    list_checkpoints,
    load_checkpoint_manifest,
    load_training_state_without_model,
    load_trainer_state,
    prune_old_checkpoints,
    resolve_resume_checkpoint,
    save_adapter_checkpoint,
    validate_resume_checkpoint,
)
from data.dataloaders import build_dataloaders, validate_sft_only_training_inputs
from data.inspection import inspect_random_batch
from eval.bfcl import run_bfcl_eval
from eval.ordinary import run_standard_eval
from preprocessing.io import load_pretokenized_split_results
from preprocessing.pipeline import prepare_pretokenized_splits
from registry.package import build_candidate_registration_args
from registry.selection import CandidateWindowSelector, RegistrationDecision
from tracking import ExperimentTracker
from trainer.callbacks import TrainerHooks
from trainer.distributed import create_accelerator, prepare_with_accelerator
from trainer.modeling import build_training_objects, load_tokenizer
from trainer.progress import TrainingProgress
from trainer.state import TrainerState
from trainer.trainer import RoutedTrainer
from utils.logging import configure_logging, get_logger
from utils.seed import set_seed


def main() -> None:
    """Top-level training orchestrator."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.preprocess.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"], choices=["train", "valid", "test"])
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--train", action="store_true", help="Run training even when training.enabled=false.")
    parser.add_argument("--inspect-random-batch", action="store_true")
    parser.add_argument("--inspect-split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--inspect-token-limit", type=int, default=32)

    args = parser.parse_args()

    logger = get_logger("train")
    logger.info("loading config: %s", args.config)
    config = load_config(args.config)
    logger.info("config loaded: project=%s run_name=%s", config.section("project")["name"], config.section("project").get("run_name"))
    set_seed(int(config.section("project").get("seed", 0)))

    training_enabled = bool(config.section("training").get("enabled", True))
    should_train = args.train or (training_enabled and not args.preprocess_only)
    runtime = create_accelerator(config) if should_train else None
    accelerator = runtime.accelerator if runtime is not None else None
    is_main_process = bool(getattr(accelerator, "is_main_process", True)) if accelerator is not None else True
    configure_logging(is_main_process=is_main_process)

    tracker = ExperimentTracker.from_config(config)
    if not is_main_process:
        tracker.enabled = False
    with tracker:
        if is_main_process:
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
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if not is_main_process:
            configure_model_source_without_side_effects(config)

        if is_main_process:
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
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if not is_main_process:
            results = load_pretokenized_split_results(config, args.splits)

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
        if is_main_process:
            tracker.log_dataloaders(dataloaders)

        if args.inspect_random_batch and is_main_process:
            report = inspect_random_batch(
                dataloaders,
                split=args.inspect_split,
                seed=int(config.section("project").get("seed", 0)),
                token_limit=args.inspect_token_limit,
            )
            logger.info("random dataloader batch inspection:\n%s", json.dumps(report, ensure_ascii=False, indent=2))

        if not should_train:
            logger.info("startup preprocessing and dataloader build complete; training skipped")
            return

        validate_sft_only_training_inputs(config, dataloaders)
        run_training(config, dataloaders, tracker, runtime=runtime)

    logger.info("training pipeline complete")


def run_training(config, dataloaders, tracker: ExperimentTracker, *, runtime) -> TrainerState:
    logger = get_logger("train")
    accelerator = runtime.accelerator

    resume_checkpoint = resolve_resume_checkpoint(config)
    if resume_checkpoint is not None:
        logger.info("resuming from checkpoint: %s", resume_checkpoint)
    resume_hash_tokenizer = load_tokenizer(config) if resume_checkpoint is not None else None
    current_resume_hashes = build_resume_hashes(config, tokenizer=resume_hash_tokenizer)
    if resume_checkpoint is not None:
        validate_resume_checkpoint(config, resume_checkpoint, current_resume_hashes)

    logger.info("loading tokenizer, model, LoRA adapter, optimizer, scheduler")
    objects = build_training_objects(
        config,
        resume_adapter_path=adapter_dir(resume_checkpoint) if resume_checkpoint is not None else None,
        tokenizer=resume_hash_tokenizer,
    )
    if resume_checkpoint is None:
        current_resume_hashes = build_resume_hashes(config, tokenizer=objects.tokenizer)

    train_loader = dataloaders["train"].dataloader
    valid_loader = dataloaders["valid"].dataloader
    model, optimizer, train_loader, valid_loader, scheduler = prepare_with_accelerator(
        runtime,
        objects.model,
        objects.optimizer,
        train_loader,
        valid_loader,
        objects.scheduler,
    )
    state = load_trainer_state(resume_checkpoint) if resume_checkpoint is not None else TrainerState()
    if resume_checkpoint is not None:
        accelerate_state = Path(resume_checkpoint) / "accelerate_state"
        if not accelerate_state.exists():
            raise FileNotFoundError(f"resume checkpoint has no accelerate_state: {accelerate_state}")
        load_training_state_without_model(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            input_dir=accelerate_state,
        )
    else:
        from accelerate.utils import set_seed as set_accelerate_seed

        set_accelerate_seed(int(config.section("project").get("seed", 0)), device_specific=True)

    async_worker = tracker.create_async_worker()
    async_context = async_worker if async_worker is not None else nullcontext()
    registry_selector = restore_registry_selector(config, state) if bool(config.section("registry").get("enabled", False)) else None
    progress_config = config.section("progress") if "progress" in config.raw else {}
    progress = TrainingProgress(
        total_steps=int(config.section("training")["max_steps"]),
        enabled=bool(progress_config.get("enabled", True)),
        main_process=bool(getattr(accelerator, "is_main_process", True)),
    )

    trainer: RoutedTrainer
    pending_registration_paths: set[Path] = set()

    def is_main_process() -> bool:
        return bool(getattr(accelerator, "is_main_process", True))

    def log_metrics(metrics: dict, state: TrainerState) -> None:
        progress.metrics(metrics, state)
        if not is_main_process():
            return
        numeric = {key: float(value) for key, value in metrics.items() if isinstance(value, Number)}
        if not numeric:
            return
        if async_worker is not None:
            async_worker.log_metrics(numeric, step=state.global_step)
        elif tracker.enabled:
            for key, value in numeric.items():
                tracker.mlflow.log_metric(key, value, step=state.global_step)

    def on_phase(name: str, state: TrainerState) -> None:
        progress.phase(name, state)
        if is_main_process():
            logger.info("%s step=%s validation=%s checkpoint=%s", name, state.global_step, state.validation_index, state.checkpoint_index)

    def standard_eval_hook(model, dataloader, state: TrainerState) -> dict[str, float]:
        del state
        return run_standard_eval(
            model=model,
            dataloader=dataloader,
            trainer=trainer,
            config=config,
            accelerator=accelerator,
        )

    def bfcl_eval_hook(model, dataloader, state: TrainerState) -> dict[str, float]:
        del dataloader, state
        return run_bfcl_eval(
            model=model,
            tokenizer=objects.tokenizer,
            config=config,
            accelerator=accelerator,
        )

    def checkpoint_hook(model, optimizer, state: TrainerState, metrics: dict[str, float]) -> str | None:
        checkpointing = config.section("checkpointing")
        checkpoint_path = save_adapter_checkpoint(
            root_dir=checkpointing["root_dir"],
            model=model,
            optimizer=optimizer,
            state=state,
            metrics=metrics,
            accelerator=accelerator,
            config_hashes=current_resume_hashes,
        )
        if not is_main_process():
            return str(checkpoint_path)
        decision = maybe_register_candidate(config, registry_selector, async_worker, checkpoint_path, state, metrics)
        if decision is not None and async_worker is not None:
            pending_registration_paths.add(decision.checkpoint.path)
        protected_paths = set(pending_registration_paths)
        if registry_selector is not None:
            protected_paths.update(registry_selector.window_checkpoint_paths())
        deleted = prune_old_checkpoints(
            checkpointing["root_dir"],
            checkpointing.get("save_total_limit"),
            protected_paths=protected_paths,
        )
        for deleted_path in deleted:
            logger.info("pruned old checkpoint: %s", deleted_path)
        return str(checkpoint_path)

    hooks = TrainerHooks(
        on_phase=on_phase,
        run_standard_eval=standard_eval_hook,
        run_bfcl_eval=bfcl_eval_hook,
        save_checkpoint=checkpoint_hook,
        log_metrics=log_metrics,
    )
    trainer = RoutedTrainer(config, accelerator=accelerator, hooks=hooks)

    with async_context, progress:
        state = trainer.fit(
            model,
            optimizer,
            train_loader,
            scheduler=scheduler,
            valid_dataloader=valid_loader,
            state=state,
        )
        accelerator.wait_for_everyone()
        if async_worker is not None:
            async_worker.flush()
        if is_main_process():
            checkpointing = config.section("checkpointing")
            deleted = prune_old_checkpoints(checkpointing["root_dir"], checkpointing.get("save_total_limit"))
            for deleted_path in deleted:
                logger.info("pruned old checkpoint after async flush: %s", deleted_path)
    return state


def maybe_register_candidate(
    config,
    registry_selector: CandidateWindowSelector | None,
    async_worker,
    checkpoint_path,
    state: TrainerState,
    metrics: dict[str, float],
) -> RegistrationDecision | None:
    if registry_selector is None:
        return None
    decision = registry_selector.observe_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_index=state.checkpoint_index,
        global_step=state.global_step,
        metrics=metrics,
    )
    if decision is None:
        return None
    args = build_candidate_registration_args(config, decision)
    if async_worker is not None:
        async_worker.run_modelctl_register(args)
    else:
        subprocess.run(args, check=True)
    return decision


def restore_registry_selector(config, state: TrainerState) -> CandidateWindowSelector:
    """Restore candidate numbering and an incomplete selection window after resume."""

    registry = config.section("registry")
    window_size = int(registry["register_every_n_checkpoints"])
    selector = CandidateWindowSelector.from_config(
        config,
        next_candidate_index=state.checkpoint_index // window_size + 1,
    )
    pending_count = state.checkpoint_index % window_size
    if pending_count == 0:
        return selector

    first_pending_index = state.checkpoint_index - pending_count + 1
    restored = []
    for checkpoint_path in list_checkpoints(config.section("checkpointing")["root_dir"]):
        manifest = load_checkpoint_manifest(checkpoint_path)
        checkpoint_index = int(manifest.get("checkpoint_index", 0))
        if checkpoint_index < first_pending_index or checkpoint_index > state.checkpoint_index:
            continue
        metrics = manifest.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        restored.append((checkpoint_index, checkpoint_path, manifest, metrics))

    if len(restored) != pending_count:
        raise FileNotFoundError(
            "cannot restore incomplete registry selection window: "
            f"expected {pending_count} checkpoints, found {len(restored)}"
        )
    for checkpoint_index, checkpoint_path, manifest, metrics in sorted(restored):
        decision = selector.observe_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_index=checkpoint_index,
            global_step=int(manifest["global_step"]),
            metrics=metrics,
        )
        if decision is not None:
            raise RuntimeError("restoring an incomplete registry selection window unexpectedly produced a decision")
    return selector


def configure_model_source_without_side_effects(config) -> None:
    model = config.section("model")
    source = model.get("source")
    if isinstance(source, dict):
        local_dir = source.get("local_dir")
        if local_dir:
            model["resolved_model_id"] = str(Path(str(local_dir)).expanduser().resolve())
            return
    model["resolved_model_id"] = str(model["base_model_id"])


if __name__ == "__main__":
    main()
