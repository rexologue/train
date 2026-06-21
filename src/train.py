from __future__ import annotations

import argparse
from contextlib import nullcontext
from numbers import Number
from pathlib import Path

from accelerate.utils import set_seed as set_accelerate_seed

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
from config import load_config
from data.dataloaders import build_dataloaders, validate_training_inputs
from eval.bfcl import run_bfcl_eval
from eval.ordinary import run_standard_eval
from preprocessing.io import load_pretokenized_split_results
from preprocessing.pipeline import prepare_pretokenized_splits
from registry.modelctl_client import ModelctlClient
from registry.package import build_candidate_registration_request
from registry.selection import CandidateWindowSelector, RegistrationDecision
from tracking import ExperimentTracker
from tracking.model_source import load_model_source_resolution_from_cache
from trainer.callbacks import TrainerHooks
from trainer.distributed import create_accelerator, prepare_with_accelerator
from trainer.modeling import build_training_objects, load_tokenizer, training_steps_for_epochs
from trainer.progress import TrainingProgress
from trainer.ref_logprobs import ensure_ref_logprob_cache
from trainer.state import TrainerState
from trainer.trainer import RoutedTrainer
from utils.logging import configure_logging, get_logger
from utils.seed import set_seed


def main() -> None:
    """Top-level training orchestrator."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    logger = get_logger("train")
    logger.info("loading config: %s", args.config)
    config = load_config(args.config)
    logger.info("config loaded: project=%s run_name=%s", config.project.name, config.project.run_name)
    
    set_seed(config.project.seed)

    runtime = create_accelerator(config)
    accelerator = runtime.accelerator
    is_main_process = bool(getattr(accelerator, "is_main_process", True))

    configure_logging(is_main_process=is_main_process)

    tracker = ExperimentTracker.from_config(config)

    if not is_main_process:
        tracker.enabled = False

    with tracker:
        if is_main_process:
            model_source = tracker.resolve_model_source()

            logger.info(
                "registry model resolved: effective_model_id=%s ref=%s pulled=%s used_local=%s",
                model_source.effective_model_id,
                model_source.ref,
                model_source.pulled,
                model_source.used_local,
            )

            tracker.log_run_start(config_path=args.config)
            tracker.log_lineage()

        accelerator.wait_for_everyone()

        if not is_main_process:
            tracker.model_source_resolution = load_model_source_resolution_from_cache(config)

        if is_main_process:
            logger.info("building training data cache")
            results = prepare_pretokenized_splits(
                config,
                ["train", "valid", "test"],
                model_source=tracker.model_source_resolution,
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

            tracker.log_preprocessing_results(results)

        accelerator.wait_for_everyone()

        if not is_main_process:
            results = load_pretokenized_split_results(config, ["train", "valid", "test"])

        logger.info("building routed dataloaders")
        dataloaders = build_dataloaders(
            config,
            results,
            num_processes=int(getattr(accelerator, "num_processes", 1)),
        )

        for split, split_loader in dataloaders.splits.items():
            summary = split_loader.summary
            logger.info(
                "dataloader ready: split=%s rows=%s batches=%s short_batches=%s "
                "replica_group_size=%s padded_replica_batches=%s loss_kinds=%s path=%s",
                split,
                summary["num_rows"],
                summary["num_batches"],
                summary["num_short_batches"],
                summary["replica_group_size"],
                summary["num_padded_replica_batches"],
                summary["loss_kind_counts"],
                summary["path"],
            )
    
        if is_main_process:
            tracker.log_dataloaders(dataloaders)

        validate_training_inputs(config, dataloaders)
        logger.info("ensuring DPO reference-logprob cache")
        ref_cache_state = ensure_ref_logprob_cache(
            config=config,
            dataloaders=dataloaders,
            accelerator=accelerator,
            model_source=tracker.model_source_resolution,
        )
        if is_main_process:
            logger.info(
                "DPO ref-logprob cache: complete=%s applied=%s missing=%s path=%s",
                ref_cache_state.complete,
                ref_cache_state.applied_rows,
                ref_cache_state.missing_rows,
                ref_cache_state.cache_dir,
            )
            tracker.log_ref_logprob_cache(ref_cache_state)

        run_training(config, dataloaders, tracker, runtime=runtime)

    logger.info("training pipeline complete")


def run_training(config, dataloaders, tracker: ExperimentTracker, *, runtime) -> TrainerState:
    logger = get_logger("train")
    accelerator = runtime.accelerator

    resume_checkpoint = resolve_resume_checkpoint(config)
    if resume_checkpoint is not None:
        logger.info("resuming from checkpoint: %s", resume_checkpoint)

    resume_hash_tokenizer = load_tokenizer(config) if resume_checkpoint is not None else None
    current_resume_hashes = build_resume_hashes(
        config,
        tokenizer=resume_hash_tokenizer,
        model_source=tracker.model_source_resolution,
    )

    if resume_checkpoint is not None:
        validate_resume_checkpoint(config, resume_checkpoint, current_resume_hashes)

    total_steps = training_steps_for_epochs(
        config,
        dataloaders["train"].dataloader,
        num_processes=int(getattr(accelerator, "num_processes", 1)),
    )
    logger.info(
        "loading tokenizer, model, LoRA adapter, optimizer, scheduler: epochs=%s optimizer_steps=%s",
        config.training.num_epochs,
        total_steps,
    )
    objects = build_training_objects(
        config,
        total_steps=total_steps,
        resume_adapter_path=adapter_dir(resume_checkpoint) if resume_checkpoint is not None else None,
        tokenizer=resume_hash_tokenizer,
    )
    if resume_checkpoint is None:
        current_resume_hashes = build_resume_hashes(
            config,
            tokenizer=objects.tokenizer,
            model_source=tracker.model_source_resolution,
        )

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
    prepared_total_steps = training_steps_for_epochs(config, train_loader)

    if prepared_total_steps != total_steps:
        raise RuntimeError(
            "resolved optimizer-step count changed after Accelerate DataLoader sharding: "
            f"before_prepare={total_steps} after_prepare={prepared_total_steps}"
        )

    total_micro_batches = len(train_loader) * config.training.num_epochs
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
        set_accelerate_seed(config.project.seed, device_specific=True)

    async_worker = tracker.create_async_worker()
    async_context = async_worker if async_worker is not None else nullcontext()
    registry_selector = restore_registry_selector(config, state)

    progress = TrainingProgress(
        total_steps=total_steps,
        enabled=config.progress.enabled,
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
        checkpoint_path = save_adapter_checkpoint(
            root_dir=config.checkpoint_dir,
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

        deleted = prune_checkpoint_retention(
            config,
            registry_selector,
            pending_registration_paths,
            protect_pending_registry=True,
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
            total_steps=total_steps,
            total_micro_batches=total_micro_batches,
        )

        accelerator.wait_for_everyone()
        if async_worker is not None:
            async_worker.flush()

        if is_main_process():
            deleted = prune_checkpoint_retention(
                config,
                registry_selector,
                pending_registration_paths,
                protect_pending_registry=False,
            )

            for deleted_path in deleted:
                logger.info("pruned old checkpoint after async flush: %s", deleted_path)

    return state


def prune_checkpoint_retention(
    config,
    registry_selector: CandidateWindowSelector | None,
    pending_registration_paths: set[Path],
    *,
    protect_pending_registry: bool,
) -> list[Path]:
    protected_paths: set[Path] = set()
    if protect_pending_registry:
        protected_paths.update(pending_registration_paths)
    if registry_selector is not None:
        protected_paths.update(registry_selector.window_checkpoint_paths())

    deleted = prune_old_checkpoints(
        config.checkpoint_dir,
        config.checkpointing.save_total_limit,
        protected_paths=protected_paths,
    )
    if not protect_pending_registry:
        pending_registration_paths.clear()
    return deleted


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

    request = build_candidate_registration_request(config, decision)
    if async_worker is not None:
        async_worker.run_modelctl_register(request)
    else:
        ModelctlClient(tracking_uri=config.mlflow.tracking_uri).register(request)

    return decision


def restore_registry_selector(config, state: TrainerState) -> CandidateWindowSelector:
    """Restore candidate numbering and an incomplete selection window after resume."""

    window_size = config.registry.register_every_n_checkpoints
    selector = CandidateWindowSelector.from_config(
        config,
        next_candidate_index=state.checkpoint_index // window_size + 1,
    )

    pending_count = state.checkpoint_index % window_size
    if pending_count == 0:
        return selector

    first_pending_index = state.checkpoint_index - pending_count + 1
    restored = []
    for checkpoint_path in list_checkpoints(config.checkpoint_dir):
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

if __name__ == "__main__":
    main()
