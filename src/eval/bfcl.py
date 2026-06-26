from __future__ import annotations

from collections import Counter
import json
import logging
from typing import Any, Iterable

from eval.predictors import BFCLModelPredictor, BFCLTokenizerError, describe_bfcl_request, tokenize_bfcl_request
from eval.ru_bfcl import BFCLRequest, BFCLValidator, dump_jsonl


logger = logging.getLogger(__name__)
_DISABLED_SAMPLE_IDS: set[str] = set()
_LOGGED_DATASET_KEYS: set[tuple[Any, ...]] = set()


def prepare_bfcl_eval(
    *,
    tokenizer: Any,
    config: Any,
    accelerator: Any | None = None,
) -> None:
    """Load BFCL metadata and quarantine tokenizer-invalid samples before training."""

    if not config.eval.bfcl.enabled:
        return

    validator = _load_validator(config)
    log_enabled = _is_main_process(accelerator)
    _log_loaded_dataset_once(validator, config, log_enabled=log_enabled)
    skipped = _preflight_tokenizer_validation(validator.samples, tokenizer, config, log_enabled=log_enabled)

    if skipped and log_enabled:
        logger.warning(
            "BFCL preflight disabled tokenizer-invalid samples: count=%s ids=%s",
            len(skipped),
            sorted(skipped),
        )


def run_bfcl_eval(
    *,
    model: Any,
    tokenizer: Any,
    config: Any,
    accelerator: Any | None = None,
) -> dict[str, float]:
    validator = _load_validator(config)
    log_enabled = _is_main_process(accelerator)
    _log_loaded_dataset_once(validator, config, log_enabled=log_enabled)

    loaded_sample_ids = {sample.id for sample in validator.samples}
    disabled_before_eval = _DISABLED_SAMPLE_IDS & loaded_sample_ids
    samples = _usable_samples(validator.samples)
    predictor = BFCLModelPredictor(model=model, tokenizer=tokenizer, config=config, accelerator=accelerator)

    if accelerator is None:
        predictions_by_id, skipped_during_eval = _predict_samples(samples, predictor, log_enabled=log_enabled)
        skipped_ids = disabled_before_eval | (skipped_during_eval & loaded_sample_ids)
        summary = _evaluate_usable_samples(validator.samples, predictions_by_id, skipped_ids)
        _write_rows(config, summary)
        return _summary_metrics(summary, loaded_total=len(validator.samples), skipped_total=len(skipped_ids))

    # FSDP forwards are collective. Every rank must execute the
    # same requests in the same order; splitting variable-length/multi-turn
    # samples between ranks can deadlock when ranks perform different numbers
    # of generate calls.
    predictions_by_id, skipped_during_eval = _predict_samples(samples, predictor, log_enabled=log_enabled)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return {}

    skipped_ids = disabled_before_eval | (skipped_during_eval & loaded_sample_ids)
    summary = _evaluate_usable_samples(validator.samples, predictions_by_id, skipped_ids)
    _write_rows(config, summary)
    return _summary_metrics(summary, loaded_total=len(validator.samples), skipped_total=len(skipped_ids))


def _load_validator(config: Any) -> BFCLValidator:
    bfcl_config = config.eval.bfcl
    return BFCLValidator.from_jsonl(
        getattr(bfcl_config, "path", None),
        categories=set(bfcl_config.categories) if bfcl_config.categories else None,
        include_multi_turn=bfcl_config.include_multi_turn,
        limit=bfcl_config.limit,
    )


def _log_loaded_dataset_once(validator: BFCLValidator, config: Any, *, log_enabled: bool) -> None:
    if not log_enabled:
        return

    bfcl_config = config.eval.bfcl
    key = (
        str(validator.source_path),
        tuple(bfcl_config.categories or ()),
        bool(bfcl_config.include_multi_turn),
        bfcl_config.limit,
    )
    if key in _LOGGED_DATASET_KEYS:
        return
    _LOGGED_DATASET_KEYS.add(key)

    category_counts = Counter(sample.category for sample in validator.samples)
    source_counts = Counter(sample.source_file for sample in validator.samples)
    logger.info(
        "BFCL eval dataset loaded: path=%s samples=%s by_category=%s by_source_file=%s include_multi_turn=%s limit=%s disabled_existing=%s",
        validator.source_path,
        len(validator.samples),
        _format_counter(category_counts),
        _format_counter(source_counts),
        bfcl_config.include_multi_turn,
        bfcl_config.limit,
        len(_DISABLED_SAMPLE_IDS),
    )


def _format_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _is_main_process(accelerator: Any | None) -> bool:
    if accelerator is None:
        return True
    return bool(getattr(accelerator, "is_main_process", True))


def _preflight_tokenizer_validation(
    samples: Iterable[Any],
    tokenizer: Any,
    config: Any,
    *,
    log_enabled: bool,
) -> set[str]:
    skipped: set[str] = set()
    for sample in samples:
        if sample.id in _DISABLED_SAMPLE_IDS:
            skipped.add(sample.id)
            continue

        try:
            for request in _iter_preflight_requests(sample):
                tokenize_bfcl_request(tokenizer=tokenizer, config=config, request=request)
        except BFCLTokenizerError as exc:
            _disable_sample(exc.request, exc, log_enabled=log_enabled, phase="preflight")
            skipped.add(sample.id)

    return skipped


def _iter_preflight_requests(sample: Any) -> Iterable[BFCLRequest]:
    if not sample.is_multi_turn:
        yield BFCLRequest(
            sample=sample,
            turn_index=0,
            messages=sample.messages,
            tools=sample.tools,
        )
        return

    context: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(sample.turns):
        turn_messages = list(turn["messages"])
        yield BFCLRequest(
            sample=sample,
            turn_index=turn_index,
            messages=[*context, *turn_messages],
            tools=sample.tools,
        )
        context.extend(turn_messages)


def _usable_samples(samples: Iterable[Any]) -> list[Any]:
    return [sample for sample in samples if sample.id not in _DISABLED_SAMPLE_IDS]


def _evaluate_usable_samples(
    all_samples: Iterable[Any],
    predictions_by_id: dict[str, Any],
    skipped_ids: set[str],
) -> dict[str, Any]:
    samples = [sample for sample in all_samples if sample.id not in skipped_ids]
    return BFCLValidator(samples).evaluate_predictions(predictions_by_id)


def _predict_samples(
    samples: list[Any],
    predictor: BFCLModelPredictor,
    *,
    log_enabled: bool,
) -> tuple[dict[str, Any], set[str]]:
    predictions: dict[str, Any] = {}
    skipped: set[str] = set()
    for sample in samples:
        if sample.id in _DISABLED_SAMPLE_IDS:
            skipped.add(sample.id)
            continue

        try:
            if sample.is_multi_turn:
                predictions[sample.id] = predict_multi_turn_sample(sample, predictor)
            else:
                predictions[sample.id] = predictor(
                    BFCLRequest(
                        sample=sample,
                        turn_index=0,
                        messages=sample.messages,
                        tools=sample.tools,
                    )
                )
        except BFCLTokenizerError as exc:
            _disable_sample(exc.request, exc, log_enabled=log_enabled, phase="eval")
            skipped.add(sample.id)
            predictions.pop(sample.id, None)

    return predictions, skipped


def _disable_sample(
    request: BFCLRequest,
    error: BFCLTokenizerError,
    *,
    log_enabled: bool,
    phase: str,
) -> None:
    first_seen = request.sample.id not in _DISABLED_SAMPLE_IDS
    _DISABLED_SAMPLE_IDS.add(request.sample.id)

    if log_enabled and first_seen:
        logger.warning(
            "BFCL sample disabled after tokenizer error: phase=%s %s error=%s",
            phase,
            describe_bfcl_request(request),
            error,
        )


def predict_multi_turn_sample(sample: Any, predictor: BFCLModelPredictor) -> list[Any]:
    """Predict a multi-turn BFCL sample with prior turn context."""

    context: list[dict[str, Any]] = []
    predictions: list[Any] = []
    for turn_index, turn in enumerate(sample.turns):
        turn_messages = list(turn["messages"])
        predicted_calls = predictor(
            BFCLRequest(
                sample=sample,
                turn_index=turn_index,
                messages=[*context, *turn_messages],
                tools=sample.tools,
            )
        )
        predictions.append(predicted_calls)
        context.extend(turn_messages)
        if predicted_calls:
            context.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls_to_messages(predicted_calls),
                }
            )
    return predictions


def tool_calls_to_messages(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert normalized predictions into assistant tool_call messages."""

    messages: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = call.get("name") or function.get("name")
        arguments = call.get("arguments") if "arguments" in call else function.get("arguments")
        messages.append(
            {
                "id": f"bfcl_call_{index}",
                "type": "function",
                "function": {
                    "name": str(name or ""),
                    "arguments": normalize_tool_call_arguments_for_template(arguments),
                },
            }
        )
    return messages


def normalize_tool_call_arguments_for_template(arguments: Any) -> dict[str, Any]:
    """Return chat-template-compatible function arguments for prior tool calls."""

    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _summary_metrics(summary: dict[str, Any], *, loaded_total: int, skipped_total: int) -> dict[str, float]:
    metrics = {
        "eval/bfcl/accuracy": float(summary["accuracy"]),
        "eval/bfcl/total": float(summary["total"]),
        "eval/bfcl/passed": float(summary["passed"]),
        "eval/bfcl/failed": float(summary["failed"]),
        "eval/bfcl/loaded_total": float(loaded_total),
        "eval/bfcl/skipped_total": float(skipped_total),
    }
    for category, bucket in summary.get("by_category", {}).items():
        metrics[f"eval/bfcl/{category}/accuracy"] = float(bucket["accuracy"])
        metrics[f"eval/bfcl/{category}/total"] = float(bucket["total"])
    return metrics


def _write_rows(config: Any, summary: dict[str, Any]) -> None:
    dump_jsonl(config.bfcl_rows_path, summary["rows"])


def merge_prediction_items(items: list[tuple[str, Any]]) -> dict[str, Any]:
    return {sample_id: prediction for sample_id, prediction in items}
