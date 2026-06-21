from __future__ import annotations

import json
from typing import Any

from eval.predictors import BFCLModelPredictor
from eval.ru_bfcl import BFCLRequest, BFCLValidator, dump_jsonl


def run_bfcl_eval(
    *,
    model: Any,
    tokenizer: Any,
    config: Any,
    accelerator: Any | None = None,
) -> dict[str, float]:
    bfcl_config = config.eval.bfcl
    validator = BFCLValidator.from_jsonl(
        None,
        categories=set(bfcl_config.categories) if bfcl_config.categories else None,
        include_multi_turn=bfcl_config.include_multi_turn,
        limit=bfcl_config.limit,
    )
    predictor = BFCLModelPredictor(model=model, tokenizer=tokenizer, config=config, accelerator=accelerator)

    if accelerator is None:
        predictions_by_id = _predict_samples(validator.samples, predictor)
        summary = validator.evaluate_predictions(predictions_by_id)
        _write_rows(config, summary)
        return _summary_metrics(summary)

    # FSDP forwards are collective. Every rank must execute the
    # same requests in the same order; splitting variable-length/multi-turn
    # samples between ranks can deadlock when ranks perform different numbers
    # of generate calls.
    predictions_by_id = _predict_samples(validator.samples, predictor)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return {}

    summary = validator.evaluate_predictions(predictions_by_id)
    _write_rows(config, summary)
    return _summary_metrics(summary)


def _predict_samples(samples: list[Any], predictor: BFCLModelPredictor) -> dict[str, Any]:
    predictions: dict[str, Any] = {}
    for sample in samples:
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
    return predictions


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
    """Return Qwen-template-compatible function arguments for prior tool calls."""

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


def _summary_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "eval/bfcl/accuracy": float(summary["accuracy"]),
        "eval/bfcl/total": float(summary["total"]),
        "eval/bfcl/passed": float(summary["passed"]),
        "eval/bfcl/failed": float(summary["failed"]),
    }
    for category, bucket in summary.get("by_category", {}).items():
        metrics[f"eval/bfcl/{category}/accuracy"] = float(bucket["accuracy"])
        metrics[f"eval/bfcl/{category}/total"] = float(bucket["total"])
    return metrics


def _write_rows(config: Any, summary: dict[str, Any]) -> None:
    dump_jsonl(config.bfcl_rows_path, summary["rows"])


def merge_prediction_items(items: list[tuple[str, Any]]) -> dict[str, Any]:
    return {sample_id: prediction for sample_id, prediction in items}
