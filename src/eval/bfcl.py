from __future__ import annotations

from pathlib import Path
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
    eval_config = config.section("eval")
    bfcl_config = eval_config["bfcl"]
    validator = BFCLValidator.from_jsonl(
        None,
        categories=set(bfcl_config["categories"]) if bfcl_config.get("categories") else None,
        include_multi_turn=bool(bfcl_config.get("include_multi_turn", True)),
        limit=bfcl_config.get("limit"),
    )
    predictor = BFCLModelPredictor(model=model, tokenizer=tokenizer, config=config, accelerator=accelerator)

    if accelerator is None:
        predictions_by_id = _predict_samples(validator.samples, predictor)
        summary = validator.evaluate_predictions(predictions_by_id)
        _write_rows(config, summary)
        return _summary_metrics(summary)

    # FSDP forward/generate calls are collective. Every rank must execute the
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
            predictions[sample.id] = [
                predictor(
                    BFCLRequest(
                        sample=sample,
                        turn_index=turn_index,
                        messages=turn["messages"],
                        tools=sample.tools,
                    )
                )
                for turn_index, turn in enumerate(sample.turns)
            ]
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
