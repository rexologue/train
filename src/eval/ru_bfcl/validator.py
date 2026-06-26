from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from eval.ru_bfcl.io import dump_jsonl, load_bfcl_eval, load_predictions, resolve_bfcl_eval_path
from eval.ru_bfcl.matching import evaluate_sample, summarize_results
from eval.ru_bfcl.schema import BFCLRequest, BFCLEvalSample


PredictionFn = Callable[[BFCLRequest], Any]


class BFCLValidator:
    def __init__(self, samples: Iterable[BFCLEvalSample], *, source_path: str | Path | None = None):
        self.samples = list(samples)
        self.source_path = None if source_path is None else Path(source_path)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path | None = None,
        *,
        categories: set[str] | None = None,
        include_multi_turn: bool = True,
        limit: int | None = None,
    ) -> "BFCLValidator":
        source_path = resolve_bfcl_eval_path(path)
        return cls(
            load_bfcl_eval(
                source_path,
                categories=categories,
                include_multi_turn=include_multi_turn,
                limit=limit,
            ),
            source_path=source_path,
        )

    def iter_requests(self) -> Iterator[BFCLRequest]:
        for sample in self.samples:
            for turn_index, turn in enumerate(sample.turns):
                yield BFCLRequest(
                    sample=sample,
                    turn_index=turn_index,
                    messages=turn["messages"],
                    tools=sample.tools,
                )

    def evaluate_predictions(self, predictions_by_id: dict[str, Any]) -> dict[str, Any]:
        return summarize_results(evaluate_sample(sample, predictions_by_id.get(sample.id)) for sample in self.samples)

    def run(self, predict: PredictionFn) -> dict[str, Any]:
        predictions_by_id: dict[str, Any] = {}
        for sample in self.samples:
            if sample.is_multi_turn:
                predictions_by_id[sample.id] = [
                    predict(
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
                predictions_by_id[sample.id] = predict(
                    BFCLRequest(
                        sample=sample,
                        turn_index=0,
                        messages=sample.messages,
                        tools=sample.tools,
                    )
                )
        return self.evaluate_predictions(predictions_by_id)


def evaluate_predictions(
    samples: Iterable[BFCLEvalSample],
    predictions_by_id: dict[str, Any],
) -> dict[str, Any]:
    return BFCLValidator(samples).evaluate_predictions(predictions_by_id)


def evaluate_model(samples: Iterable[BFCLEvalSample], predict: PredictionFn) -> dict[str, Any]:
    return BFCLValidator(samples).run(predict)


def evaluate_predictions_file(
    *,
    eval_path: str | Path | None = None,
    predictions_path: str | Path,
    categories: set[str] | None = None,
    include_multi_turn: bool = True,
    limit: int | None = None,
    require_all: bool = False,
    rows_out: str | Path | None = None,
) -> dict[str, Any]:
    predictions = load_predictions(predictions_path)
    validator = BFCLValidator.from_jsonl(
        eval_path,
        categories=categories,
        include_multi_turn=include_multi_turn,
        limit=limit,
    )
    if not require_all:
        validator = BFCLValidator(
            (sample for sample in validator.samples if sample.id in predictions),
            source_path=validator.source_path,
        )
    summary = validator.evaluate_predictions(predictions)
    if rows_out is not None:
        dump_jsonl(rows_out, summary["rows"])
    return summary
