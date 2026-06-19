from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
from typing import Any, Iterable

from eval.ru_bfcl.schema import BFCLEvalSample


def default_eval_path() -> Path:
    return Path(str(files("eval.ru_bfcl.data").joinpath("bfcl_eval.jsonl")))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_bfcl_eval(
    path: str | Path | None = None,
    *,
    categories: set[str] | None = None,
    include_multi_turn: bool = True,
    limit: int | None = None,
) -> list[BFCLEvalSample]:
    samples: list[BFCLEvalSample] = []
    for row in load_jsonl(default_eval_path() if path is None else path):
        sample = BFCLEvalSample(
            id=row["id"],
            category=row["category"],
            source_file=row["source_file"],
            turns=row["turns"],
            tools=row["tools"],
            expected_type=row["expected_type"],
            expected=row["expected"],
            metadata={
                key: value
                for key, value in row.items()
                if key
                not in {
                    "id",
                    "category",
                    "source_file",
                    "turns",
                    "tools",
                    "expected_type",
                    "expected",
                }
            },
        )

        if categories is not None and sample.category not in categories:
            continue
        if not include_multi_turn and sample.is_multi_turn:
            continue

        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break

    return samples


def load_predictions(path: str | Path) -> dict[str, Any]:
    predictions: dict[str, Any] = {}
    for row in load_jsonl(path):
        predictions[row["id"]] = row.get("prediction", row.get("tool_calls", row.get("message")))
    return predictions
