from __future__ import annotations

import argparse
import json
from typing import Any

from eval.ru_bfcl.io import default_eval_path
from eval.ru_bfcl.matching import summary_without_rows
from eval.ru_bfcl.validator import evaluate_predictions_file


def parse_categories(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def print_text_report(summary: dict[str, Any], *, failures: int) -> None:
    print(f"total: {summary['total']}")
    print(f"passed: {summary['passed']}")
    print(f"failed: {summary['failed']}")
    print(f"accuracy: {summary['accuracy']:.4f}")
    print()
    print("by category:")
    for category, bucket in sorted(summary["by_category"].items()):
        print(f"  {category}: {bucket['passed']}/{bucket['total']} ({bucket['accuracy']:.4f})")

    failed_rows = [row for row in summary["rows"] if not row["passed"]]
    if failed_rows and failures:
        print()
        print("failures:")
        for row in failed_rows[:failures]:
            print(f"  {row['id']} [{row['category']}]: {row['reason']}")
            for issue in row.get("issues", [])[:3]:
                path = f" at {issue['path']}" if issue.get("path") else ""
                print(f"    - {issue['code']}{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RU BFCL tool-call prediction validator.")
    parser.add_argument("--eval-path", default=str(default_eval_path()))
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--categories")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-multi-turn", action="store_true")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--failures", type=int, default=20)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Evaluate all selected samples. By default CLI evaluates only ids present in the predictions file.",
    )
    parser.add_argument("--rows-out", help="Optional JSONL path for per-sample results.")
    args = parser.parse_args()

    summary = evaluate_predictions_file(
        eval_path=args.eval_path,
        predictions_path=args.predictions,
        categories=parse_categories(args.categories),
        include_multi_turn=not args.no_multi_turn,
        limit=args.limit,
        require_all=args.require_all,
        rows_out=args.rows_out,
    )

    if args.format == "json":
        print(json.dumps(summary_without_rows(summary), ensure_ascii=False, indent=2))
    else:
        print_text_report(summary, failures=args.failures)


if __name__ == "__main__":
    main()
