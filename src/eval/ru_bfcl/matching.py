from __future__ import annotations

import json
from typing import Any, Iterable

from eval.ru_bfcl.schema import BFCLEvalSample, MatchIssue, SampleResult, TurnResult


def parse_arguments(arguments: Any) -> Any:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        return json.loads(arguments)
    return arguments


def normalize_tool_call(call: Any) -> dict[str, Any]:
    if isinstance(call, str):
        return {"name": call, "arguments": {}}

    if not isinstance(call, dict):
        raise TypeError(f"Tool call must be dict or str, got {type(call).__name__}")

    if "function" in call:
        function = call["function"]
        return {
            "name": function["name"],
            "arguments": parse_arguments(function.get("arguments")),
        }

    return {
        "name": call["name"],
        "arguments": parse_arguments(call.get("arguments", {})),
    }


def normalize_prediction(prediction: Any) -> list[dict[str, Any]]:
    if prediction is None:
        return []

    if isinstance(prediction, dict):
        if "tool_calls" in prediction:
            return [normalize_tool_call(call) for call in prediction.get("tool_calls") or []]
        if "message" in prediction and isinstance(prediction["message"], dict):
            return normalize_prediction(prediction["message"])
        if "choices" in prediction:
            choices = prediction.get("choices") or []
            if not choices:
                return []
            return normalize_prediction(choices[0].get("message"))
        if "name" in prediction or "function" in prediction:
            return [normalize_tool_call(prediction)]

    if isinstance(prediction, list):
        return [normalize_tool_call(call) for call in prediction]

    raise TypeError(f"Unsupported prediction shape: {type(prediction).__name__}")


def normalize_multi_turn_prediction(prediction: Any, num_turns: int) -> list[list[dict[str, Any]]]:
    if prediction is None:
        return [[] for _ in range(num_turns)]

    if isinstance(prediction, dict) and "turns" in prediction:
        prediction = prediction["turns"]

    if not isinstance(prediction, list):
        raise TypeError(f"Multi-turn prediction must be a list, got {type(prediction).__name__}")

    normalized = [normalize_prediction(turn_prediction) for turn_prediction in prediction]
    while len(normalized) < num_turns:
        normalized.append([])
    return normalized


def expected_num_calls(sample: BFCLEvalSample) -> int:
    if sample.expected_type != "exact_tool_calls":
        return 0
    if sample.is_multi_turn:
        return sum(len(turn) for turn in sample.expected)
    return len(sample.expected)


def evaluate_sample(sample: BFCLEvalSample, prediction: Any) -> SampleResult:
    if sample.is_multi_turn:
        return evaluate_multi_turn_prediction(sample, prediction)
    return evaluate_single_turn_prediction(sample, prediction)


def evaluate_single_turn_prediction(sample: BFCLEvalSample, prediction: Any) -> SampleResult:
    predicted_calls = normalize_prediction(prediction)
    issues: list[MatchIssue] = []

    if sample.expected_type == "no_tool_call":
        if predicted_calls:
            issues.append(
                MatchIssue(
                    "unexpected_tool_call",
                    "model called a tool when no tool call was expected",
                    expected=0,
                    predicted=len(predicted_calls),
                )
            )
    elif sample.expected_type == "any_relevant_tool_call":
        if not predicted_calls:
            issues.append(
                MatchIssue(
                    "missing_tool_call",
                    "model did not call any tool",
                    expected="at least one tool call",
                    predicted=0,
                )
            )
    elif sample.expected_type == "exact_tool_calls":
        issues = _exact_calls_issues(sample.expected, predicted_calls)
    else:
        raise ValueError(f"Unknown expected_type: {sample.expected_type}")

    return SampleResult(
        id=sample.id,
        category=sample.category,
        source_file=sample.source_file,
        expected_type=sample.expected_type,
        is_multi_turn=False,
        passed=not issues,
        reason=_reason(issues),
        expected_num_calls=expected_num_calls(sample),
        predicted_num_calls=len(predicted_calls),
        issues=issues,
    )


def evaluate_multi_turn_prediction(sample: BFCLEvalSample, prediction: Any) -> SampleResult:
    if sample.expected_type != "exact_tool_calls":
        raise ValueError(f"Unsupported multi-turn expected_type: {sample.expected_type}")

    predicted_turns = normalize_multi_turn_prediction(prediction, len(sample.expected))
    turn_results: list[TurnResult] = []

    for turn_index in range(max(len(sample.expected), len(predicted_turns))):
        expected_turn = sample.expected[turn_index] if turn_index < len(sample.expected) else []
        predicted_turn = predicted_turns[turn_index] if turn_index < len(predicted_turns) else []
        issues = _exact_calls_issues(expected_turn, predicted_turn, order_sensitive=True)
        turn_results.append(
            TurnResult(
                turn_index=turn_index,
                passed=not issues,
                expected_num_calls=len(expected_turn),
                predicted_num_calls=len(predicted_turn),
                issues=issues,
            )
        )

    issues: list[MatchIssue] = []
    if len(predicted_turns) != len(sample.expected):
        issues.append(
            MatchIssue(
                "turn_count_mismatch",
                "number of predicted turns does not match expected turns",
                expected=len(sample.expected),
                predicted=len(predicted_turns),
            )
        )
    for turn_result in turn_results:
        issues.extend(
            MatchIssue(
                issue.code,
                issue.message,
                _format_path(f"turns[{turn_result.turn_index}]", issue.path) if issue.path else f"turns[{turn_result.turn_index}]",
                issue.expected,
                issue.predicted,
            )
            for issue in turn_result.issues
        )

    passed_turns = sum(turn.passed for turn in turn_results)
    return SampleResult(
        id=sample.id,
        category=sample.category,
        source_file=sample.source_file,
        expected_type=sample.expected_type,
        is_multi_turn=True,
        passed=not issues,
        reason=_reason(issues),
        expected_num_calls=expected_num_calls(sample),
        predicted_num_calls=sum(len(turn) for turn in predicted_turns),
        issues=issues,
        turns_total=len(sample.expected),
        turns_passed=passed_turns,
        turn_accuracy=passed_turns / len(turn_results) if turn_results else 0.0,
        turn_results=turn_results,
    )


def summarize_results(results: Iterable[SampleResult]) -> dict[str, Any]:
    rows = [result.to_dict() for result in results]
    by_category: dict[str, dict[str, Any]] = {}
    passed = sum(row["passed"] for row in rows)

    for row in rows:
        bucket = by_category.setdefault(row["category"], {"total": 0, "passed": 0, "accuracy": 0.0})
        bucket["total"] += 1
        bucket["passed"] += int(row["passed"])

    for bucket in by_category.values():
        bucket["accuracy"] = bucket["passed"] / bucket["total"] if bucket["total"] else 0.0

    return {
        "total": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "accuracy": passed / len(rows) if rows else 0.0,
        "by_category": by_category,
        "rows": rows,
    }


def summary_without_rows(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "rows"}


def _format_path(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]" if path else f"[{key}]"
    return f"{path}.{key}" if path else key


def _missing_allowed(options: Any) -> bool:
    if isinstance(options, list):
        return "" in options or None in options
    return options == "" or options is None


def _value_match_issue(predicted: Any, options: Any, path: str) -> MatchIssue | None:
    if isinstance(options, list):
        if options == []:
            if predicted == []:
                return None
            return MatchIssue(
                "value_mismatch",
                "expected an explicit empty array",
                path,
                expected=[],
                predicted=predicted,
            )
        if any(_option_matches(predicted, option) for option in options):
            return None
        return MatchIssue(
            "value_mismatch",
            "argument value does not match any allowed option",
            path,
            expected=options,
            predicted=predicted,
        )

    if _option_matches(predicted, options):
        return None
    return MatchIssue(
        "value_mismatch",
        "argument value does not match expected value",
        path,
        expected=options,
        predicted=predicted,
    )


def _option_matches(predicted: Any, option: Any) -> bool:
    if isinstance(option, dict):
        return isinstance(predicted, dict) and not _argument_match_issues(predicted, option)

    if isinstance(option, list):
        if not isinstance(predicted, list) or len(option) != len(predicted):
            return False
        return all(_option_matches(predicted_item, option_item) for predicted_item, option_item in zip(predicted, option))

    return predicted == option


def _argument_match_issues(
    predicted_args: dict[str, Any],
    argument_options: dict[str, Any],
    *,
    path: str = "arguments",
) -> list[MatchIssue]:
    issues: list[MatchIssue] = []

    if not isinstance(predicted_args, dict):
        return [
            MatchIssue(
                "invalid_arguments",
                "tool arguments must be an object",
                path,
                expected="object",
                predicted=predicted_args,
            )
        ]

    expected_keys = set(argument_options)
    predicted_keys = set(predicted_args)

    for key in sorted(predicted_keys - expected_keys):
        issues.append(
            MatchIssue(
                "unexpected_argument",
                "unexpected argument in exact tool call",
                _format_path(path, key),
                predicted=predicted_args[key],
            )
        )

    for key, options in argument_options.items():
        key_path = _format_path(path, key)
        if key not in predicted_args:
            if _missing_allowed(options):
                continue
            issues.append(
                MatchIssue(
                    "missing_argument",
                    "required argument is missing",
                    key_path,
                    expected=options,
                )
            )
            continue

        issue = _value_match_issue(predicted_args[key], options, key_path)
        if issue is not None:
            issues.append(issue)

    return issues


def _call_match_issues(expected_call: dict[str, Any], predicted_call: dict[str, Any], path: str) -> list[MatchIssue]:
    issues: list[MatchIssue] = []
    if expected_call["name"] != predicted_call.get("name"):
        issues.append(
            MatchIssue(
                "tool_name_mismatch",
                "tool name does not match",
                _format_path(path, "name"),
                expected=expected_call["name"],
                predicted=predicted_call.get("name"),
            )
        )
        return issues

    predicted_args = predicted_call.get("arguments", {})
    if "argument_options" in expected_call:
        return _argument_match_issues(predicted_args, expected_call["argument_options"], path=_format_path(path, "arguments"))

    expected_args = expected_call.get("canonical_arguments", {})
    if predicted_args != expected_args:
        issues.append(
            MatchIssue(
                "arguments_mismatch",
                "tool arguments do not match canonical arguments",
                _format_path(path, "arguments"),
                expected=expected_args,
                predicted=predicted_args,
            )
        )
    return issues


def _exact_calls_issues(
    expected_calls: list[dict[str, Any]],
    predicted_calls: list[dict[str, Any]],
    *,
    order_sensitive: bool = False,
) -> list[MatchIssue]:
    if len(expected_calls) != len(predicted_calls):
        return [
            MatchIssue(
                "tool_call_count_mismatch",
                "number of tool calls does not match",
                expected=len(expected_calls),
                predicted=len(predicted_calls),
            )
        ]

    if order_sensitive:
        issues: list[MatchIssue] = []
        for index, (expected, predicted) in enumerate(zip(expected_calls, predicted_calls)):
            issues.extend(_call_match_issues(expected, predicted, f"tool_calls[{index}]"))
        return issues

    unmatched = list(enumerate(predicted_calls))
    best_issue: MatchIssue | None = None
    for expected_index, expected in enumerate(expected_calls):
        best_candidate: tuple[int, list[MatchIssue]] | None = None
        for predicted_index, predicted in unmatched:
            issues = _call_match_issues(expected, predicted, f"tool_calls[{predicted_index}]")
            if not issues:
                unmatched = [(i, call) for i, call in unmatched if i != predicted_index]
                break
            if best_candidate is None or len(issues) < len(best_candidate[1]):
                best_candidate = (predicted_index, issues)
        else:
            if best_candidate is not None and best_candidate[1]:
                best_issue = best_candidate[1][0]
            return [
                best_issue
                or MatchIssue(
                    "missing_matching_tool_call",
                    "no predicted tool call matches expected call",
                    f"expected_tool_calls[{expected_index}]",
                    expected=expected,
                )
            ]

    return []


def _reason(issues: list[MatchIssue]) -> str:
    if not issues:
        return "passed"
    return issues[0].message
