from __future__ import annotations

from eval.ru_bfcl import (
    BFCLEvalSample,
    BFCLValidator,
    default_eval_path,
    dump_jsonl,
    evaluate_predictions_file,
    evaluate_sample,
    normalize_prediction,
)


def sample(expected, *, turns=None):
    return BFCLEvalSample(
        id="sample_0",
        category="test",
        source_file="BFCL_v3_simple.json",
        turns=turns or [{"messages": [{"role": "user", "content": "test"}]}],
        tools=[],
        expected_type="exact_tool_calls",
        expected=expected,
    )


def test_empty_array_is_required_value():
    item = sample(
        [
            {
                "name": "start_payment",
                "argument_options": {"schedule": []},
            }
        ]
    )

    missing = evaluate_sample(item, [{"name": "start_payment", "arguments": {}}])
    correct = evaluate_sample(item, [{"name": "start_payment", "arguments": {"schedule": []}}])

    assert not missing.passed
    assert missing.issues[0].code == "missing_argument"
    assert correct.passed


def test_exact_match_rejects_unexpected_arguments():
    item = sample(
        [
            {
                "name": "book",
                "argument_options": {"city": ["Moscow"]},
            }
        ]
    )

    result = evaluate_sample(
        item,
        [{"name": "book", "arguments": {"city": "Moscow", "extra": True}}],
    )

    assert not result.passed
    assert result.issues[0].code == "unexpected_argument"


def test_extra_multi_turn_predictions_are_not_truncated():
    item = sample(
        [
            [{"name": "first", "canonical_arguments": {}}],
            [{"name": "second", "canonical_arguments": {}}],
        ],
        turns=[
            {"messages": [{"role": "user", "content": "one"}]},
            {"messages": [{"role": "user", "content": "two"}]},
        ],
    )

    result = BFCLValidator([item]).evaluate_predictions(
        {
            "sample_0": [
                [{"name": "first", "arguments": {}}],
                [{"name": "second", "arguments": {}}],
                [{"name": "third", "arguments": {}}],
            ]
        }
    )

    row = result["rows"][0]
    assert not row["passed"]
    assert row["predicted_num_calls"] == 3
    assert row["issues"][0]["code"] == "turn_count_mismatch"


def test_optional_empty_string_argument_may_be_omitted():
    item = sample(
        [
            {
                "name": "lookup",
                "argument_options": {"query": ["abc"], "locale": ["", "ru"]},
            }
        ]
    )

    result = evaluate_sample(item, [{"name": "lookup", "arguments": {"query": "abc"}}])

    assert result.passed


def test_normalize_prediction_accepts_openai_tool_calls():
    prediction = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": "{\"query\":\"abc\"}",
                },
            }
        ]
    }
    expected = [{"name": "lookup", "arguments": {"query": "abc"}}]

    assert normalize_prediction(prediction) == expected


def test_evaluate_predictions_file_offline(tmp_path):
    eval_path = tmp_path / "bfcl_eval.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    rows_out = tmp_path / "rows.jsonl"

    dump_jsonl(
        eval_path,
        [
            {
                "id": "simple_0",
                "category": "simple",
                "source_file": "BFCL_v3_simple.json",
                "turns": [{"messages": [{"role": "user", "content": "test"}]}],
                "tools": [],
                "expected_type": "exact_tool_calls",
                "expected": [{"name": "lookup", "canonical_arguments": {"query": "abc"}}],
            },
            {
                "id": "irrelevance_0",
                "category": "irrelevance",
                "source_file": "BFCL_v3_irrelevance.json",
                "turns": [{"messages": [{"role": "user", "content": "test"}]}],
                "tools": [],
                "expected_type": "no_tool_call",
                "expected": [],
            },
        ],
    )
    dump_jsonl(
        predictions_path,
        [
            {"id": "simple_0", "prediction": [{"name": "lookup", "arguments": {"query": "abc"}}]},
        ],
    )

    summary = evaluate_predictions_file(
        eval_path=eval_path,
        predictions_path=predictions_path,
        require_all=False,
        rows_out=rows_out,
    )

    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert rows_out.exists()


def test_default_eval_path_is_bundled_project_data():
    path = default_eval_path()

    assert path.exists()
    assert path.parts[-4:] == ("eval", "ru_bfcl", "data", "bfcl_eval.jsonl")
