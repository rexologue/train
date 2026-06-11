from __future__ import annotations


def validate_tool_call_prediction(prediction: dict, expected: dict) -> bool:
    return prediction == expected

