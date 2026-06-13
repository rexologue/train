from __future__ import annotations

from eval.bfcl import merge_prediction_items


def test_merge_prediction_items_uses_gathered_list_of_pairs():
    assert merge_prediction_items([("a", []), ("b", [{"name": "lookup", "arguments": {}}])]) == {
        "a": [],
        "b": [{"name": "lookup", "arguments": {}}],
    }
