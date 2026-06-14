from __future__ import annotations

from types import SimpleNamespace

from eval.bfcl import merge_prediction_items, run_bfcl_eval


def test_merge_prediction_items_uses_gathered_list_of_pairs():
    assert merge_prediction_items([("a", []), ("b", [{"name": "lookup", "arguments": {}}])]) == {
        "a": [],
        "b": [{"name": "lookup", "arguments": {}}],
    }


def test_distributed_bfcl_runs_all_samples_on_every_rank_in_lockstep(monkeypatch):
    samples = [
        SimpleNamespace(id="a", is_multi_turn=False, messages=[], tools=[]),
        SimpleNamespace(id="b", is_multi_turn=False, messages=[], tools=[]),
    ]
    validator = SimpleNamespace(
        samples=samples,
        evaluate_predictions=lambda predictions: {
            "accuracy": 1.0,
            "total": len(predictions),
            "passed": len(predictions),
            "failed": 0,
            "by_category": {},
            "rows": [],
        },
    )
    seen = []

    class Predictor:
        def __init__(self, **kwargs):
            del kwargs

        def __call__(self, request):
            seen.append(request.sample.id)
            return []

    class FakeValidator:
        @classmethod
        def from_jsonl(cls, *args, **kwargs):
            del cls, args, kwargs
            return validator

    class FakeAccelerator:
        is_main_process = False

        def wait_for_everyone(self):
            return None

    config = SimpleNamespace(
        section=lambda name: {
            "eval": {
                "bfcl": {
                    "categories": None,
                    "include_multi_turn": True,
                    "limit": None,
                }
            }
        }[name]
    )
    monkeypatch.setattr("eval.bfcl.BFCLValidator", FakeValidator)
    monkeypatch.setattr("eval.bfcl.BFCLModelPredictor", Predictor)

    assert run_bfcl_eval(model=object(), tokenizer=object(), config=config, accelerator=FakeAccelerator()) == {}
    assert seen == ["a", "b"]
