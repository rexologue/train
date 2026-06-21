from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from eval.bfcl import tool_calls_to_messages
from eval.ordinary import run_standard_eval
from eval.predictors import BFCLModelPredictor
from preprocessing.pipeline import enforce_preprocessing_quality
from conftest import example_config


class FixedEvalTrainer:
    def __init__(self) -> None:
        self.last_loss_metrics = {}

    def compute_loss(self, model, batch):
        del model
        if batch["loss_kind"] == "dpo_target":
            self.last_loss_metrics = {"dpo/accuracy": 0.75, "dpo/reward_margin": 1.5}
            return torch.tensor(0.5)
        self.last_loss_metrics = {}
        return torch.tensor(2.0)


def test_standard_eval_emits_route_level_metrics() -> None:
    config = SimpleNamespace(
        ignore_index=-100,
        eval=SimpleNamespace(standard=SimpleNamespace(max_batches=None)),
    )
    dataloader = [
        {
            "loss_kind": "sft_target",
            "labels": torch.tensor([[-100, 1, 2]]),
        },
        {
            "loss_kind": "dpo_target",
            "sample_id": ["pair-0", "pair-1"],
            "chosen_labels": torch.tensor([[-100, 1], [-100, 1]]),
            "rejected_labels": torch.tensor([[-100, 2], [-100, 2]]),
        },
    ]

    metrics = run_standard_eval(
        model=SimpleNamespace(eval=lambda: None, train=lambda: None),
        dataloader=dataloader,
        trainer=FixedEvalTrainer(),
        config=config,
    )

    assert metrics["eval/sft/loss"] == 2.0
    assert metrics["eval/dpo/loss"] == 0.5
    assert metrics["eval/dpo/accuracy"] == 0.75
    assert metrics["eval/dpo/reward_margin"] == 1.5
    assert "eval/loss" in metrics


class EosOnlyTokenizer:
    eos_token_id = 2
    pad_token_id = 0


class EosOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, *, input_ids, attention_mask=None, use_cache=None):
        del attention_mask
        assert use_cache is False
        self.calls += 1
        logits = torch.zeros((*input_ids.shape, 4), dtype=torch.float32)
        logits[:, -1, 2] = 10.0
        return SimpleNamespace(logits=logits)


def test_bfcl_forward_loop_stops_when_all_sequences_finish() -> None:
    model = EosOnlyModel()
    predictor = BFCLModelPredictor(
        model=model,
        tokenizer=EosOnlyTokenizer(),
        config=SimpleNamespace(),
    )

    output = predictor._generate_with_forward_loop(
        {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])},
        SimpleNamespace(max_new_tokens=8, do_sample=False, temperature=0.0, top_p=1.0),
    )

    assert model.calls == 1
    assert output.tolist() == [[1, 2]]


def test_bfcl_tool_call_context_keeps_arguments_as_mapping() -> None:
    messages = tool_calls_to_messages([{"name": "lookup", "arguments": "{\"q\":\"x\"}"}])

    assert messages[0]["function"]["arguments"] == {"q": "x"}


def test_preprocessing_quality_rejects_large_rejected_fraction() -> None:
    config = example_config(
        preprocessing={
            "quality": {
                "max_rejected_fraction": 0.1,
                "min_processed_rows_per_loss_kind": {"sft_target": 0, "sft_tool": 0, "dpo_target": 0},
                "min_supervised_tokens": 0,
            }
        }
    )

    with pytest.raises(ValueError, match="rejected fraction"):
        enforce_preprocessing_quality(
            split="train",
            config=config,
            num_raw_rows=10,
            rejected_rows=[{"row_index": 0}, {"row_index": 1}],
            stats=Counter(),
        )
