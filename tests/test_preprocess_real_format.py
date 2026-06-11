from __future__ import annotations

from qwen35_tuning.config.loader import load_config
from qwen35_tuning.data.preprocess import preprocess_raw_rows
from qwen35_tuning.rendering.qwen_template import QwenTemplateRenderer

from conftest import CharTokenizer


def test_preprocess_raw_rows_rejects_zero_supervised_without_aborting():
    config = load_config("configs/smoke.yaml")
    rows = [
        {
            "sample_id": "short-dropped",
            "loss_kind": "sft_target",
            "messages": [
                {"role": "assistant", "content": "intro"},
                {"role": "user", "content": "not interested"},
                {"role": "assistant", "content": "ok"},
            ],
        },
        {
            "sample_id": "long-kept",
            "loss_kind": "sft_target",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "long answer " * 10},
            ],
        },
    ]

    processed, audits, manifest = preprocess_raw_rows(rows, "valid", QwenTemplateRenderer(None, config.section("rendering")), CharTokenizer(), config)
    assert [row["sample_id"] for row in processed] == ["long-kept"]
    assert manifest["num_raw_rows"] == 2
    assert manifest["num_rejected_rows"] == 1
    assert any(audit.get("rejected") for audit in audits)

