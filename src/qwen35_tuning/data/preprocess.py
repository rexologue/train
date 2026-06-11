from __future__ import annotations

from typing import Any

from qwen35_tuning.config.hashing import stable_hash
from qwen35_tuning.config.schema import TrainingConfig
from qwen35_tuning.data.canonicalize import canonicalize_row
from qwen35_tuning.data.schemas import CanonicalRow
from qwen35_tuning.masking.target_selection import select_sft_target_spans, select_sft_tool_spans
from qwen35_tuning.masking.token_mask import MaskingError, build_labels, tokenize_with_offsets
from qwen35_tuning.rendering.qwen_template import QwenTemplateRenderer
from qwen35_tuning.rendering.reasoning import audit_reasoning


def preprocess_sft_row(
    row: CanonicalRow,
    renderer: QwenTemplateRenderer,
    tokenizer: Any,
    config: TrainingConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    
    rendered = renderer.render_sft(row)
    policies = config.masking_policies
    
    if row.loss_kind == "sft_tool":
        target_spans, selection_summary = select_sft_tool_spans(row, rendered.assistant_spans, policies["sft_tool"])
    else:
        target_spans, selection_summary = select_sft_target_spans(row, rendered.assistant_spans, policies["sft_target"])

    supervised_ranges = [(span.start, span.end) for span in target_spans]
    reasoning_audit = audit_reasoning(rendered.rendered_text, supervised_ranges, config.reasoning)
    encoded = tokenize_with_offsets(
        tokenizer,
        rendered.rendered_text,
        add_special_tokens=bool(config.section("tokenizer").get("add_special_tokens", False)),
    )
    labels, supervised_tokens = build_labels(
        encoded["input_ids"],
        encoded["offset_mapping"],
        supervised_ranges,
        ignore_index=config.ignore_index,
        require_positive=bool(config.section("masking").get("require_positive_supervised_tokens", True)),
    )

    processed = {
        "sample_id": row.sample_id,
        "split": row.split,
        "loss_kind": row.loss_kind,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "length": len(encoded["input_ids"]),
        "num_supervised_tokens": supervised_tokens,
        "num_prompt_tokens": len(encoded["input_ids"]) - supervised_tokens,
        "num_supervised_chars": sum(end - start for start, end in supervised_ranges),
        "target_turn_indices": [span.message_index for span in target_spans],
        "target_selection_reasons": [span.reason for span in target_spans],
        "render_hash": rendered.render_metadata["render_hash"],
        "template_hash": rendered.render_metadata["template_hash"],
        "tools_hash": rendered.render_metadata["tools_hash"],
        "source_hash": row.metadata.get("source_hash"),
        "metadata": row.metadata,
    }
    audit = {
        "sample_id": row.sample_id,
        "rendered_text": rendered.rendered_text,
        "supervised_text": "\n".join(rendered.rendered_text[start:end] for start, end in supervised_ranges),
        "assistant_span_char_ranges": [(span.start, span.end) for span in rendered.assistant_spans],
        "token_supervision_ranges": supervised_ranges,
        "target_selection": selection_summary.__dict__,
        "reasoning_audit": reasoning_audit.__dict__,
    }
    return processed, audit


def preprocess_dpo_row(
    row: CanonicalRow,
    renderer: QwenTemplateRenderer,
    tokenizer: Any,
    config: TrainingConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if row.prompt is None or row.chosen is None or row.rejected is None:
        raise ValueError("dpo_target row requires prompt, chosen, rejected")

    chosen_rendered, chosen_range = renderer.render_dpo_completion(row.prompt, row.chosen)
    rejected_rendered, rejected_range = renderer.render_dpo_completion(row.prompt, row.rejected)
    audit_reasoning(chosen_rendered.rendered_text, [chosen_range], config.reasoning)
    audit_reasoning(rejected_rendered.rendered_text, [rejected_range], config.reasoning)

    add_special = bool(config.section("tokenizer").get("add_special_tokens", False))
    chosen_encoded = tokenize_with_offsets(tokenizer, chosen_rendered.rendered_text, add_special_tokens=add_special)
    rejected_encoded = tokenize_with_offsets(tokenizer, rejected_rendered.rendered_text, add_special_tokens=add_special)

    chosen_labels, chosen_tokens = build_labels(
        chosen_encoded["input_ids"],
        chosen_encoded["offset_mapping"],
        [chosen_range],
        config.ignore_index,
    )
    rejected_labels, rejected_tokens = build_labels(
        rejected_encoded["input_ids"],
        rejected_encoded["offset_mapping"],
        [rejected_range],
        config.ignore_index,
    )

    processed = {
        "sample_id": row.sample_id,
        "split": row.split,
        "loss_kind": row.loss_kind,
        "chosen_input_ids": chosen_encoded["input_ids"],
        "chosen_attention_mask": chosen_encoded["attention_mask"],
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_encoded["input_ids"],
        "rejected_attention_mask": rejected_encoded["attention_mask"],
        "rejected_labels": rejected_labels,
        "chosen_completion_token_count": chosen_tokens,
        "rejected_completion_token_count": rejected_tokens,
        "chosen_ref_logp": None,
        "rejected_ref_logp": None,
        "prompt_render_hash": stable_hash(row.prompt),
        "chosen_render_hash": chosen_rendered.render_metadata["render_hash"],
        "rejected_render_hash": rejected_rendered.render_metadata["render_hash"],
        "template_hash": chosen_rendered.render_metadata["template_hash"],
        "metadata": row.metadata,
    }
    audit = {
        "sample_id": row.sample_id,
        "chosen_rendered_text": chosen_rendered.rendered_text,
        "rejected_rendered_text": rejected_rendered.rendered_text,
        "chosen_supervised_text": chosen_rendered.rendered_text[chosen_range[0] : chosen_range[1]],
        "rejected_supervised_text": rejected_rendered.rendered_text[rejected_range[0] : rejected_range[1]],
        "chosen_token_supervision_range": chosen_range,
        "rejected_token_supervision_range": rejected_range,
    }
    return processed, audit


def preprocess_raw_rows(
    raw_rows: list[dict[str, Any]],
    split: str,
    renderer: QwenTemplateRenderer,
    tokenizer: Any,
    config: TrainingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    processed_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    rejected_counts: dict[str, int] = {}

    for index, raw in enumerate(raw_rows):
        row = canonicalize_row(raw, split, index)  # type: ignore[arg-type]
        counts[row.loss_kind] = counts.get(row.loss_kind, 0) + 1
        try:
            if row.loss_kind == "dpo_target":
                processed, audit = preprocess_dpo_row(row, renderer, tokenizer, config)
            else:
                processed, audit = preprocess_sft_row(row, renderer, tokenizer, config)
        except MaskingError as exc:
            reason = str(exc)
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            audit_rows.append(
                {
                    "sample_id": row.sample_id,
                    "split": split,
                    "loss_kind": row.loss_kind,
                    "rejected": True,
                    "reason": reason,
                }
            )
            continue
        processed_rows.append(processed)
        audit_rows.append(audit)

    manifest = {
        "split": split,
        "num_raw_rows": len(raw_rows),
        "num_rows": len(processed_rows),
        "num_rejected_rows": sum(rejected_counts.values()),
        "rejected_counts": rejected_counts,
        "loss_kind_counts": counts,
        "processed_hash": stable_hash(processed_rows),
    }
    return processed_rows, audit_rows, manifest
