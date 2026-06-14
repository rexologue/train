from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from config import TrainingConfig, effective_model_id, file_sha256, sha256_text, stable_hash
from preprocessing.io import (
    PretokSplitResult,
    cache_root,
    dataframe_to_rows,
    load_manifest,
    manifest_path,
    read_raw_dataframe,
    resolve_split_paths,
    split_cache_is_valid,
    split_parquet_path,
    write_split_cache,
)
from preprocessing.masking import (
    CanonicalRow,
    MaskingError,
    build_labels,
    canonicalize_row,
    select_sft_target_spans,
    select_sft_tool_spans,
    stable_uniform_0_1,
    tokenize_with_offsets,
)
from preprocessing.rendering import QwenTemplateRenderer, reject_forbidden_raw_markers
from utils.logging import get_logger


ASSISTANT_HEADER = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"
THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>\s*", re.IGNORECASE | re.DOTALL)
UNKNOWN_MODEL_CONTEXT_THRESHOLD = 1_000_000_000
PREPROCESSING_SCHEMA_VERSION = 2
AUDIT_EXAMPLES_PER_LOSS_KIND = 5


def load_tokenizer(config: TrainingConfig) -> Any:
    """Instantiate the tokenizer configured for preprocessing."""

    from transformers import AutoTokenizer

    tokenizer_config = config.section("tokenizer")
    model_config = config.section("model")
    return AutoTokenizer.from_pretrained(
        effective_model_id(config),
        use_fast=bool(tokenizer_config.get("use_fast", True)),
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )


def configured_max_seq_len(config: TrainingConfig) -> int:
    """Return `preprocessing.sequence.max_seq_len` as a positive integer."""

    value = int(config.sequence.get("max_seq_len", 0))
    if value <= 0:
        raise ValueError("preprocessing.sequence.max_seq_len must be a positive integer")
    return value


def tokenizer_model_context(tokenizer: Any) -> int | None:
    """Read the tokenizer-declared model context when it is finite.

    Hugging Face tokenizers often use a very large sentinel for "unknown".
    Such values are not useful for a safety comparison, so they are treated as
    absent and logged by the caller instead of silently approving everything.
    """

    raw_value = getattr(tokenizer, "model_max_length", None)
    if raw_value is None:
        raw_value = getattr(tokenizer, "init_kwargs", {}).get("model_max_length")
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    if value <= 0 or value >= UNKNOWN_MODEL_CONTEXT_THRESHOLD:
        return None
    return value


def validate_configured_max_seq_len(config: TrainingConfig, tokenizer: Any) -> int:
    """Ensure configured sequence length does not exceed tokenizer/model context."""

    max_seq_len = configured_max_seq_len(config)
    model_context = tokenizer_model_context(tokenizer)
    if model_context is not None and max_seq_len > model_context:
        raise ValueError(f"preprocessing.sequence.max_seq_len={max_seq_len} exceeds tokenizer model_max_length={model_context}")
    return max_seq_len


def build_preprocessing_signature(config: TrainingConfig, tokenizer: Any) -> str:
    """Hash the per-split tokenization/masking contract used for cache reuse."""

    tokenizer_config = config.section("tokenizer")
    return stable_hash(
        {
            "schema_version": PREPROCESSING_SCHEMA_VERSION,
            "sequence": config.sequence,
            "rendering": config.rendering,
            "reasoning": config.reasoning,
            "masking": config.preprocessing["masking"],
            "tokenizer": {
                "effective_id": effective_model_id(config),
                "use_fast": bool(tokenizer_config.get("use_fast", True)),
                "add_special_tokens": bool(tokenizer_config.get("add_special_tokens", False)),
                "class": tokenizer.__class__.__name__,
                "chat_template_hash": sha256_text(getattr(tokenizer, "chat_template", "") or ""),
            },
        }
    )


def enforce_max_seq_len(processed: dict[str, Any], max_seq_len: int) -> None:
    """Reject tokenized samples that exceed configured max length while truncation is disabled."""

    fields = ["input_ids", "chosen_input_ids", "rejected_input_ids"]
    for field in fields:
        tokens = processed.get(field)
        if isinstance(tokens, list) and len(tokens) > max_seq_len:
            raise ValueError(f"{field} length={len(tokens)} exceeds preprocessing.sequence.max_seq_len={max_seq_len}; truncation=false")


def decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode token ids without filtering special tokens; fallback supports char-level tests."""

    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except TypeError:
            return tokenizer.decode(token_ids)
    try:
        return "".join(chr(token_id) for token_id in token_ids)
    except ValueError:
        return "".join(str(token_id) for token_id in token_ids)


def decode_labeled_token_spans(
    tokenizer: Any,
    input_ids: list[int],
    labels: list[int],
    ignore_index: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Build loss-only debug text from `input_ids` and `labels`.

    This is intentionally downstream of masking. Debug loss text must prove
    what the trainer will see: for each token position `i`, only positions with
    `labels[i] != ignore_index` are decoded. Contiguous supervised token runs
    are kept as separate spans so multi-turn SFT examples remain inspectable.
    """

    spans: list[dict[str, Any]] = []
    start: int | None = None
    for index, label in enumerate(labels + [ignore_index]):
        if label != ignore_index and start is None:
            start = index
        elif label == ignore_index and start is not None:
            token_ids = input_ids[start:index]
            spans.append({"start_token": start, "end_token": index, "text": decode_token_ids(tokenizer, token_ids)})
            start = None
    return "\n".join(span["text"] for span in spans), spans


def normalize_tool_call_arguments(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Convert OpenAI-like JSON-string tool arguments into dicts for Qwen template rendering.

    The raw dataset is allowed to store `tool_calls[].function.arguments` as a
    JSON string. The local Qwen chat template renders function parameters by
    iterating `arguments|items`, so it needs a mapping. We keep this conversion
    in preprocessing instead of mutating raw parquet.
    """

    normalized = copy.deepcopy(messages)
    converted = 0
    for message_index, message in enumerate(normalized):
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call_index, tool_call in enumerate(tool_calls):
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, dict):
                continue
            if not isinstance(arguments, str):
                raise ValueError(f"message {message_index} tool_call {call_index}: function.arguments must be JSON string or object")
            try:
                parsed = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"message {message_index} tool_call {call_index}: function.arguments is not valid JSON"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"message {message_index} tool_call {call_index}: function.arguments JSON must be object")
            function["arguments"] = parsed
            converted += 1
    return normalized, converted


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    config: TrainingConfig,
) -> tuple[str, list[str]]:
    """Render messages through tokenizer chat template with configured thinking mode.

    Qwen templates may accept `enable_thinking`. Older tokenizer versions may
    reject it. In that case we retry with only stable chat-template arguments
    and record that the kwarg was unsupported.
    """

    base_kwargs = {"tokenize": False, "add_generation_prompt": False}
    if tools is not None:
        base_kwargs["tools"] = tools
    reasoning_kwargs = {"enable_thinking": bool(config.reasoning["enable_thinking"])}
    try:
        rendered = tokenizer.apply_chat_template(messages, **base_kwargs, **reasoning_kwargs)
        unsupported: list[str] = []
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, **base_kwargs)
        unsupported = sorted(reasoning_kwargs)
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template(..., tokenize=False) must return str")
    return rendered, unsupported


def assistant_blocks(rendered: str) -> list[tuple[int, int, str]]:
    """Find assistant body ranges in Qwen rendered chat text."""

    blocks: list[tuple[int, int, str]] = []
    search_from = 0
    while True:
        header_start = rendered.find(ASSISTANT_HEADER, search_from)
        if header_start < 0:
            return blocks
        body_start = header_start + len(ASSISTANT_HEADER)
        body_end = rendered.find(IM_END, body_start)
        if body_end < 0:
            raise ValueError("assistant block has no <|im_end|>")
        blocks.append((body_start, body_end, rendered[body_start:body_end]))
        search_from = body_end + len(IM_END)


def assistant_loss_segments(body_start: int, body: str, *, include_thinking: bool) -> tuple[list[tuple[int, int]], str, int]:
    """Return assistant body ranges, optionally excluding `<think>...</think>` blocks."""

    if include_thinking:
        return [(body_start, body_start + len(body))], body, 0

    ranges: list[tuple[int, int]] = []
    text_parts: list[str] = []
    removed = 0
    cursor = 0
    for match in THINK_BLOCK_RE.finditer(body):
        if match.start() > cursor:
            ranges.append((body_start + cursor, body_start + match.start()))
            text_parts.append(body[cursor : match.start()])
        cursor = match.end()
        removed += 1
    if cursor < len(body):
        ranges.append((body_start + cursor, body_start + len(body)))
        text_parts.append(body[cursor:])
    return ranges, "".join(text_parts), removed


def apply_thinking_policy_to_ranges(
    rendered_text: str,
    ranges: list[tuple[int, int]],
    *,
    include_thinking: bool,
) -> list[tuple[int, int]]:
    """Apply the same think-block masking policy to already selected char ranges."""

    if include_thinking:
        return ranges
    output: list[tuple[int, int]] = []
    think_ranges = [(match.start(), match.end()) for match in THINK_BLOCK_RE.finditer(rendered_text)]
    for start, end in ranges:
        parts = [(start, end)]
        for think_start, think_end in think_ranges:
            next_parts: list[tuple[int, int]] = []
            for part_start, part_end in parts:
                if part_end <= think_start or part_start >= think_end:
                    next_parts.append((part_start, part_end))
                    continue
                if part_start < think_start:
                    next_parts.append((part_start, think_start))
                if think_end < part_end:
                    next_parts.append((think_end, part_end))
            parts = next_parts
        output.extend(parts)
    return output


def message_reply_chars(message: dict[str, Any], rendered_body: str) -> int:
    """Count assistant reply chars for `sft_target` length policy."""

    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    if content is not None:
        return len(json.dumps(content, ensure_ascii=False, sort_keys=True))
    if message.get("tool_calls") is not None:
        return len(rendered_body)
    return len(rendered_body)


def select_sft_loss_blocks(
    row: dict[str, Any],
    messages: list[dict[str, Any]],
    rendered: str,
    config: TrainingConfig,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]], Counter[str]]:
    """Select rendered assistant text that will become supervised SFT labels.

    `sft_tool` supervises every assistant block. `sft_target` supervises all
    long assistant blocks and samples short blocks deterministically. The
    function returns both human-readable selected text for debug and character
    ranges for token label construction.
    """

    policy = config.masking_policies["sft_tool" if row["loss_kind"] == "sft_tool" else "sft_target"]
    blocks = assistant_blocks(rendered)
    assistant_indices = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    if len(blocks) != len(assistant_indices):
        raise ValueError(f"assistant block count mismatch: rendered={len(blocks)} messages={len(assistant_indices)}")

    sample_id = str(row["payload"].get("sample_id") or row["payload"].get("id") or stable_hash(row["payload"]))
    seed = int(policy.get("short_response_sampling_seed", 42))
    min_chars = int(policy["min_guaranteed_assistant_chars"]) if row["loss_kind"] == "sft_target" else 0
    short_prob = float(policy.get("loss_on_short_assistant_reply_prob", 0.3))
    selected: list[dict[str, Any]] = []
    supervised_ranges: list[tuple[int, int]] = []
    stats: Counter[str] = Counter()
    include_thinking = bool(config.reasoning["enable_thinking"])

    for assistant_order, (message_index, (start, end, body)) in enumerate(zip(assistant_indices, blocks)):
        ranges, text, removed = assistant_loss_segments(start, body, include_thinking=include_thinking)
        stats["removed_think_blocks_from_loss"] += removed
        if row["loss_kind"] == "sft_tool":
            keep = True
            reason = "all_assistant"
            chars = message_reply_chars(messages[message_index], body)
        else:
            chars = message_reply_chars(messages[message_index], body)
            stats["sft_target_candidates"] += 1
            if chars > min_chars:
                keep = True
                reason = "long_response"
                stats["sft_target_long_kept"] += 1
            else:
                stats["sft_target_short_total"] += 1
                # Stable sampling keeps short-response decisions reproducible across runs.
                keep = stable_uniform_0_1(f"{sample_id}:{message_index}:{seed}") < short_prob
                reason = "short_sampled" if keep else "short_dropped"
                if keep:
                    stats["sft_target_short_kept"] += 1
        if keep:
            supervised_ranges.extend(ranges)
            supervised_ranges.append((end, end + len(IM_END)))
            selected.append(
                {
                    "assistant_order": assistant_order,
                    "message_index": message_index,
                    "reason": reason,
                    "assistant_chars": chars,
                    "text": text,
                }
            )
    return selected, supervised_ranges, stats


def _preprocess_sft_dataset_row(row: dict[str, Any], tokenizer: Any, config: TrainingConfig) -> tuple[dict[str, Any], dict[str, Any], Counter[str]]:
    """Render, tokenize, and label one raw SFT/tool dataset row.

    This is the main per-row route for `sft_target` and `sft_tool` in the real
    training-data pipeline. It keeps the raw parquet payload immutable,
    normalizes only the copy passed to Qwen's template, selects supervised
    assistant spans according to `loss_kind`, tokenizes the full rendered text
    with offsets, and finally builds labels that only expose selected assistant
    completion tokens to the loss.
    """

    messages = row["payload"].get("messages")
    if not isinstance(messages, list):
        raise ValueError("SFT row must contain messages list")
    messages, converted_arguments = normalize_tool_call_arguments(messages)
    tools = row["payload"].get("tools")
    if tools is not None and not isinstance(tools, list):
        raise ValueError("tools must be a list when present")
    reject_forbidden_raw_markers(messages, bool(config.rendering.get("reject_raw_special_markers", True)))
    rendered, unsupported_kwargs = apply_chat_template(tokenizer, messages, tools, config)
    selected, supervised_ranges, stats = select_sft_loss_blocks(row, messages, rendered, config)
    stats["converted_tool_argument_strings"] += converted_arguments
    for key in unsupported_kwargs:
        stats[f"unsupported_apply_chat_template_kwargs/{key}"] += 1

    encoded = tokenize_with_offsets(
        tokenizer,
        rendered,
        add_special_tokens=bool(config.section("tokenizer").get("add_special_tokens", False)),
    )
    labels, supervised_tokens = build_labels(
        encoded["input_ids"],
        encoded["offset_mapping"],
        supervised_ranges,
        config.ignore_index,
        require_positive=bool(config.preprocessing["masking"].get("require_positive_supervised_tokens", True)),
    )
    loss_only_text, loss_only_token_spans = decode_labeled_token_spans(tokenizer, encoded["input_ids"], labels, config.ignore_index)
    sample_id = str(row["payload"].get("sample_id") or row["payload"].get("id") or stable_hash(row["payload"]))
    processed = {
        "sample_id": sample_id,
        "row_index": row["row_index"],
        "loss_kind": row["loss_kind"],
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "length": len(encoded["input_ids"]),
        "num_supervised_tokens": supervised_tokens,
        "render_hash": stable_hash(rendered),
        "source_hash": stable_hash(row["payload"]),
    }
    debug = {
        "sample_id": sample_id,
        "row_index": row["row_index"],
        "loss_kind": row["loss_kind"],
        "rendered_text": rendered,
        "loss_only_text": loss_only_text,
        "loss_only_token_spans": loss_only_token_spans,
        "target_selection": [{key: value for key, value in item.items() if key != "text"} for item in selected],
        "supervised_char_ranges": supervised_ranges,
        "enable_thinking": bool(config.reasoning["enable_thinking"]),
        "unsupported_apply_chat_template_kwargs": unsupported_kwargs,
    }
    return processed, debug, stats


def last_assistant_loss_ranges(rendered: str, *, include_thinking: bool) -> tuple[list[tuple[int, int]], str, int]:
    """Return supervised ranges/text for the last assistant block in a DPO render."""

    blocks = assistant_blocks(rendered)
    if not blocks:
        raise ValueError("rendered DPO completion has no assistant block")
    start, end, body = blocks[-1]
    ranges, text, removed = assistant_loss_segments(start, body, include_thinking=include_thinking)
    return [*ranges, (end, end + len(IM_END))], text + IM_END, removed


def _preprocess_dpo_dataset_row(row: dict[str, Any], tokenizer: Any, config: TrainingConfig) -> tuple[dict[str, Any], dict[str, Any], Counter[str]]:
    """Render and tokenize one DPO row as two prompt+completion branches.

    DPO samples are intentionally not converted into an SFT-like single row.
    The prompt is rendered with `chosen` and with `rejected` separately so both
    branches get their own template tokens, offsets, labels, and render hashes.
    Only the final assistant completion range in each branch is supervised;
    `<think>...</think>` supervision follows `preprocessing.reasoning.enable_thinking`.
    """

    payload = row["payload"]
    prompt = payload.get("prompt")
    chosen = payload.get("chosen")
    rejected = payload.get("rejected")
    if not isinstance(prompt, list) or not isinstance(chosen, dict) or not isinstance(rejected, dict):
        raise ValueError("DPO row must contain prompt list and chosen/rejected objects")

    processed: dict[str, Any] = {
        "sample_id": str(payload.get("sample_id") or payload.get("id") or stable_hash(payload)),
        "row_index": row["row_index"],
        "loss_kind": row["loss_kind"],
        "source_hash": stable_hash(payload),
    }
    debug: dict[str, Any] = {
        "sample_id": processed["sample_id"],
        "row_index": row["row_index"],
        "loss_kind": row["loss_kind"],
        "target_selection": [],
    }
    stats: Counter[str] = Counter()
    include_thinking = bool(config.reasoning["enable_thinking"])
    for side, completion in [("chosen", chosen), ("rejected", rejected)]:
        messages = [*copy.deepcopy(prompt), copy.deepcopy(completion)]
        messages, converted = normalize_tool_call_arguments(messages)
        reject_forbidden_raw_markers(messages, bool(config.rendering.get("reject_raw_special_markers", True)))
        rendered, unsupported_kwargs = apply_chat_template(tokenizer, messages, None, config)
        ranges, loss_text, removed = last_assistant_loss_ranges(rendered, include_thinking=include_thinking)
        stats["converted_tool_argument_strings"] += converted
        stats["removed_think_blocks_from_loss"] += removed
        for key in unsupported_kwargs:
            stats[f"unsupported_apply_chat_template_kwargs/{key}"] += 1
        encoded = tokenize_with_offsets(
            tokenizer,
            rendered,
            add_special_tokens=bool(config.section("tokenizer").get("add_special_tokens", False)),
        )
        labels, supervised_tokens = build_labels(encoded["input_ids"], encoded["offset_mapping"], ranges, config.ignore_index)
        loss_only_text_from_labels, loss_only_token_spans = decode_labeled_token_spans(
            tokenizer,
            encoded["input_ids"],
            labels,
            config.ignore_index,
        )
        processed[f"{side}_input_ids"] = encoded["input_ids"]
        processed[f"{side}_attention_mask"] = encoded["attention_mask"]
        processed[f"{side}_labels"] = labels
        processed[f"{side}_length"] = len(encoded["input_ids"])
        processed[f"{side}_completion_token_count"] = supervised_tokens
        processed[f"{side}_render_hash"] = stable_hash(rendered)
        debug[f"{side}_rendered_text"] = rendered
        debug[f"{side}_loss_only_text"] = loss_only_text_from_labels
        debug[f"{side}_loss_only_token_spans"] = loss_only_token_spans
        debug["enable_thinking"] = include_thinking
        debug.setdefault("unsupported_apply_chat_template_kwargs", {})[side] = unsupported_kwargs
        debug["target_selection"].append({"side": side, "reason": f"dpo_{side}"})
    return processed, debug, stats


def preprocess_split(
    split: str,
    raw_path: Path,
    tokenizer: Any,
    config: TrainingConfig,
    *,
    preprocessing_signature: str | None = None,
) -> PretokSplitResult:
    """Prepare or reuse one pretokenized split cache.

    The split path is resolved by the caller. This function owns cache
    validation, dataframe read, row
    processing with a progress bar, flat parquet/debug/manifest writes, and
    rejected-row accounting.
    """

    from tqdm.auto import tqdm

    logger = get_logger(__name__)
    output_dir = cache_root(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    pretok_path = split_parquet_path(output_dir, split)
    cache_manifest_path = manifest_path(output_dir)

    raw_hash = file_sha256(raw_path)
    max_seq_len = configured_max_seq_len(config)
    valid, manifest = split_cache_is_valid(output_dir, split, raw_hash, preprocessing_signature)
    if valid and manifest is not None:
        logger.info("reusing pretokenized %s split from %s", split, pretok_path)
        row_counts = (manifest.get("rows") or {}).get(split) or {}
        legacy_split = (manifest.get("splits") or {}).get(split)
        legacy_split = legacy_split if isinstance(legacy_split, dict) else {}
        split_manifest = {
            "split": split,
            "raw_path": str(raw_path),
            "input_sha256": raw_hash,
            "pretok_sha256": (manifest.get("pretokenized") or {}).get(split) or legacy_split.get("pretok_sha256"),
            "num_raw_rows": row_counts.get("raw", legacy_split.get("num_raw_rows")),
            "num_rows": row_counts.get("processed", legacy_split.get("num_rows")),
            "num_rejected_rows": row_counts.get("rejected", legacy_split.get("num_rejected_rows")),
        }
        return PretokSplitResult(split, raw_path, output_dir, pretok_path, cache_manifest_path, True, split_manifest)

    logger.info("reading %s raw dataframe from %s", split, raw_path)
    frame = read_raw_dataframe(raw_path)
    logger.info("%s dataframe loaded: rows=%s columns=%s", split, len(frame), list(frame.columns))
    rows = dataframe_to_rows(frame)

    processed_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    rejected_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()

    for row in tqdm(rows, desc=f"preprocess {split}", unit="row"):
        stats[f"raw_loss_kind/{row['loss_kind']}"] += 1
        try:
            if row["loss_kind"] == "dpo_target":
                processed, debug, row_stats = _preprocess_dpo_dataset_row(row, tokenizer, config)
            else:
                processed, debug, row_stats = _preprocess_sft_dataset_row(row, tokenizer, config)
            enforce_max_seq_len(processed, max_seq_len)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            rejected_rows.append(
                {"row_index": row["row_index"], "loss_kind": row["loss_kind"], "reason": reason}
            )
            rejected_counts[reason] += 1
            stats["rejected_rows"] += 1
            continue

        processed_rows.append(processed)
        debug["split"] = split
        debug_rows.append(debug)
        stats.update(row_stats)
        stats[f"processed_loss_kind/{row['loss_kind']}"] += 1
        stats["tokens"] += int(processed.get("length") or 0)
        stats["supervised_tokens"] += int(processed.get("num_supervised_tokens") or 0)

    base_manifest = load_manifest(output_dir) or {}
    split_manifest = {
        "split": split,
        "raw_path": str(raw_path),
        "input_sha256": raw_hash,
        "num_raw_rows": len(rows),
        "num_rows": len(processed_rows),
        "num_rejected_rows": len(rejected_rows),
        "rejected_counts": dict(rejected_counts),
        "rejected": rejected_rows,
        "stats": dict(stats),
        "preprocessing_signature": preprocessing_signature,
    }
    write_split_cache(
        output_dir,
        split,
        processed_rows,
        debug_rows,
        split_manifest,
        base_manifest=base_manifest,
        examples_per_loss_kind=AUDIT_EXAMPLES_PER_LOSS_KIND,
    )
    logger.info("%s preprocessing complete: rows=%s rejected=%s pretok=%s", split, len(processed_rows), len(rejected_rows), pretok_path)
    if rejected_counts:
        logger.warning("%s rejected counts: %s", split, dict(rejected_counts))
    return PretokSplitResult(split, raw_path, output_dir, pretok_path, cache_manifest_path, False, split_manifest)


def prepare_pretokenized_splits(
    config: TrainingConfig,
    splits: list[str],
) -> list[PretokSplitResult]:
    """Build or reuse the tokenized training-data cache.

    The function loads exactly one tokenizer from the resolved model directory,
    logs the effective template hash, resolves configured raw splits, and delegates each split to cache-aware
    preprocessing. If hashes match an existing pretokenized split, the split is
    reused; otherwise the raw parquet is rendered/tokenized into the flat cache.
    """

    logger = get_logger(__name__)
    logger.info("loading tokenizer from model directory: %s", effective_model_id(config))
    tokenizer = load_tokenizer(config)
    max_seq_len = validate_configured_max_seq_len(config, tokenizer)
    model_context = tokenizer_model_context(tokenizer)
    preprocessing_signature = build_preprocessing_signature(config, tokenizer)
    logger.info(
        "tokenizer loaded: class=%s fast=%s template_hash=%s max_seq_len=%s model_context=%s",
        tokenizer.__class__.__name__,
        bool(getattr(tokenizer, "is_fast", False)),
        sha256_text(getattr(tokenizer, "chat_template", "") or ""),
        max_seq_len,
        model_context if model_context is not None else "unknown",
    )

    results: list[PretokSplitResult] = []
    for split, raw_path in resolve_split_paths(config, splits):
        results.append(
            preprocess_split(
                split,
                raw_path,
                tokenizer,
                config,
                preprocessing_signature=preprocessing_signature,
            )
        )
    return results


def preprocess_sft_row(row: CanonicalRow, renderer: QwenTemplateRenderer, tokenizer: Any, config: TrainingConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility helper for tests using `QwenTemplateRenderer` directly."""

    rendered = renderer.render_sft(row)
    policies = config.masking_policies
    if row.loss_kind == "sft_tool":
        target_spans, selection_summary = select_sft_tool_spans(row, rendered.assistant_spans, policies["sft_tool"])
    else:
        target_spans, selection_summary = select_sft_target_spans(row, rendered.assistant_spans, policies["sft_target"])

    supervised_ranges = apply_thinking_policy_to_ranges(
        rendered.rendered_text,
        [(span.start, span.end) for span in target_spans],
        include_thinking=bool(config.reasoning["enable_thinking"]),
    )
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
        require_positive=bool(config.preprocessing["masking"].get("require_positive_supervised_tokens", True)),
    )
    loss_only_text, loss_only_token_spans = decode_labeled_token_spans(tokenizer, encoded["input_ids"], labels, config.ignore_index)

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
        "loss_only_text": loss_only_text,
        "assistant_span_char_ranges": [(span.start, span.end) for span in rendered.assistant_spans],
        "loss_only_token_spans": loss_only_token_spans,
        "supervised_char_ranges": supervised_ranges,
        "target_selection": selection_summary.__dict__,
        "enable_thinking": bool(config.reasoning["enable_thinking"]),
    }
    return processed, audit


def preprocess_dpo_row(row: CanonicalRow, renderer: QwenTemplateRenderer, tokenizer: Any, config: TrainingConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility helper for DPO mask tests with direct renderer usage."""

    if row.prompt is None or row.chosen is None or row.rejected is None:
        raise ValueError("dpo_target row requires prompt, chosen, rejected")

    chosen_rendered, chosen_range = renderer.render_dpo_completion(row.prompt, row.chosen)
    rejected_rendered, rejected_range = renderer.render_dpo_completion(row.prompt, row.rejected)

    add_special = bool(config.section("tokenizer").get("add_special_tokens", False))
    chosen_encoded = tokenize_with_offsets(tokenizer, chosen_rendered.rendered_text, add_special_tokens=add_special)
    rejected_encoded = tokenize_with_offsets(tokenizer, rejected_rendered.rendered_text, add_special_tokens=add_special)

    chosen_ranges = apply_thinking_policy_to_ranges(
        chosen_rendered.rendered_text,
        [chosen_range],
        include_thinking=bool(config.reasoning["enable_thinking"]),
    )
    rejected_ranges = apply_thinking_policy_to_ranges(
        rejected_rendered.rendered_text,
        [rejected_range],
        include_thinking=bool(config.reasoning["enable_thinking"]),
    )

    chosen_labels, chosen_tokens = build_labels(chosen_encoded["input_ids"], chosen_encoded["offset_mapping"], chosen_ranges, config.ignore_index)
    rejected_labels, rejected_tokens = build_labels(
        rejected_encoded["input_ids"], rejected_encoded["offset_mapping"], rejected_ranges, config.ignore_index
    )
    chosen_loss_only_text, chosen_loss_only_token_spans = decode_labeled_token_spans(
        tokenizer,
        chosen_encoded["input_ids"],
        chosen_labels,
        config.ignore_index,
    )
    rejected_loss_only_text, rejected_loss_only_token_spans = decode_labeled_token_spans(
        tokenizer,
        rejected_encoded["input_ids"],
        rejected_labels,
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
        "chosen_loss_only_text": chosen_loss_only_text,
        "rejected_loss_only_text": rejected_loss_only_text,
        "chosen_loss_only_token_spans": chosen_loss_only_token_spans,
        "rejected_loss_only_token_spans": rejected_loss_only_token_spans,
        "chosen_supervised_char_ranges": chosen_ranges,
        "rejected_supervised_char_ranges": rejected_ranges,
    }
    return processed, audit


def preprocess_raw_rows(
    raw_rows: list[dict[str, Any]],
    split: str,
    renderer: QwenTemplateRenderer,
    tokenizer: Any,
    config: TrainingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compatibility batch helper used by older preprocessing tests.

    The production path reads parquet via `preprocess_split`. This helper keeps
    old unit tests focused on canonical rows and fallback renderers while still
    exercising the same target selection and token masking
    invariants. Rejected rows are reported in audit/manifest data and never
    included in the processed training rows.
    """

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
            audit_rows.append({"sample_id": row.sample_id, "split": split, "loss_kind": row.loss_kind, "rejected": True, "reason": reason})
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
