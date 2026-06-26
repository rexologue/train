from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from jinja2.exceptions import TemplateError

from eval.ru_bfcl import BFCLRequest, normalize_prediction


TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class BFCLTokenizerError(RuntimeError):
    """Raised when a BFCL sample cannot be rendered or tokenized."""

    def __init__(self, request: BFCLRequest, cause: BaseException):
        self.request = request
        self.cause = cause
        super().__init__(
            "BFCL tokenizer/rendering failed: "
            f"{describe_bfcl_request(request)}; "
            f"error={type(cause).__name__}: {cause}"
        )


def describe_bfcl_request(request: BFCLRequest) -> str:
    """Return compact BFCL request context for skip/debug logs."""

    roles = [message.get("role") for message in request.messages if isinstance(message, dict)]
    user_messages = [
        message
        for message in request.messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message.get("content", "").strip()
    ]
    last_role = roles[-1] if roles else None
    return (
        f"sample_id={request.sample.id} "
        f"category={request.sample.category} "
        f"source_file={request.sample.source_file} "
        f"turn_index={request.turn_index} "
        f"is_multi_turn={request.sample.is_multi_turn} "
        f"roles={roles} "
        f"last_role={last_role} "
        f"non_empty_user_messages={len(user_messages)} "
        f"tools={len(request.tools)}"
    )


def render_bfcl_request(*, tokenizer: Any, config: Any, request: BFCLRequest) -> str:
    """Render a BFCL request through the model chat template.

    Tool-calling chat templates can raise Jinja TemplateError for malformed
    conversational shapes, for example when no user query is present. Convert
    those errors into BFCLTokenizerError so eval code can quarantine the sample
    without hiding unrelated generation/runtime failures.
    """

    kwargs = {
        "tools": request.tools,
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": config.preprocessing.reasoning.enable_thinking,
    }

    try:
        return tokenizer.apply_chat_template(request.messages, **kwargs)
    except TypeError:
        # Some tokenizers/templates do not accept enable_thinking. Preserve the
        # previous compatibility path, but wrap failures from the retry as BFCL
        # tokenizer/rendering errors.
        kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(request.messages, **kwargs)
        except (TemplateError, TypeError, ValueError, KeyError) as exc:
            raise BFCLTokenizerError(request, exc) from exc
    except (TemplateError, ValueError, KeyError) as exc:
        raise BFCLTokenizerError(request, exc) from exc


def tokenize_bfcl_request(*, tokenizer: Any, config: Any, request: BFCLRequest) -> dict[str, Any]:
    """Render and tokenize a BFCL request, converting tokenizer failures."""

    rendered = render_bfcl_request(tokenizer=tokenizer, config=config, request=request)
    try:
        return tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    except (TypeError, ValueError, KeyError) as exc:
        raise BFCLTokenizerError(request, exc) from exc


class BFCLModelPredictor:
    def __init__(self, *, model: Any, tokenizer: Any, config: Any, accelerator: Any | None = None):
        """Create a BFCL predictor around a prepared model and tokenizer."""

        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = accelerator

    def __call__(self, request: BFCLRequest) -> list[dict[str, Any]]:
        inputs = tokenize_bfcl_request(tokenizer=self.tokenizer, config=self.config, request=request)
        device = self.accelerator.device if self.accelerator is not None else next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[-1])
        generation = self.config.eval.bfcl.generation

        with torch.no_grad():
            with self._evaluation_mode():
                output_ids = self._generate_with_forward_loop(inputs, generation)

        generated_ids = output_ids[0, input_length:]
        generated_ids = trim_after_eos(generated_ids, getattr(self.tokenizer, "eos_token_id", None))
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return extract_tool_calls(text)

    def _generate_with_forward_loop(self, inputs: dict[str, torch.Tensor], generation: Any) -> torch.Tensor:
        """Generate tokens through regular model forwards so FSDP stays sharded."""

        generated = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(generated)

        eos_token_ids = token_id_set(getattr(self.tokenizer, "eos_token_id", None))
        pad_token_id = resolve_pad_token_id(self.tokenizer, eos_token_ids)
        finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)

        for _ in range(generation.max_new_tokens):
            outputs = self.model(input_ids=generated, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, -1, :]
            selected = select_next_token(
                logits,
                do_sample=generation.do_sample,
                temperature=generation.temperature,
                top_p=generation.top_p,
            )
            selected = torch.where(finished, torch.full_like(selected, pad_token_id), selected)

            generated = torch.cat([generated, selected[:, None]], dim=-1)
            attention_mask = torch.cat([attention_mask, (~finished).to(attention_mask.dtype)[:, None]], dim=-1)
            if eos_token_ids:
                finished = finished | token_is_in(selected, eos_token_ids)
                if bool(finished.all().item()):
                    break

        return generated

    @contextmanager
    def _evaluation_mode(self) -> Iterator[None]:
        """Temporarily disable train-time layers while preserving the caller's model mode."""

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            yield
        finally:
            if was_training:
                self.model.train()


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    candidates = [match.group(1) for match in TOOL_CALL_BLOCK_RE.finditer(text)]
    if not candidates:
        stripped = _strip_code_fence(text.strip())
        if stripped:
            candidates = [stripped]

    calls: list[dict[str, Any]] = []
    for candidate in candidates:
        parsed = _try_parse_json(candidate)
        if parsed is None:
            continue
        try:
            calls.extend(normalize_prediction(parsed))
        except (KeyError, TypeError, ValueError):
            continue
    return calls


def _try_parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1])
    return text


def select_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Select the next token with greedy or nucleus sampling semantics."""

    if not do_sample or temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    scaled_logits = logits / temperature
    if top_p < 1.0:
        scaled_logits = apply_top_p_filter(scaled_logits, top_p)

    probabilities = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


def apply_top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Mask logits outside the nucleus sampling probability mass."""

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    sorted_indices_to_remove = cumulative_probabilities > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    filtered_sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float("-inf"))
    filtered_logits = torch.full_like(logits, float("-inf"))
    return filtered_logits.scatter(dim=-1, index=sorted_indices, src=filtered_sorted_logits)


def token_id_set(value: Any) -> set[int]:
    """Normalize tokenizer token id fields to a set."""

    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def resolve_pad_token_id(tokenizer: Any, eos_token_ids: set[int]) -> int:
    """Return a token id suitable for padding finished generations."""

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    if eos_token_ids:
        return min(eos_token_ids)
    return 0


def token_is_in(tokens: torch.Tensor, token_ids: set[int]) -> torch.Tensor:
    """Return a mask for tokens whose id is present in token_ids."""

    result = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in token_ids:
        result = result | (tokens == token_id)
    return result


def trim_after_eos(tokens: torch.Tensor, eos_token_id: Any) -> torch.Tensor:
    """Drop tokens after the first generated EOS token."""

    eos_token_ids = token_id_set(eos_token_id)
    if not eos_token_ids:
        return tokens

    eos_mask = token_is_in(tokens, eos_token_ids)
    eos_positions = torch.nonzero(eos_mask, as_tuple=False)
    if eos_positions.numel() == 0:
        return tokens

    stop = int(eos_positions[0].item()) + 1
    return tokens[:stop]
