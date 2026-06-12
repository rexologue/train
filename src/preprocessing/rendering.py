from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import sha256_text, stable_hash, stable_json_dumps
from preprocessing.masking import AssistantSpan, CanonicalRow, RenderedSample


FORBIDDEN_RAW_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
)


class RenderingAuditError(ValueError):
    """Raised when raw text already contains reserved template markers."""


class ChatTemplateRenderError(ValueError):
    """Raised when tokenizer chat rendering does not produce expected text."""


def reject_forbidden_raw_markers(messages: list[dict[str, Any]], enabled: bool = True) -> None:
    """Reject raw string content containing tokens owned by the chat template."""

    if not enabled:
        return
    for index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for marker in FORBIDDEN_RAW_MARKERS:
            if marker in content:
                raise RenderingAuditError(f"raw message {index} contains forbidden marker {marker!r}")


def canonical_tools_json(tools: list[dict[str, Any]] | None) -> str | None:
    """Serialize tool schemas for audit output."""

    return None if tools is None else stable_json_dumps(tools)


def tools_hash(tools: list[dict[str, Any]] | None) -> str | None:
    """Hash tool schemas when present."""

    return None if tools is None else stable_hash(tools)


def _message_completion_text(message: dict[str, Any]) -> str:
    """Fallback renderer body for tests that do not instantiate a real tokenizer."""

    if message.get("tool_calls"):
        return str(message.get("tool_calls"))
    return str(message.get("content") or "")


class QwenTemplateRenderer:
    """Small wrapper around tokenizer `apply_chat_template`.

    Production preprocessing uses the official tokenizer template. The fallback
    renderer exists only for unit tests with a char-level tokenizer; it should
    not be used for real datasets.
    """

    def __init__(self, tokenizer: Any | None, config: dict[str, Any]):
        self.tokenizer = tokenizer
        self.config = config
        self.template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
        self.template_hash = sha256_text(self.template or "fallback-test-template")

    def render_sft(self, row: CanonicalRow) -> RenderedSample:
        """Render an SFT/tool row and return assistant spans when available."""

        reject_forbidden_raw_markers(row.messages, self.config.get("reject_raw_special_markers", True))

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            rendered, metadata = self._render_with_tokenizer(row.messages, row.tools)
            spans: list[AssistantSpan] = []
        else:
            rendered, spans = self._fallback_render(row.messages)
            metadata = {"renderer": "fallback", "unsupported_apply_chat_template_kwargs": []}

        metadata.update(
            {
                "render_hash": stable_hash(rendered),
                "template_hash": self.template_hash,
                "tools_hash": tools_hash(row.tools),
            }
        )
        return RenderedSample(rendered, spans, metadata)

    def render_dpo_completion(
        self,
        prompt: list[dict[str, Any]],
        completion: dict[str, Any],
    ) -> tuple[RenderedSample, tuple[int, int]]:
        """Render prompt + one completion and return the completion char range."""

        messages = [*prompt, completion]
        reject_forbidden_raw_markers(messages, self.config.get("reject_raw_special_markers", True))

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            rendered, metadata = self._render_with_tokenizer(messages, None)
            completion_text = str(completion.get("content") or "")
            start = rendered.rfind(completion_text)
            if start < 0:
                raise ChatTemplateRenderError("could not locate DPO completion text in rendered template")
            completion_range = (start, start + len(completion_text))
            spans = [AssistantSpan(len(messages) - 1, completion_range[0], completion_range[1], "content")]
        else:
            rendered, spans = self._fallback_render(messages)
            metadata = {"renderer": "fallback", "unsupported_apply_chat_template_kwargs": []}
            if not spans:
                raise ChatTemplateRenderError("DPO completion did not produce assistant span")
            last_span = spans[-1]
            completion_range = (last_span.start, last_span.end)

        metadata.update({"render_hash": stable_hash(rendered), "template_hash": self.template_hash, "tools_hash": None})
        return RenderedSample(rendered, spans, metadata), completion_range

    def _render_with_tokenizer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> tuple[str, dict[str, Any]]:
        """Call tokenizer chat template and record unsupported kwargs."""

        kwargs = dict(self.config.get("apply_chat_template_kwargs") or {})
        base_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": False}
        if tools is not None:
            base_kwargs["tools"] = tools

        try:
            rendered = self.tokenizer.apply_chat_template(messages, **base_kwargs, **kwargs)
            unsupported: list[str] = []
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(messages, **base_kwargs)
            unsupported = sorted(kwargs)

        if not isinstance(rendered, str):
            raise ChatTemplateRenderError("tokenizer.apply_chat_template(..., tokenize=False) must return str")
        return rendered, {"renderer": "tokenizer", "unsupported_apply_chat_template_kwargs": unsupported}

    @staticmethod
    def _fallback_render(messages: list[dict[str, Any]]) -> tuple[str, list[AssistantSpan]]:
        """Render a simple role-tagged transcript for unit tests."""

        parts: list[str] = []
        spans: list[AssistantSpan] = []
        cursor = 0
        for index, message in enumerate(messages):
            header = f"<|{message.get('role')}|>\n"
            parts.append(header)
            cursor += len(header)
            body = _message_completion_text(message)
            start = cursor
            parts.append(body)
            cursor += len(body)
            if message.get("role") == "assistant":
                kind = "tool_call" if message.get("tool_calls") else "content"
                spans.append(AssistantSpan(index, start, cursor, kind))
            suffix = "\n<|end|>\n"
            parts.append(suffix)
            cursor += len(suffix)
        return "".join(parts), spans
