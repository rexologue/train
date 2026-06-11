from __future__ import annotations

from typing import Any

from qwen35_tuning.config.hashing import sha256_text, stable_hash
from qwen35_tuning.data.schemas import AssistantSpan, CanonicalRow, RenderedSample
from qwen35_tuning.rendering.audit import reject_forbidden_raw_markers
from qwen35_tuning.rendering.tools import tools_hash


class ChatTemplateRenderError(ValueError):
    pass


def _message_completion_text(message: dict[str, Any]) -> str:
    if message.get("tool_calls"):
        return str(message.get("tool_calls"))
    return str(message.get("content") or "")


class QwenTemplateRenderer:
    def __init__(self, tokenizer: Any | None, config: dict[str, Any]):
        self.tokenizer = tokenizer
        self.config = config
        self.template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
        self.template_hash = sha256_text(self.template or "fallback-test-template")

    def render_sft(self, row: CanonicalRow) -> RenderedSample:
        reject_forbidden_raw_markers(row.messages, self.config.get("reject_raw_special_markers", True))

        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            rendered, metadata = self._render_with_tokenizer(row.messages, row.tools)
            spans = []
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

        metadata.update(
            {
                "render_hash": stable_hash(rendered),
                "template_hash": self.template_hash,
                "tools_hash": None,
            }
        )
        return RenderedSample(rendered, spans, metadata), completion_range

    def _render_with_tokenizer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> tuple[str, dict]:
        kwargs = dict(self.config.get("apply_chat_template_kwargs") or {})
        base_kwargs = {
            "tokenize": False,
            "add_generation_prompt": False,
        }
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
        parts: list[str] = []
        spans: list[AssistantSpan] = []
        cursor = 0
        for index, message in enumerate(messages):
            role = message.get("role")
            header = f"<|{role}|>\n"
            parts.append(header)
            cursor += len(header)
            body = _message_completion_text(message)
            start = cursor
            parts.append(body)
            cursor += len(body)
            end = cursor
            if role == "assistant":
                kind = "tool_call" if message.get("tool_calls") else "content"
                spans.append(AssistantSpan(index, start, end, kind))
            suffix = "\n<|end|>\n"
            parts.append(suffix)
            cursor += len(suffix)
        return "".join(parts), spans
