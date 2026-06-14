from __future__ import annotations

import json
import re
from typing import Any

from eval.ru_bfcl import BFCLRequest, normalize_prediction


TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class BFCLModelPredictor:
    def __init__(self, *, model: Any, tokenizer: Any, config: Any, accelerator: Any | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = accelerator

    def __call__(self, request: BFCLRequest) -> list[dict[str, Any]]:
        import torch

        rendered = self._render_request(request)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        device = self.accelerator.device if self.accelerator is not None else next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[-1])
        generation = self.config.section("eval")["bfcl"]["generation"]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(generation.get("max_new_tokens", 1024)),
                temperature=float(generation.get("temperature", 0.0)),
                top_p=float(generation.get("top_p", 1.0)),
                do_sample=bool(generation.get("do_sample", False)),
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                synced_gpus=bool(self.accelerator is not None and getattr(self.accelerator, "num_processes", 1) > 1),
            )
        generated_ids = output_ids[0, input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return extract_tool_calls(text)

    def _render_request(self, request: BFCLRequest) -> str:
        kwargs = {
            "tools": request.tools,
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": bool(self.config.reasoning.get("enable_thinking", False)),
        }
        try:
            return self.tokenizer.apply_chat_template(request.messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(request.messages, **kwargs)


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
