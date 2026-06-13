from __future__ import annotations

from eval.predictors import extract_tool_calls


def test_extract_tool_calls_from_qwen_tool_call_block():
    text = '<tool_call>\n{"name":"lookup","arguments":{"query":"abc"}}\n</tool_call>'

    assert extract_tool_calls(text) == [{"name": "lookup", "arguments": {"query": "abc"}}]


def test_extract_tool_calls_from_openai_message_json():
    text = (
        '{"tool_calls":[{"type":"function","function":'
        '{"name":"lookup","arguments":"{\\"query\\":\\"abc\\"}"}}]}'
    )

    assert extract_tool_calls(text) == [{"name": "lookup", "arguments": {"query": "abc"}}]


def test_extract_tool_calls_returns_empty_for_plain_text():
    assert extract_tool_calls("No tool needed.") == []
