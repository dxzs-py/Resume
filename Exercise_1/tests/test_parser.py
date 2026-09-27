# -*- coding: utf-8 -*-
"""parser 单元测试：思考过程 / 工具调用 / 最终答案 三类输出的解析与重序列化。"""

import json
from types import SimpleNamespace

from agent.parser import parse_response, serialize_tool_calls


def _msg(content=None, tool_calls=None):
    """构造一个模拟的 ChatCompletionMessage。"""
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tc(call_id="call_1", name="calculator", args='{"expression": "1+1"}'):
    """构造一个模拟的 tool_call 对象。"""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=args),
    )


class TestFinalAnswer:
    def test_no_tool_calls_content_is_final_answer(self):
        parsed = parse_response(_msg(content="你好，我是助手"))
        assert parsed.final_answer == "你好，我是助手"
        assert parsed.tool_calls == []
        assert parsed.thought is None

    def test_empty_content_final_answer_none(self):
        parsed = parse_response(_msg(content="  "))
        assert parsed.final_answer is None


class TestToolCalls:
    def test_content_with_tool_calls_becomes_thought(self):
        parsed = parse_response(_msg(content="我先算一下", tool_calls=[_tc()]))
        assert parsed.thought == "我先算一下"
        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0].name == "calculator"
        assert parsed.tool_calls[0].arguments == {"expression": "1+1"}
        assert parsed.final_answer is None

    def test_tool_call_without_content_thought_none(self):
        parsed = parse_response(_msg(content=None, tool_calls=[_tc()]))
        assert parsed.thought is None
        assert parsed.tool_calls[0].arguments == {"expression": "1+1"}

    def test_invalid_json_arguments_records_parse_error(self):
        parsed = parse_response(_msg(tool_calls=[_tc(args="{bad json")]))
        tc = parsed.tool_calls[0]
        assert tc.arguments is None
        assert tc.parse_error is not None
        assert "JSON" in tc.parse_error

    def test_non_object_arguments_records_parse_error(self):
        parsed = parse_response(_msg(tool_calls=[_tc(args="[1,2]")]))
        tc = parsed.tool_calls[0]
        assert tc.arguments is None
        assert tc.parse_error is not None

    def test_missing_arguments_defaults_empty_dict(self):
        parsed = parse_response(_msg(tool_calls=[_tc(args=None)]))
        assert parsed.tool_calls[0].arguments == {}
        assert parsed.tool_calls[0].parse_error is None


class TestSerializeToolCalls:
    def test_roundtrip(self):
        parsed = parse_response(_msg(tool_calls=[_tc()]))
        serialized = serialize_tool_calls(parsed.tool_calls)
        assert serialized == [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"expression": "1+1"}, ensure_ascii=False),
                },
            }
        ]

    def test_parse_error_serializes_empty_args(self):
        parsed = parse_response(_msg(tool_calls=[_tc(args="{bad")]))
        serialized = serialize_tool_calls(parsed.tool_calls)
        assert serialized[0]["function"]["arguments"] == "{}"
        assert serialized[0]["id"] == "call_1"
