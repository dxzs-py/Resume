# -*- coding: utf-8 -*-
"""LLM 输出解析器：把 assistant 消息解析为 思考 / 工具调用 / 最终答案 三类结构。"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 三类输出分离规则：
# 1. content + tool_calls 同时存在 → content 视为调用工具前的一句话思考（thought），
#    tool_calls 为本轮要执行的动作；
# 2. 仅有 content、无 tool_calls → 模型不再请求工具，content 即最终答案（final_answer）；
# 3. 仅有 tool_calls（content 为空）→ 纯动作，thought 记为 None。
# JSON 解析失败的 tool_call 仍保留 id/name，arguments 置 None 并记录 parse_error，
# 由 runtime 通过 tool 消息把错误回传给 LLM 促其自我纠正。


@dataclass
class ToolCallData:
    """单个工具调用的解析结果。"""

    id: str
    name: str
    arguments: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None


@dataclass
class ParsedResponse:
    """一次 assistant 消息的结构化解析结果。"""

    thought: Optional[str] = None
    tool_calls: List[ToolCallData] = field(default_factory=list)
    final_answer: Optional[str] = None


def parse_response(message) -> ParsedResponse:
    """将 openai ChatCompletionMessage 解析为 ParsedResponse。

    - 有 tool_calls：content（若非空）作为 thought，逐个解析工具调用参数；
    - 无 tool_calls：content 作为 final_answer。
    """
    content = getattr(message, "content", None)
    raw_calls = getattr(message, "tool_calls", None) or []

    if raw_calls:
        thought = content.strip() if content and content.strip() else None
        tool_calls: List[ToolCallData] = []
        for tc in raw_calls:
            function = getattr(tc, "function", None)
            call_id = getattr(tc, "id", "") or ""
            name = getattr(function, "name", "") or ""
            raw_args = getattr(function, "arguments", None)
            try:
                arguments = json.loads(raw_args) if raw_args else {}
                if not isinstance(arguments, dict):
                    raise TypeError("工具参数应为 JSON 对象")
                tool_calls.append(ToolCallData(id=call_id, name=name, arguments=arguments))
            except (json.JSONDecodeError, TypeError) as e:
                tool_calls.append(
                    ToolCallData(
                        id=call_id,
                        name=name,
                        arguments=None,
                        parse_error=f"工具参数不是合法 JSON: {e}",
                    )
                )
        return ParsedResponse(thought=thought, tool_calls=tool_calls, final_answer=None)

    final_answer = (content or "").strip() or None
    return ParsedResponse(thought=None, tool_calls=[], final_answer=final_answer)


def serialize_tool_calls(tool_calls: List[ToolCallData]) -> List[dict]:
    """把解析后的工具调用重序列化为 OpenAI assistant 消息的 tool_calls 格式。

    parse_error 的调用回传空参数 {}，错误信息由 runtime 通过 tool 消息回传给 LLM。
    """
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": (
                    json.dumps(tc.arguments, ensure_ascii=False)
                    if isinstance(tc.arguments, dict)
                    else "{}"
                ),
            },
        }
        for tc in tool_calls
    ]
