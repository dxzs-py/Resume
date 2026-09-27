# -*- coding: utf-8 -*-
"""Runtime 主循环测试：用 fake LLM 验证分支决策、工具回填、迭代上限、trace。

覆盖题目 Loop 四步：
  Step1 接收输入 → Step2 判断直答/调工具 → Step3 执行工具 →
  Step4 依据工具结果决定继续循环或返回答案。
"""

import json
from types import SimpleNamespace

from agent.context.manager import ContextManager
from agent.runtime import AgentRuntime
from agent.session.store import Session, SessionStore
from agent.tools.base import Tool, ToolRegistry

SYS = "测试系统提示"


class FakeLLM:
    """按脚本依次返回预设消息的假 LLM 客户端。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        return self.script.pop(0)


def _final(content):
    return SimpleNamespace(content=content, tool_calls=None)


def _tool_call_msg(call_id, name, args, thought=None):
    return SimpleNamespace(
        content=thought,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                type="function",
                function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
            )
        ],
    )


def _registry():
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="echo",
            description="回声测试",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            func=lambda a: f"echo: {a['text']}",
        )
    )
    return reg


def _build(script, tmp_path, max_iterations=8):
    llm = FakeLLM(script)
    store = SessionStore(str(tmp_path / "sessions"))
    session = Session(session_id="rt_test")
    ctx = ContextManager(SYS, max_messages=40, keep_recent=12)
    trace_file = str(tmp_path / "trace.jsonl")
    rt = AgentRuntime(
        llm, _registry(), store, session, ctx,
        max_iterations=max_iterations, trace_file=trace_file,
    )
    return rt, llm, store, trace_file


class TestLoop:
    def test_direct_answer_no_tools(self, tmp_path):
        """Step2 判定为直接回复：一次 LLM 调用即返回，不执行工具。"""
        rt, llm, store, _ = _build([_final("你好呀")], tmp_path)
        result = rt.run("你好")
        assert result.answer == "你好呀"
        assert result.stats["llm_calls"] == 1
        assert result.stats["tool_calls"] == 0
        # 消息序列：user + assistant
        assert [m["role"] for m in rt.session.messages] == ["user", "assistant"]
        assert store.load("rt_test").message_count == 2  # 已落盘

    def test_tool_call_then_final_answer(self, tmp_path):
        """完整循环：调工具 → 结果回填 → 继续循环 → 最终答案。"""
        rt, llm, _, _ = _build(
            [_tool_call_msg("c1", "echo", {"text": "hi"}, thought="调用回声"), _final("工具说是: echo: hi")],
            tmp_path,
        )
        result = rt.run("做个回声")
        assert result.answer == "工具说是: echo: hi"
        assert result.stats["llm_calls"] == 2
        assert result.stats["tool_calls"] == 1
        assert result.stats["iterations"] == 2
        roles = [m["role"] for m in rt.session.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        # tool 消息与 tool_call_id 配对
        tool_msg = rt.session.messages[2]
        assert tool_msg["tool_call_id"] == "c1"
        assert tool_msg["content"] == "echo: hi"
        # assistant 消息保留 tool_calls 与思考过程
        asst = rt.session.messages[1]
        assert asst["content"] == "调用回声"
        assert asst["tool_calls"][0]["id"] == "c1"

    def test_tools_injected_into_llm_request(self, tmp_path):
        """LLM 请求携带工具 schema，LLM 据此自主决策。"""
        rt, llm, _, _ = _build([_final("ok")], tmp_path)
        rt.run("hi")
        tools = llm.calls[0]["tools"]
        assert tools and tools[0]["function"]["name"] == "echo"
        assert llm.calls[0]["messages"][0]["role"] == "system"

    def test_unknown_tool_error_backfilled_for_self_correction(self, tmp_path):
        """LLM 幻觉出未注册工具：错误文本回传，循环继续促其纠正。"""
        rt, _, _, _ = _build(
            [_tool_call_msg("c1", "nonexistent", {}), _final("明白了，该工具不可用")],
            tmp_path,
        )
        result = rt.run("试试")
        tool_msg = rt.session.messages[2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["content"].startswith("Error")
        assert "工具不存在" in tool_msg["content"]
        assert result.stats["iterations"] == 2

    def test_invalid_arguments_error_backfilled(self, tmp_path):
        """参数不是合法 JSON：不执行工具，错误回传。"""
        bad = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(id="c9", type="function",
                                        function=SimpleNamespace(name="echo", arguments="{bad"))],
        )
        rt, _, _, _ = _build([bad, _final("已纠正")], tmp_path)
        rt.run("x")
        assert rt.session.messages[2]["content"].startswith("Error")

    def test_tool_exception_returns_error_string(self, tmp_path):
        """工具运行时抛异常：错误以字符串回传，不中断循环。"""
        reg = ToolRegistry()
        reg.register(Tool(name="boom", description="",
                          parameters={"type": "object", "properties": {}, "required": []},
                          func=lambda a: 1 / 0))
        llm = FakeLLM([_tool_call_msg("c1", "boom", {}), _final("工具坏了")])
        store = SessionStore(str(tmp_path / "s"))
        rt = AgentRuntime(llm, reg, store, Session(session_id="rt2"),
                          ContextManager(SYS), trace_file=str(tmp_path / "t.jsonl"))
        rt.run("炸")
        assert "ZeroDivisionError" in rt.session.messages[2]["content"]

    def test_max_iterations_reached(self, tmp_path):
        """LLM 一直要求调工具：达到迭代上限后返回系统提示。"""
        script = [_tool_call_msg(f"c{i}", "echo", {"text": str(i)}) for i in range(20)]
        rt, _, _, _ = _build(script, tmp_path, max_iterations=3)
        result = rt.run("停不下来")
        assert "最大迭代上限" in result.answer
        assert result.stats["iterations"] == 3
        assert result.stats["llm_calls"] == 3

    def test_trace_events_written(self, tmp_path):
        """每次 run 产生完整 trace：输入、请求、响应、工具、统计。"""
        rt, _, _, trace_file = _build(
            [_tool_call_msg("c1", "echo", {"text": "hi"}), _final("done")], tmp_path
        )
        rt.run("trace 测试")
        lines = [json.loads(l) for l in open(trace_file, encoding="utf-8")]
        types = [e["type"] for e in lines]
        assert "user_input" in types
        assert "llm_request" in types
        assert "llm_response" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "run_stats" in types
        assert all(e.get("session_id") == "rt_test" for e in lines)

    def test_followup_uses_history(self, tmp_path):
        """追问：第二次请求的 messages 包含第一轮全部历史。"""
        rt, llm, _, _ = _build([_final("第一次回答"), _final("第二次回答")], tmp_path)
        rt.run("第一问")
        rt.run("那刚才说的能再展开吗")
        second = llm.calls[1]["messages"]
        contents = [m.get("content", "") for m in second]
        assert "第一问" in contents
        assert "第一次回答" in contents
