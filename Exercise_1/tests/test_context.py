# -*- coding: utf-8 -*-
"""ContextManager 测试：消息组装、滑动窗口截断、摘要压缩、tool 配对完整性。"""

import pytest

from agent.context.manager import ContextManager
from agent.session.store import Session

SYS = "你是助手"


def _session_with(messages):
    s = Session(session_id="t")
    s.messages = list(messages)
    return s


class TestBuildMessages:
    def test_system_prompt_always_first(self):
        cm = ContextManager(SYS)
        out = cm.build_messages(_session_with([{"role": "user", "content": "hi"}]))
        assert out[0] == {"role": "system", "content": SYS}
        assert out[1]["role"] == "user"

    def test_under_limit_full_history(self):
        cm = ContextManager(SYS, max_messages=10, keep_recent=4)
        history = [{"role": "user", "content": f"m{i}"} for i in range(8)]
        out = cm.build_messages(_session_with(history))
        # system + 全量 8 条，无摘要
        assert len(out) == 9
        assert not any("[历史摘要]" in m.get("content", "") for m in out)

    def test_over_limit_triggers_summary_and_window(self):
        cm = ContextManager(SYS, max_messages=10, keep_recent=4)
        history = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        out = cm.build_messages(_session_with(history))
        # system + 摘要 + 最近 4 条
        assert len(out) == 6
        assert "[历史摘要]" in out[1]["content"]
        assert out[-1]["content"] == "m19"

    def test_summary_contains_early_user_topics(self):
        cm = ContextManager(SYS, max_messages=6, keep_recent=2)
        history = [{"role": "user", "content": "帮我查北京天气"}] + [
            {"role": "user", "content": f"m{i}"} for i in range(10)
        ]
        out = cm.build_messages(_session_with(history))
        assert "北京天气" in out[1]["content"]

    def test_window_never_starts_with_orphan_tool_message(self):
        """回归测试：切分点不得把 role=tool 与其配对的 assistant.tool_calls 切开。

        若窗口以孤儿 tool 消息开头，下一次 LLM 请求会因协议校验报 400。
        """
        cm = ContextManager(SYS, max_messages=8, keep_recent=3)
        history = [
            {"role": "user", "content": "旧问题"},
            # 一组完整的工具调用配对（恰好骑在切分边界上）
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "weather", "content": '{"city": "北京"}'},
        ] + [{"role": "user", "content": f"新消息{i}"} for i in range(10)]
        out = cm.build_messages(_session_with(history))
        body = out[2:]  # 去掉 system 与摘要
        for i, msg in enumerate(body):
            if msg.get("role") == "tool":
                # tool 消息前必须紧邻携带同 id tool_calls 的 assistant 消息
                prev = body[i - 1]
                assert prev.get("role") == "assistant"
                assert any(
                    tc.get("id") == msg.get("tool_call_id")
                    for tc in prev.get("tool_calls", [])
                )

    def test_invalid_config_asserts(self):
        with pytest.raises(AssertionError):
            ContextManager(SYS, max_messages=5, keep_recent=5)
