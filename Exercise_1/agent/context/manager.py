# -*- coding: utf-8 -*-
"""上下文管理器：为 LLM 组装发送的消息列表，控制上下文长度与 token 成本。"""

from typing import Any, Dict, List, Optional

from agent.session.store import Session


class ContextManager:
    """把 Session 历史组装成发送给 LLM 的 messages 列表。

    上下文放置策略：
    1. system prompt 恒置顶：全局指令优先级最高，任何截断都不影响它；
    2. 工具调用与结果原样保留在历史里：assistant 的 tool_calls 消息与
       role=tool 的结果消息必须成对出现，且 tool 消息必须携带
       tool_call_id——这是 API 协议的硬性要求，缺失会导致请求报错；
    3. 超长时滑动窗口截断 + 摘要占位：仅保留最近 keep_recent 条消息，
       更早内容压缩为一条摘要 system 消息，既保住早期话题线索，
       又控制 token 成本；
    4. max_messages / keep_recent 均可配置，可按模型窗口大小灵活调整。

    注意：迭代上限（max_iterations）由 runtime 负责，本类不管理。
    """

    def __init__(
        self,
        system_prompt: str,
        max_messages: int = 40,
        keep_recent: int = 12,
    ) -> None:
        assert keep_recent < max_messages, "keep_recent 必须小于 max_messages"
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.keep_recent = keep_recent

    @staticmethod
    def _summarize(old_messages: List[Dict[str, Any]]) -> str:
        """启发式摘要：抽取早期 user 消息要点，生成压缩占位文本。"""
        total = len(old_messages)
        user_lines: List[str] = []
        for msg in old_messages:
            if msg.get("role") == "user" and msg.get("content"):
                user_lines.append("- " + str(msg["content"])[:40])
                if len(user_lines) >= 10:
                    break
        if not user_lines:
            return f"早期 {total} 条消息已压缩。"
        return (
            f"本次对话早期内容已压缩，共 {total} 条消息。用户此前询问/执行过：\n"
            + "\n".join(user_lines)
            + "\n如需引用早期细节，请向用户确认。"
        )

    def build_messages(self, session: Session) -> List[Dict[str, Any]]:
        """组装发送给 LLM 的消息：system prompt +（超限时加摘要）+ 历史。

        未超限直接返回全量历史；超限则把窗口外旧消息替换为一条
        "[历史摘要]" system 消息，只保留最近 keep_recent 条，
        保证 tool_calls 与 tool 结果成对留在窗口内。
        """
        sys = {"role": "system", "content": self.system_prompt}
        history = session.messages
        if len(history) <= self.max_messages:
            return [sys] + list(history)
        old = history[: -self.keep_recent]
        recent = history[-self.keep_recent :]
        return [
            sys,
            {"role": "system", "content": "[历史摘要] " + self._summarize(old)},
        ] + recent
