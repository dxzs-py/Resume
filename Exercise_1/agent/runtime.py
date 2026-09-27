# -*- coding: utf-8 -*-
"""手写 Agent Runtime 主循环。

本文件是手写 Agent Runtime 核心，未使用任何 agent 框架
（无 LangChain / LangGraph / AutoGen 等），循环控制完全手工实现。

循环设计说明：

1. 为什么用 tool 角色消息回填：
   OpenAI function calling 协议要求 assistant 消息里的每个 tool_call
   必须有一条 role="tool" 且携带对应 tool_call_id 的消息与之配对，
   缺失会导致下一次 LLM 请求直接报 400。因此每执行完一个工具调用，
   runtime 立即以 role="tool" 把结果写回 session，保证消息序列在
   任意时刻都满足配对约束，可安全重放或恢复。

2. 为什么 LLMError 不在 runtime 内吞掉：
   LLM 调用失败意味着模型侧不可用（超时/连接/限流重试已耗尽），
   runtime 无法在不依赖外部干预的情况下自我恢复，继续循环只会
   重复失败。且 chat() 抛错发生在写入 assistant 消息之前，session
   中不会留下未配对的 tool_calls 消息，状态保持合法，向上传播给
   CLI 层兜底提示用户即可。

3. 保存时机的安全性：
   store.save 放在循环收尾（得到最终答案或迭代耗尽）执行，此刻
   消息序列必然满足 "assistant.tool_calls 与 tool 结果成对"，落盘
   文件可被 --resume 安全恢复；若中途保存，可能在部分 tool_call
   尚未回填时写入不合法的消息序列，恢复后再次请求会报协议错误。
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.exceptions import ToolNotFoundError
from agent.logger import get_logger, trace_event
from agent.parser import parse_response, serialize_tool_calls

logger = get_logger(__name__)


@dataclass
class RunResult:
    """一次 run 的产出：最终答案与运行统计。

    stats 固定包含: llm_calls / tool_calls / iterations / elapsed_seconds
    """

    answer: str
    stats: Dict[str, Any]


def _truncate(value: Any, limit: int) -> str:
    """把任意值转成字符串并截断到 limit 字符，用于 trace 字段瘦身。"""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(共{len(text)}字，已截断)"


class AgentRuntime:
    """手写 Agent 主循环：LLM 请求 → 解析 → 工具执行 → 结果回填，直到给出最终答案。"""

    def __init__(
        self,
        llm_client,
        registry,
        store,
        session,
        context_manager,
        max_iterations: int = 8,
        trace_file: Optional[str] = None,
    ) -> None:
        """依赖全部由外部注入，便于测试时替换假实现。

        trace_file 为 None 时回退到默认 logs/trace.jsonl。
        """
        self.llm_client = llm_client
        self.registry = registry
        self.store = store
        self.session = session
        self.context_manager = context_manager
        self.max_iterations = max_iterations
        self.trace_file = trace_file or "logs/trace.jsonl"

    def _trace(self, event: dict) -> None:
        """统一 trace 入口：自动附带 session_id，写入失败不影响主流程。"""
        event.setdefault("session_id", self.session.session_id)
        trace_event(event, self.trace_file)

    def run(self, user_input: str) -> RunResult:
        """执行一次完整 Agent 任务，手写主循环，返回最终答案与统计。

        单轮循环做的事：组装上下文 → 调 LLM → 解析响应 →
        分支 A（无工具调用）取最终答案跳出；分支 B（有工具调用）
        逐个执行并把结果以 role="tool" 回填 session，进入下一轮。
        LLMError 不在本层捕获，向上传播给 CLI 兜底。
        """
        t_start = time.perf_counter()
        stats: Dict[str, Any] = {
            "llm_calls": 0,
            "tool_calls": 0,
            "iterations": 0,
            "elapsed_seconds": 0.0,
        }

        # 1. 记录用户输入
        self.session.add_message("user", user_input)
        self._trace({"type": "user_input", "content": _truncate(user_input, 300)})

        answer = ""
        # 2. 主循环：最多 max_iterations 轮
        for iteration in range(1, self.max_iterations + 1):
            stats["iterations"] = iteration

            messages = self.context_manager.build_messages(self.session)
            self._trace(
                {"type": "llm_request", "iteration": iteration, "message_count": len(messages)}
            )

            resp = self.llm_client.chat(messages, tools=self.registry.export_schemas())
            stats["llm_calls"] += 1

            parsed = parse_response(resp)

            if not parsed.tool_calls:
                # 分支 A：无工具调用 → content 即最终答案，跳出循环进入收尾
                self.session.add_message("assistant", content=resp.content)
                self._trace(
                    {
                        "type": "llm_response",
                        "outcome": "final_answer",
                        "iteration": iteration,
                        "content": _truncate(resp.content or "", 500),
                    }
                )
                answer = parsed.final_answer or (resp.content or "")
                break

            # 分支 B：有工具调用 → 先落 assistant 消息（tool_calls 与后续 tool 消息成对）
            self.session.add_message(
                "assistant",
                content=parsed.thought,
                tool_calls=serialize_tool_calls(parsed.tool_calls),
            )
            self._trace(
                {
                    "type": "llm_response",
                    "outcome": "tool_calls",
                    "iteration": iteration,
                    "thought": _truncate(parsed.thought or "", 300),
                }
            )

            for tc in parsed.tool_calls:
                t0 = time.perf_counter()
                if tc.parse_error:
                    # 参数不是合法 JSON：不执行工具，把解析错误回传给 LLM 促其自我纠正
                    result = f"Error: {tc.parse_error}"
                else:
                    try:
                        result = self.registry.execute(tc.name, tc.arguments)
                    except ToolNotFoundError as e:
                        # LLM 幻觉出未注册的工具名，同样以错误文本回传
                        result = f"Error: {e}"
                duration_ms = round((time.perf_counter() - t0) * 1000, 1)

                stats["tool_calls"] += 1
                self.session.add_message(
                    "tool", content=result, tool_call_id=tc.id, name=tc.name
                )

                self._trace(
                    {
                        "type": "tool_call",
                        "iteration": iteration,
                        "tool": tc.name,
                        "args": _truncate(tc.arguments, 300),
                    }
                )
                self._trace(
                    {
                        "type": "tool_result",
                        "iteration": iteration,
                        "tool": tc.name,
                        "result": _truncate(result, 500),
                        "duration_ms": duration_ms,
                        "is_error": isinstance(result, str) and result.startswith("Error"),
                    }
                )

        else:
            # for 循环耗尽仍未 break：任务未完成，给出系统提示
            answer = (
                f"[系统提示] 已达到单次任务最大迭代上限（{self.max_iterations} 轮），"
                "任务未完成。请简化任务或重试。"
            )
            self._trace({"type": "max_iterations_reached", "max_iterations": self.max_iterations})

        # 3. 收尾（成功与迭代耗尽共用）：此刻消息序列必然合法，可安全落盘
        stats["elapsed_seconds"] = round(time.perf_counter() - t_start, 2)
        self.store.save(self.session)
        self._trace({"type": "run_stats", "stats": stats})
        logger.info(
            "运行完成: session=%s, LLM调用=%d, 工具调用=%d, 轮次=%d, 耗时=%.2fs",
            self.session.session_id,
            stats["llm_calls"],
            stats["tool_calls"],
            stats["iterations"],
            stats["elapsed_seconds"],
        )
        return RunResult(answer=answer, stats=stats)
