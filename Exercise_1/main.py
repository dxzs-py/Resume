# -*- coding: utf-8 -*-
"""CLI 入口：手写 Agent Runtime 的命令行交互（REPL）。

用法：
    python main.py                  # 新建会话并进入交互
    python main.py --list           # 列出全部历史会话后退出
    python main.py --resume <id>    # 恢复指定历史会话继续对话
    python main.py --session-id <id>  # 以自定义 id 新建会话
"""

import argparse
import sys
from typing import List

from agent.config import Config, load_config
from agent.context.manager import ContextManager
from agent.exceptions import AgentError
from agent.llm_client import LLMClient
from agent.prompts import SYSTEM_PROMPT
from agent.runtime import AgentRuntime
from agent.session.store import Session, SessionStore
from agent.tools.base import ToolRegistry
from agent.tools.calculator import create_calculator_tool
from agent.tools.search import create_search_tool
from agent.tools.weather import create_weather_tool

# Windows 控制台 UTF-8 兜底：避免中文输出乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def build_runtime(store: SessionStore, session: Session, config: Config) -> AgentRuntime:
    """按配置组装一次可运行的 AgentRuntime（依赖注入，便于替换测试）。"""
    llm = LLMClient(config)
    registry = ToolRegistry()
    registry.register(create_calculator_tool())
    registry.register(create_search_tool())
    registry.register(create_weather_tool())
    ctx = ContextManager(
        SYSTEM_PROMPT,
        max_messages=config.context_max_messages,
        keep_recent=config.context_keep_recent,
    )
    return AgentRuntime(
        llm,
        registry,
        store,
        session,
        ctx,
        max_iterations=config.max_iterations,
        trace_file=config.trace_file,
    )


def print_session_table(sessions: List[dict]) -> None:
    """以表格形式打印会话摘要列表。"""
    print(f"{'ID':<34}{'消息数':>4}   {'更新时间':<21}  标题")
    print("-" * 80)
    for s in sessions:
        print(
            f"{s['id']:<34}{s['message_count']:>4}   {s['updated_at']:<21}  {s['title']}"
        )


def print_history(session: Session) -> None:
    """紧凑打印当前会话消息历史。"""
    if not session.messages:
        print("（当前会话暂无消息）")
        return
    for i, msg in enumerate(session.messages, 1):
        role = msg.get("role", "?")
        if role == "assistant" and msg.get("tool_calls"):
            # assistant 的工具调用消息：逐个列出工具名与参数（前 60 字）
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {}) or {}
                print(f"  [{i}] [工具调用] {fn.get('name', '?')}({str(fn.get('arguments', ''))[:60]})")
        elif role == "tool":
            print(f"  [{i}] [工具结果] {str(msg.get('content', ''))[:80]}")
        else:
            print(f"  [{i}] {role}: {str(msg.get('content', ''))[:80]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="手写 Agent Runtime CLI（DeepSeek 驱动）")
    parser.add_argument("--list", action="store_true", help="列出全部历史会话后退出")
    parser.add_argument("--resume", metavar="ID", help="恢复指定会话继续对话")
    parser.add_argument("--session-id", metavar="ID", help="以自定义 id 新建会话")
    args = parser.parse_args()

    try:
        config = load_config()
    except RuntimeError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    store = SessionStore(config.sessions_dir)

    # --list：列出会话后退出
    if args.list:
        sessions = store.list_sessions()
        if not sessions:
            print("暂无历史会话")
        else:
            print(f"共 {len(sessions)} 个会话：")
            print_session_table(sessions)
        return

    # 会话选择：--resume 恢复，否则新建
    if args.resume:
        try:
            session = store.load(args.resume)
        except AgentError as e:
            print(f"[错误] {e}")
            sys.exit(1)
        print(f"已恢复会话 {session.session_id}（{session.message_count} 条历史消息）")
    else:
        session = store.create(args.session_id)
        store.save(session)  # 立即落盘，会话文件从创建即存在

    runtime = build_runtime(store, session, config)

    # REPL 主循环
    print("=" * 56)
    print("手写 Agent Runtime | 会话 id:")
    print(f"  {session.session_id}")
    print("输入 /help 查看命令")
    print("=" * 56)

    while True:
        try:
            text = input("\n你> ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break

        text = text.strip()
        if not text:
            continue

        # 斜杠命令
        if text.startswith("/"):
            command = text.split()[0].lower()
            if command == "/help":
                print("可用命令：")
                print("  /new      新建会话（当前会话已落盘，可稍后 --resume 恢复）")
                print("  /exit     退出程序")
                print("  /history  查看当前会话历史消息")
                print("  /sessions 列出全部历史会话")
                print("  /help     显示本帮助")
            elif command == "/exit":
                print("再见")
                break
            elif command == "/new":
                session = store.create()
                store.save(session)
                runtime = build_runtime(store, session, config)
                print(f"已新建会话: {session.session_id}")
            elif command == "/history":
                print_history(session)
            elif command == "/sessions":
                sessions = store.list_sessions()
                if not sessions:
                    print("暂无历史会话")
                else:
                    print(f"共 {len(sessions)} 个会话：")
                    print_session_table(sessions)
            else:
                print(f"未知命令: {command}（输入 /help 查看可用命令）")
            continue

        # 普通输入：交给 runtime 执行；LLMError 等 AgentError 打印后继续循环不退出
        try:
            result = runtime.run(text)
            print(f"\nAgent> {result.answer}")
            print(
                f"[统计] LLM调用 {result.stats['llm_calls']} 次 | "
                f"工具调用 {result.stats['tool_calls']} 次 | "
                f"轮次 {result.stats['iterations']} | "
                f"耗时 {result.stats['elapsed_seconds']}s"
            )
        except AgentError as e:
            print(f"[错误] {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n再见")
