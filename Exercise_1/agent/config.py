# -*- coding: utf-8 -*-
"""配置模块：加载项目根 .env，提供全局 Config 数据类。"""

import logging
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

def _load_dotenv(path: Path) -> None:
    """.env 加载：调用 python-dotenv，已存在环境变量优先"""
    if not path.is_file():
        return
    try:
        load_dotenv(dotenv_path=path, override=False)
    except OSError:
        logging.getLogger(__name__).warning("读取 .env 失败: %s", path)

@dataclass
class Config:
    """Agent 全局配置。"""

    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    max_iterations: int = 8
    context_max_messages: int = 40
    context_keep_recent: int = 12
    llm_timeout: float = 60.0
    llm_max_retries: int = 3
    trace_file: str = "logs/trace.jsonl"
    sessions_dir: str = "sessions"


def _get_int(name: str, default: int) -> int:
    """从环境变量读取 int，缺失或非法值回退默认。"""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    """从环境变量读取 float，缺失或非法值回退默认。"""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_config() -> Config:
    """定位项目根目录 .env 并加载，然后从环境变量构造 Config。"""
    project_root = Path(__file__).resolve().parent.parent
    _load_dotenv(project_root / ".env")

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "缺少 DEEPSEEK_API_KEY：请在项目根目录 .env 中配置"
            "（可参考 .env.example），或通过环境变量注入。"
        )

    return Config(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        max_iterations=_get_int("AGENT_MAX_ITERATIONS", 8),
        context_max_messages=_get_int("CONTEXT_MAX_MESSAGES", 40),
        context_keep_recent=_get_int("CONTEXT_KEEP_RECENT", 12),
        llm_timeout=_get_float("LLM_TIMEOUT", 60.0),
        llm_max_retries=_get_int("LLM_MAX_RETRIES", 3),
        trace_file=os.environ.get("trace_file", "logs/trace.jsonl"),
        sessions_dir=os.environ.get("SESSIONS_DIR", "sessions"),
    )
