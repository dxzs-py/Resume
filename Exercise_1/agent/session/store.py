# -*- coding: utf-8 -*-
"""会话存储：Session 数据模型与"每个会话一个 JSON 文件"的持久化存储。"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.exceptions import AgentError
from agent.logger import get_logger

logger = get_logger(__name__)


class Session:
    """一次对话会话；messages 采用 OpenAI messages 完全兼容的格式。

    多窗口 = 多个 Session 实例：每个实例持有独立 session_id 与独立消息
    历史，互不共享状态，从而实现多窗口会话隔离。
    """

    def __init__(
        self,
        session_id: str,
        title: str = "",
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.session_id = session_id
        self.title = title
        self.created_at = created_at or self._now()
        self.updated_at = updated_at or self._now()
        self.messages: List[Dict[str, Any]] = messages if messages is not None else []

    @staticmethod
    def _now() -> str:
        """当前本地时间，ISO 格式、精确到秒。"""
        return datetime.now().isoformat(timespec="seconds")

    def add_message(
        self,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[List[dict]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """追加一条 OpenAI 格式消息；只保留非 None 字段，并刷新 updated_at。

        若是第一条 user 消息且标题为空，则取其前 20 字符作为会话标题。
        """
        message: Dict[str, Any] = {"role": role}
        if content is not None:
            message["content"] = content
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        if tool_call_id is not None:
            message["tool_call_id"] = tool_call_id
        if name is not None:
            message["name"] = name
        self.messages.append(message)
        self.updated_at = self._now()
        if role == "user" and not self.title and content:
            self.title = content[:20]
        return message

    @property
    def message_count(self) -> int:
        """当前消息条数。"""
        return len(self.messages)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可直接 JSON 化的 dict。"""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """从 dict 恢复 Session；字段容错，缺字段用默认值。"""
        return cls(
            session_id=data.get("session_id", ""),
            title=data.get("title", ""),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            messages=data.get("messages") or [],
        )


class SessionStore:
    """会话持久化存储：内存即文件镜像，save 落盘、load 恢复。

    为什么采用"每个 session 一个 JSON 文件"：
    1. 多窗口隔离：不同 session_id 各占一个文件，多个窗口（多个会话）
       读写互不干扰，单个文件损坏也不会波及其他会话；
    2. 恢复简单：重启后按 session_id 直接读取对应文件即可完整还原会话，
       无需索引、无需全量扫描；
    3. 无需数据库：纯标准库实现、零部署依赖，JSON 人类可读，
       便于调试与人工排查；对单机会话场景足够可靠。
    """

    def __init__(self, sessions_dir: str = "sessions") -> None:
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.sessions_dir / f"{session_id}.json"

    def create(self, session_id: Optional[str] = None) -> Session:
        """创建新会话；id 缺省为 时间戳_短uuid，天然唯一且可读。"""
        if session_id is None:
            session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        return Session(session_id=session_id)

    def save(self, session: Session) -> None:
        """将会话完整落盘到 <dir>/<session_id>.json（UTF-8）。"""
        path = self._path(session.session_id)
        path.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, session_id: str) -> Session:
        """按 session_id 从磁盘恢复会话。

        文件不存在或 JSON 损坏均抛出 AgentError，由上层决定如何处理。
        """
        path = self._path(session_id)
        if not path.exists():
            raise AgentError(f"会话不存在: {session_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise AgentError(f"会话文件损坏: {session_id}")
        return Session.from_dict(data)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出全部会话摘要，按 updated_at 倒序；损坏文件跳过并告警。"""
        result: List[Dict[str, Any]] = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                result.append(
                    {
                        "id": data.get("session_id", path.stem),
                        "title": data.get("title", ""),
                        "updated_at": data.get("updated_at", ""),
                        "message_count": len(data.get("messages") or []),
                    }
                )
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                logger.warning("跳过损坏的会话文件 %s: %s", path.name, exc)
        result.sort(key=lambda item: item["updated_at"], reverse=True)
        return result
