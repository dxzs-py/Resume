# -*- coding: utf-8 -*-
"""Session 管理测试：多会话隔离、持久化、恢复、损坏容错、列表。"""

import pytest

from agent.exceptions import AgentError
from agent.session.store import Session, SessionStore


class TestSession:
    def test_add_message_keeps_openai_format(self):
        s = Session(session_id="s1")
        msg = s.add_message("user", content="你好")
        assert msg == {"role": "user", "content": "你好"}
        assert s.message_count == 1

    def test_tool_message_fields(self):
        s = Session(session_id="s1")
        msg = s.add_message("tool", content="结果", tool_call_id="c1", name="weather")
        assert msg["tool_call_id"] == "c1"
        assert msg["name"] == "weather"

    def test_title_from_first_user_message(self):
        s = Session(session_id="s1")
        s.add_message("user", content="帮我查一下北京的天气情况如何")
        assert s.title == "帮我查一下北京的天气情况如何"[:20]
        s.add_message("user", content="第二条")
        assert "第二条" not in s.title  # 标题只取第一条


class TestSessionStore:
    def test_create_has_unique_id(self, tmp_path):
        store = SessionStore(str(tmp_path))
        s1, s2 = store.create(), store.create()
        assert s1.session_id != s2.session_id

    def test_save_load_roundtrip(self, tmp_path):
        store = SessionStore(str(tmp_path))
        s = store.create(session_id="abc")
        s.add_message("user", content="问题")
        s.add_message("assistant", content="回答")
        store.save(s)

        loaded = store.load("abc")
        assert loaded.session_id == "abc"
        assert loaded.message_count == 2
        assert loaded.messages[0]["content"] == "问题"

    def test_load_missing_raises(self, tmp_path):
        store = SessionStore(str(tmp_path))
        with pytest.raises(AgentError, match="不存在"):
            store.load("ghost")

    def test_load_corrupted_raises(self, tmp_path):
        store = SessionStore(str(tmp_path))
        (store.sessions_dir / "bad.json").write_text("{不是json", encoding="utf-8")
        with pytest.raises(AgentError, match="损坏"):
            store.load("bad")

    def test_sessions_isolated_from_each_other(self, tmp_path):
        """两个会话（模拟两个窗口）互不影响。"""
        store = SessionStore(str(tmp_path))
        w1 = store.create(session_id="window1")
        w2 = store.create(session_id="window2")

        w1.add_message("user", content="查天气记待办")
        store.save(w1)
        w2.add_message("user", content="写周报记待办")
        store.save(w2)

        # 重新加载 window1，只含自己的历史
        r1 = store.load("window1")
        assert r1.message_count == 1
        assert "查天气" in r1.messages[0]["content"]
        assert not any("周报" in str(m) for m in r1.messages)

        r2 = store.load("window2")
        assert r2.message_count == 1
        assert "周报" in r2.messages[0]["content"]

    def test_list_sessions_sorted_and_skips_corrupted(self, tmp_path):
        store = SessionStore(str(tmp_path))
        s1 = store.create(session_id="old")
        s1.updated_at = "2026-01-01T00:00:00"
        s1.add_message("user", content="第一条")
        store.save(s1)
        s2 = store.create(session_id="new")
        s2.updated_at = "2026-09-01T00:00:00"
        s2.add_message("user", content="第二条")
        store.save(s2)
        (store.sessions_dir / "corrupt.json").write_text("{{{", encoding="utf-8")

        rows = store.list_sessions()
        ids = [r["id"] for r in rows]
        assert ids == ["new", "old"]  # 按 updated_at 倒序，损坏文件被跳过
