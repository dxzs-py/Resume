# -*- coding: utf-8 -*-
"""异常体系：Agent 全部自定义异常的基类与具体异常子类。"""


class AgentError(Exception):
    """Agent 所有自定义异常的基类。"""


class LLMError(AgentError):
    """LLM 调用失败（重试耗尽等）。"""


class ToolNotFoundError(AgentError):
    """工具不存在。"""


class ToolExecutionError(AgentError):
    """工具执行失败。"""


class MaxIterationsError(AgentError):
    """超出迭代上限。"""
