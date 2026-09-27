# -*- coding: utf-8 -*-
"""工具系统基础：Tool 数据类定义与 ToolRegistry 注册/导出/执行机制。"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent.exceptions import AgentError, ToolNotFoundError
from agent.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Tool:
    """可被 LLM 调用的工具定义。

    字段说明：
        name: 工具唯一名称，LLM 通过它选择工具。
        description: 中文功能描述，帮助 LLM 判断何时使用该工具。
        parameters: 参数的 JSON Schema（type/properties/required）。
        func: 实际执行函数，签名统一为 ``func(arguments: dict) -> Any``，
              由注册中心负责调用并序列化结果。
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable[[Dict[str, Any]], Any]

    def to_llm_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling 兼容的工具 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册中心：统一登记工具、导出 schema、执行并序列化结果。

    注册机制：
        工具实例通过 :meth:`register` 登记到内部字典，重名注册直接抛
        ``AgentError``，避免同名的多个实现被静默覆盖导致 LLM 调用到错误实现。

    设计权衡——"错误回传而非中断"：
        :meth:`execute` 对业务异常永不向上抛出，而是把失败原因包装成
        ``"Error: ..."`` 字符串返回。这样 runtime 可以把错误文本当作工具结果
        回传给 LLM，LLM 能够依据错误信息自我修正（补充缺失参数、修正表达式、
        换用其他工具）后再次发起调用，而不是让整轮 Agent 循环因一次工具失败
        而中断。代价是调用方无法用异常机制感知失败，需解析返回字符串判断。
        唯一例外是"工具不存在"（``ToolNotFoundError``），它属于调用方逻辑
        缺陷而非 LLM 可修正的输入问题，因此仍抛异常交由 runtime 单独处理。
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """登记工具；重名抛出 AgentError。"""
        if tool.name in self._tools:
            raise AgentError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """按名称获取工具；不存在抛出 ToolNotFoundError。"""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"工具不存在: {name}")
        return tool

    def export_schemas(self) -> List[Dict[str, Any]]:
        """导出全部工具的 LLM schema，用于注入对话请求。"""
        return [tool.to_llm_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: Optional[dict]) -> str:
        """执行工具并把结果序列化为字符串（永不抛业务异常）。

        步骤：
            1. 未知工具 → 抛 ToolNotFoundError（由 runtime 单独捕获）；
            2. arguments 为 None → 视为空参数 {}；
            3. arguments 不是 dict → 返回错误字符串；
            4. 按 JSON Schema 的 required 校验必填参数，缺失 → 错误字符串；
            5. 执行工具函数，内部异常 → 记 warning 并返回错误字符串；
            6. 结果为 str 直接返回，否则 json.dumps（ensure_ascii=False）。
        """
        # 1. 未知工具属于调用方逻辑错误，仍抛异常由 runtime 单独捕获
        tool = self.get(name)

        # 2. None 视为无参数
        if arguments is None:
            arguments = {}

        # 3. 参数必须是 JSON 对象
        if not isinstance(arguments, dict):
            return "Error: 工具参数必须是 JSON 对象"

        # 4. 必填参数校验
        required: List[str] = tool.parameters.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            return f"Error: 缺少必填参数: {missing}"

        # 5. 执行工具；失败时把错误回传给 LLM 而不是中断 Agent 循环
        try:
            result = tool.func(arguments)
        except Exception as exc:  # noqa: BLE001 —— 故意兜底，错误以字符串回传
            logger.warning("工具 %s 执行失败: %s: %s", name, type(exc).__name__, exc)
            return f"Error: 工具执行失败: {type(exc).__name__}: {exc}"

        # 6. 结果序列化
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)
