# -*- coding: utf-8 -*-
"""工具系统测试：注册机制、schema 导出、执行与错误回传，以及三个内置工具。"""

import pytest

from agent.exceptions import AgentError, ToolNotFoundError
from agent.tools.base import Tool, ToolRegistry
from agent.tools.calculator import create_calculator_tool, _evaluate_expression
from agent.tools.search import create_search_tool
from agent.tools.weather import create_weather_tool


def _make_tool(name="dummy", required=None):
    return Tool(
        name=name,
        description="测试工具",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": required or [],
        },
        func=lambda args: {"echo": args},
    )


class TestRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = _make_tool()
        reg.register(tool)
        assert reg.get("dummy") is tool

    def test_duplicate_register_raises(self):
        reg = ToolRegistry()
        reg.register(_make_tool())
        with pytest.raises(AgentError, match="重复注册"):
            reg.register(_make_tool())

    def test_get_unknown_raises_tool_not_found(self):
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            reg.get("nope")

    def test_export_schemas_openai_format(self):
        reg = ToolRegistry()
        reg.register(_make_tool())
        schemas = reg.export_schemas()
        assert schemas == [
            {
                "type": "function",
                "function": {
                    "name": "dummy",
                    "description": "测试工具",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "required": [],
                    },
                },
            }
        ]

    def test_execute_serializes_dict_result(self):
        reg = ToolRegistry()
        reg.register(_make_tool())
        out = reg.execute("dummy", {"x": 3})
        assert '"echo"' in out and '"x": 3' in out

    def test_execute_missing_required_returns_error_string(self):
        reg = ToolRegistry()
        reg.register(_make_tool(required=["x"]))
        out = reg.execute("dummy", {})
        assert out.startswith("Error")
        assert "x" in out

    def test_execute_non_dict_arguments_returns_error_string(self):
        reg = ToolRegistry()
        reg.register(_make_tool())
        assert reg.execute("dummy", "oops").startswith("Error")

    def test_execute_tool_exception_returns_error_string(self):
        def boom(_args):
            raise ZeroDivisionError("炸了")

        reg = ToolRegistry()
        reg.register(Tool(name="boom", description="", parameters={"type": "object", "properties": {}, "required": []}, func=boom))
        out = reg.execute("boom", {})
        assert out.startswith("Error")
        assert "ZeroDivisionError" in out

    def test_execute_none_arguments_treated_as_empty(self):
        reg = ToolRegistry()
        reg.register(_make_tool())
        out = reg.execute("dummy", None)
        assert "Error" not in out


class TestCalculator:
    def test_basic_arithmetic(self):
        assert _evaluate_expression("(3+5)*2") == 16
        assert _evaluate_expression("2**10") == 1024
        assert _evaluate_expression("10/4") == 2.5

    def test_whitelisted_functions_and_constants(self):
        assert _evaluate_expression("sqrt(16)") == 4.0
        assert _evaluate_expression("round(pi, 2)") == 3.14
        assert _evaluate_expression("max(1, 5, 3)") == 5

    def test_empty_expression_raises(self):
        with pytest.raises(ValueError):
            _evaluate_expression("   ")

    def test_division_by_zero_propagates(self):
        with pytest.raises(ZeroDivisionError):
            _evaluate_expression("1/0")

    @pytest.mark.parametrize(
        "bad",
        [
            "__import__('os').system('ls')",  # 函数调用不在白名单
            "().__class__",  # 属性访问语法被拒绝
            "'a'+'b'",  # 字符串字面量不允许
            "[1,2]",  # 非法语法（列表）
        ],
    )
    def test_dangerous_syntax_rejected(self, bad):
        with pytest.raises((ValueError, SyntaxError, TypeError)):
            _evaluate_expression(bad)

    def test_via_registry_error_string(self):
        reg = ToolRegistry()
        reg.register(create_calculator_tool())
        out = reg.execute("calculator", {"expression": "__import__('os')"}) 
        assert out.startswith("Error")


class TestSearch:
    def test_returns_relevant_results(self):
        reg = ToolRegistry()
        reg.register(create_search_tool())
        out = reg.execute("search", {"query": "什么是 RAG 检索增强"})
        assert "results" in out
        assert "RAG" in out

    def test_no_match_returns_empty_with_note(self):
        reg = ToolRegistry()
        reg.register(create_search_tool())
        out = reg.execute("search", {"query": "zzzz"})
        assert "未找到相关内容" in out

    def test_custom_knowledge_base(self):
        reg = ToolRegistry()
        reg.register(create_search_tool(knowledge_base={"独有词条": "测试内容"}))
        out = reg.execute("search", {"query": "独有词条"})
        assert "独有词条" in out


class TestWeather:
    def test_builtin_city_fixed_data(self):
        reg = ToolRegistry()
        reg.register(create_weather_tool())
        out1 = reg.execute("weather", {"city": "北京"})
        out2 = reg.execute("weather", {"city": "北京"})
        assert "晴" in out1 and '"source": "mock"' in out1
        assert out1 == out2  # 内置城市数据固定

    def test_unknown_city_deterministic(self):
        reg = ToolRegistry()
        reg.register(create_weather_tool())
        out1 = reg.execute("weather", {"city": "亚特兰蒂斯"})
        out2 = reg.execute("weather", {"city": "亚特兰蒂斯"})
        assert out1 == out2  # 同一城市结果确定

    def test_missing_required_city(self):
        reg = ToolRegistry()
        reg.register(create_weather_tool())
        assert reg.execute("weather", {}).startswith("Error")
