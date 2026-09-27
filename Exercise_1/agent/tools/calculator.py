# -*- coding: utf-8 -*-
"""安全计算器工具：基于 ast 白名单的数学表达式求值，严禁 eval/exec。"""

import ast
import math
import operator
from typing import Any, Dict

from agent.tools.base import Tool

# 允许的二元运算符
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# 允许的一元运算符
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 允许调用的函数白名单
_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

# 允许引用的常量白名单
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> Any:
    """递归求值 AST 节点，只允许白名单内的语法、运算符、函数与常量。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        # 只接受数字字面量；bool 是 int 子类，用 type 精确排除 True/False/字符串等
        if type(node.value) not in (int, float):
            raise ValueError(f"不允许的常量: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不允许的运算符: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不允许的运算符: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))

    if isinstance(node, ast.Call):
        # 函数名必须在白名单内（同时拒绝属性链调用等非法形式）
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            name = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
            raise ValueError(f"不允许调用的函数: {name}")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        args = [_safe_eval(arg) for arg in node.args]
        return _FUNCS[node.func.id](*args)

    if isinstance(node, ast.Name):
        if node.id not in _CONSTANTS:
            raise ValueError(f"未知的名称: {node.id}")
        return _CONSTANTS[node.id]

    raise ValueError(f"不允许的语法: {type(node).__name__}")


def _evaluate_expression(expression: str) -> Any:
    """解析并求值数学表达式（ast 白名单，不使用 eval/exec）。

    表达式为空或含非法语法时抛出 SyntaxError/ValueError/ZeroDivisionError 等，
    由 ToolRegistry 统一转换为 "Error: ..." 字符串回传 LLM。
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("表达式为空")
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree)


def create_calculator_tool() -> Tool:
    """创建安全计算器工具实例。"""

    def _run(arguments: Dict[str, Any]) -> Any:
        return _evaluate_expression(arguments["expression"])

    return Tool(
        name="calculator",
        description="精确计算数学表达式。任何算术运算都必须使用此工具，避免心算误差。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 (3+5)*2、sqrt(2)**10、round(pi, 2)",
                }
            },
            "required": ["expression"],
        },
        func=_run,
    )
