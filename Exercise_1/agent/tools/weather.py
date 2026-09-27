# -*- coding: utf-8 -*-
"""mock 天气工具：内置城市返回固定数据，未知城市返回确定性伪随机数据。"""

import random
from typing import Any, Dict

from agent.tools.base import Tool

# 内置城市的固定天气数据（mock）
_CITY_WEATHER: Dict[str, Dict[str, Any]] = {
    "北京": {"temperature_c": 12, "condition": "晴", "humidity_percent": 32, "wind": "西北风 3 级"},
    "上海": {"temperature_c": 18, "condition": "多云", "humidity_percent": 65, "wind": "东风 2 级"},
    "广州": {"temperature_c": 26, "condition": "阵雨", "humidity_percent": 82, "wind": "南风 2 级"},
    "深圳": {"temperature_c": 27, "condition": "多云", "humidity_percent": 78, "wind": "微风"},
    "杭州": {"temperature_c": 20, "condition": "晴转多云", "humidity_percent": 60, "wind": "东南风 3 级"},
    "成都": {"temperature_c": 17, "condition": "阴", "humidity_percent": 75, "wind": "微风"},
    "西安": {"temperature_c": 15, "condition": "晴", "humidity_percent": 45, "wind": "东北风 2 级"},
    "武汉": {"temperature_c": 22, "condition": "多云转晴", "humidity_percent": 68, "wind": "北风 2 级"},
}

# 未知城市生成伪随机数据时的候选项
_CONDITIONS = ["晴", "多云", "阴", "小雨", "阵雨", "雷阵雨", "雾"]
_WINDS = ["微风", "北风 2 级", "南风 3 级", "东风 2 级", "西风 3 级"]


def _lookup_weather(city: str, date: str) -> Dict[str, Any]:
    """查询城市天气：内置城市走固定数据，未知城市用确定性伪随机生成。"""
    city = city.strip()
    if city in _CITY_WEATHER:
        weather = dict(_CITY_WEATHER[city])
    else:
        # 以城市名为种子，同一城市在同一进程内结果确定
        rng = random.Random(hash(city) & 0xFFFF)
        weather = {
            "temperature_c": rng.randint(-5, 35),
            "condition": rng.choice(_CONDITIONS),
            "humidity_percent": rng.randint(20, 95),
            "wind": rng.choice(_WINDS),
        }
    return {"city": city, "date": date, "weather": weather, "source": "mock"}


def create_weather_tool() -> Tool:
    """创建 mock 天气工具实例。"""

    def _run(arguments: Dict[str, Any]) -> Dict[str, Any]:
        city = arguments["city"]
        date = arguments.get("date") or "今天"
        return _lookup_weather(city, date)

    return Tool(
        name="weather",
        description="查询指定城市的天气（mock 数据源）。",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如 北京"},
                "date": {"type": "string", "description": "日期描述，默认今天，如 明天"},
            },
            "required": ["city"],
        },
        func=_run,
    )
