# -*- coding: utf-8 -*-
"""mock 搜索工具：基于字符 bigram 重叠度的内部知识库检索（无第三方依赖）。"""

from typing import Any, Dict, List, Optional, Set

from agent.tools.base import Tool

# 默认知识库：标题 -> 正文（mock 数据源）
_DEFAULT_KNOWLEDGE_BASE: Dict[str, str] = {
    "LangChain": (
        "LangChain 是面向大模型应用开发的开源框架，提供模型调用、提示词模板、"
        "链式编排、工具集成与记忆管理等组件，支持把 LLM 与外部数据和 API 结合，"
        "快速构建检索问答、智能体等应用。"
    ),
    "Agent（智能体）": (
        "Agent 是能自主感知任务、规划步骤并调用工具完成目标的智能体系统。"
        "它以 LLM 为核心推理引擎，通过循环执行思考、行动、观察来分步解决问题，"
        "并维护上下文记忆以支持多轮交互。"
    ),
    "RAG": (
        "RAG 即检索增强生成：先从外部知识库检索与问题相关的文档片段，"
        "再将其注入提示词交给大模型生成答案，可缓解模型幻觉、补充私有知识"
        "与时效信息，是企业知识问答的主流架构。"
    ),
    "Function Calling": (
        "Function Calling 允许大模型根据用户意图输出结构化的函数调用请求，"
        "包括函数名与 JSON 参数，由应用侧执行真实函数并把结果回传模型继续推理，"
        "是 LLM 连接外部工具与 API 的标准机制。"
    ),
    "DeepSeek": (
        "DeepSeek 是国产大模型系列，以高性价比和强推理能力著称，"
        "其开源的 V3、R1 等模型支持长上下文与工具调用，API 兼容 OpenAI 格式，"
        "广泛用于对话、代码生成与智能体场景。"
    ),
    "ReAct 模式": (
        "ReAct 将推理与行动结合：模型先输出思考过程，再选择工具执行行动，"
        "然后观察结果并循环往复，直到得出最终答案。该模式让 LLM 的决策过程"
        "可解释、可纠错，是构建工具调用型智能体的经典范式。"
    ),
    "KV Cache": (
        "KV Cache 是 Transformer 推理优化技术：自回归生成时缓存已计算的 "
        "Key、Value 张量，避免每生成一个 token 都重算全部历史的注意力投影，"
        "大幅降低计算量，代价是显存占用随上下文长度线性增长。"
    ),
    "滑动窗口限流": (
        "滑动窗口限流以时间为轴统计最近 N 秒内的请求数并拒绝超额请求，"
        "相比固定窗口计数能消除窗口边界的流量突刺，常用 Redis 的 ZSet "
        "或计数数组实现，是接口限流的常见方案。"
    ),
}


def _bigrams(text: str) -> Set[str]:
    """提取字符 bigram 集合（相邻两字符组成一个元素）。"""
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _search(knowledge_base: Dict[str, str], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """对知识库按 bigram 交集大小打分，返回得分最高的至多 2 条结果。"""
    query = arguments["query"]
    query_grams = _bigrams(query)

    scored: List[Dict[str, Any]] = []
    for title, content in knowledge_base.items():
        doc_grams = _bigrams(title + content)
        score = len(query_grams & doc_grams)
        if score > 0:
            scored.append({"title": title, "content": content, "score": score})

    # 按得分降序、标题升序排列，保证结果确定性
    scored.sort(key=lambda item: (-item["score"], item["title"]))
    top = scored[:2]

    if not top:
        return {"query": query, "results": [], "note": "知识库中未找到相关内容"}
    return {"query": query, "results": top}


def create_search_tool(knowledge_base: Optional[Dict[str, str]] = None) -> Tool:
    """创建 mock 搜索工具实例；可传入自定义知识库替换默认数据。"""
    kb = knowledge_base if knowledge_base is not None else _DEFAULT_KNOWLEDGE_BASE

    def _run(arguments: Dict[str, Any]) -> Dict[str, Any]:
        return _search(kb, arguments)

    return Tool(
        name="search",
        description="搜索内部知识库（mock 数据源），返回与查询最相关的资料条目。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词或问题"}
            },
            "required": ["query"],
        },
        func=_run,
    )
