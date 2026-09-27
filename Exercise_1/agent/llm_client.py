# -*- coding: utf-8 -*-
"""DeepSeek LLM 客户端：封装 openai SDK v1 的 Chat Completions 调用与重试策略。"""

import time
from typing import List, Optional

from openai import APITimeoutError, APIConnectionError, OpenAI, RateLimitError

from agent.config import Config
from agent.exceptions import LLMError
from agent.logger import get_logger

logger = get_logger(__name__)

# 仅对这些"暂时性"错误重试：超时、网络连接失败、限流
RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, RateLimitError)


class LLMClient:
    """DeepSeek 聊天模型客户端，负责单次 LLM 调用与可重试错误的退避重试。"""

    def __init__(self, config: Config):
        """保存配置并初始化 openai SDK 客户端。"""
        self.config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.llm_timeout,
        )

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> object:
        """调用 Chat Completions，成功时返回 resp.choices[0].message。

        重试策略说明：
        chat 是幂等的只读调用——同样的 messages/tools 再发一次不会产生副作用，
        且超时 / 连接失败 / 429 限流都是"下一刻可能成功"的暂时性故障，
        因此用 2**attempt 秒的指数退避安全重试。

        反之，鉴权失败（401）、参数非法（400）、模型不存在等属于确定性错误，
        重试只会原样失败、白白拖慢失败路径，故立即包装为 LLMError 快速失败。
        """
        kwargs = {"model": self.config.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools  # 不设置 tool_choice，使用默认 auto

        last_exc: Exception | None = None
        for attempt in range(self.config.llm_max_retries + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message
            except RETRYABLE_ERRORS as e:
                last_exc = e
                if attempt == self.config.llm_max_retries:
                    raise LLMError(f"LLM 调用失败（已重试 {attempt} 次）: {e}") from e
                wait = 2**attempt
                logger.warning(
                    "LLM 暂时性错误（第 %d 次），%.0f 秒后重试: %s",
                    attempt + 1,
                    wait,
                    e,
                )
                time.sleep(wait)
            except Exception as e:
                # 非暂时性错误（鉴权/参数/模型等）：快速失败，不重试
                raise LLMError(f"LLM 调用失败: {e}") from e

        # 理论上不可达（最后一次可重试错误已在循环内抛出），兜底防御
        raise LLMError(f"LLM 调用失败（已重试 {self.config.llm_max_retries} 次）: {last_exc}")
