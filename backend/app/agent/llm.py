"""LLM 工厂。

生产环境返回 DeepSeek 的 ChatOpenAI；测试通过 set_chat_model_factory
注入脚本化模型，使 Agent 循环测试不需要 API key 与网络。
"""
from __future__ import annotations

from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import settings

_override: Callable[[], BaseChatModel] | None = None


def set_chat_model_factory(factory: Callable[[], BaseChatModel] | None) -> None:
    """测试专用。传 None 恢复默认 DeepSeek。"""
    global _override
    _override = factory


def get_chat_model() -> BaseChatModel:
    if _override is not None:
        return _override()

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.2,
    )
