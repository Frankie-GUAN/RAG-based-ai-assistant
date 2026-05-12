from __future__ import annotations

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import GEMINI_API_KEY_ENV, LLM_MODEL


def _require_gemini_key() -> str:
    # 仅从环境变量读取 Gemini 密钥
    api_key = os.getenv(GEMINI_API_KEY_ENV)
    if not api_key:
        raise ValueError("未检测到 GEMINI_API_KEY 环境变量")
    return api_key


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    # 初始化 Gemini 对话模型
    api_key = _require_gemini_key()
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=temperature,
    )


def route_question(llm: ChatGoogleGenerativeAI, question: str, history: str, has_docs: bool) -> str:
    # 使用 LLM 进行智能路由分类
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是路由分类器，只能输出 web / rag / direct 之一。"
                "若问题涉及最新动态、时效性、实时数据、2025-2026 新内容，输出 web。"
                "若问题明显依赖已上传私有文档，且 has_docs=是，输出 rag。"
                "其余常识或无需外部资料的问题输出 direct。",
            ),
            (
                "human",
                "对话历史:\n{history}\n\n是否有私有文档: {has_docs}\n\n用户问题: {question}\n\n"
                "仅输出: web / rag / direct",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "question": question,
            "history": history or "(无)",
            "has_docs": "是" if has_docs else "否",
        }
    )
    decision = raw.strip().lower()
    if "web" in decision:
        return "web"
    if "rag" in decision:
        return "rag"
    return "direct"


def answer_direct(llm: ChatGoogleGenerativeAI, question: str, history: str) -> str:
    # 直接由大模型回答
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是专业中文助手，基于对话历史直接回答问题。若不知道，说明不确定。",
            ),
            (
                "human",
                "对话历史:\n{history}\n\n用户问题: {question}\n\n请用简洁中文回答。",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "history": history or "(无)"})


def answer_with_rag(
    llm: ChatGoogleGenerativeAI,
    question: str,
    history: str,
    context: str,
) -> str:
    # 基于私有文档上下文回答
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是中文问答助手，只能基于提供的资料回答，禁止编造。",
            ),
            (
                "human",
                "对话历史:\n{history}\n\n已知资料:\n{context}\n\n"
                "用户问题: {question}\n\n请用简洁中文回答。",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "question": question,
            "history": history or "(无)",
            "context": context or "(无相关资料)",
        }
    )


def answer_with_web(
    llm: ChatGoogleGenerativeAI,
    question: str,
    history: str,
    web_context: str,
) -> str:
    # 基于联网检索结果回答
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是中文问答助手，需要结合检索结果回答，避免臆测。",
            ),
            (
                "human",
                "对话历史:\n{history}\n\n联网检索摘要:\n{web_context}\n\n"
                "用户问题: {question}\n\n请用简洁中文回答。",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "question": question,
            "history": history or "(无)",
            "web_context": web_context or "(无检索结果)",
        }
    )
