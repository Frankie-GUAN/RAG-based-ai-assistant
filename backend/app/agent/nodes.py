import os
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agent.state import AgentState
from app.config import settings
from app.tools.registry import tool_registry


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )


def router_node(state: AgentState) -> dict:
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是路由分类器。根据用户问题判断应该走哪条路径。"
         "输出: web (需要联网搜索最新信息) / rag (需要检索私有文档) / direct (直接回答)。"
         "只输出这三个词之一。"),
        ("human", "问题: {question}\n是否有私有文档: {has_docs}"),
    ])
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"question": state["question"], "has_docs": "是" if state.get("has_docs") else "否"})
    decision = raw.strip().lower()

    if "web" in decision:
        route = "web"
    elif "rag" in decision and state.get("has_docs"):
        route = "rag"
    else:
        route = "direct"

    return {"route": route, "iteration": state.get("iteration", 0) + 1}


def retrieve_node(state: AgentState) -> dict:
    result = tool_registry.execute("search_documents", query=state["question"], top_k=4)
    return {"context": result}


def web_search_node(state: AgentState) -> dict:
    result = tool_registry.execute("search_web", query=state["question"])
    return {"context": result}


def generate_node(state: AgentState) -> dict:
    llm = _get_llm()
    context = state.get("context", "")
    history = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in state["messages"][-6:]
    )

    if context:
        system_text = (
            "你是专业中文助手，基于提供的参考资料回答问题。若资料不足，说明不确定。"
            f"\n\n参考资料:\n{context}"
        )
    else:
        system_text = "你是专业中文助手，基于自身知识回答问题。若不知道，说明不确定。"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "对话历史:\n{history}\n\n用户问题: {question}\n\n请用简洁中文回答。"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"history": history, "question": state["question"]})
    return {"final_answer": answer}
