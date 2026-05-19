import time
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agent.graph import agent_graph
from app.agent.state import AgentState
from app.config import settings


def _run_query(question: str, has_docs: bool) -> tuple[str, str]:
    state: AgentState = {
        "messages": [],
        "question": question,
        "has_docs": has_docs,
        "route": "",
        "context": "",
        "iteration": 0,
        "final_answer": "",
    }
    result = agent_graph.invoke(state)
    return result["final_answer"], result["context"]


def _evaluate_single(query: str, answer: str, context: str) -> dict:
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
    )

    # Faithfulness: answer grounded in context?
    faith_prompt = ChatPromptTemplate.from_messages([
        ("system", "评分0-1：回答是否完全基于给定上下文？1=完全基于上下文。只输出数字。"),
        ("human", "上下文: {context}\n\n回答: {answer}"),
    ])
    faith_chain = faith_prompt | llm | StrOutputParser()
    try:
        faithfulness = float(faith_chain.invoke({"context": context, "answer": answer}).strip())
    except ValueError:
        faithfulness = 0.0

    # Answer Relevancy
    relev_prompt = ChatPromptTemplate.from_messages([
        ("system", "评分0-1：回答与问题的相关程度。1=高度相关。只输出数字。"),
        ("human", "问题: {query}\n\n回答: {answer}"),
    ])
    relev_chain = relev_prompt | llm | StrOutputParser()
    try:
        answer_relevancy = float(relev_chain.invoke({"query": query, "answer": answer}).strip())
    except ValueError:
        answer_relevancy = 0.0

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
    }


def run_evaluation(queries: List[str], has_docs: bool) -> List[dict]:
    results = []
    for i, query in enumerate(queries):
        try:
            answer, context = _run_query(query, has_docs)
            # Rate-limit friendly: pause 3s between queries
            if i < len(queries) - 1:
                time.sleep(3)
            scores = _evaluate_single(query, answer, context)
            results.append({
                "query": query,
                "answer": answer,
                "context": context,
                "error": None,
                **scores,
            })
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                msg = "Gemini API 免费额度已用完（每分钟 20 次），请稍后重试"
            results.append({
                "query": query,
                "answer": "",
                "context": "",
                "faithfulness": None,
                "answer_relevancy": None,
                "error": msg,
            })
    return results
