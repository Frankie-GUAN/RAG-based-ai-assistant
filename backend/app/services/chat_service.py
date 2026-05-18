from langchain_core.messages import HumanMessage, AIMessage
from app.agent.graph import agent_graph
from app.agent.state import AgentState


def run_agent(question: str, history: list[dict], has_docs: bool) -> dict:
    lc_messages = []
    for m in history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))
    lc_messages.append(HumanMessage(content=question))

    initial_state: AgentState = {
        "messages": lc_messages,
        "question": question,
        "has_docs": has_docs,
        "route": "",
        "context": "",
        "iteration": 0,
        "final_answer": "",
    }
    result = agent_graph.invoke(initial_state)
    return {
        "answer": result["final_answer"],
        "route": result.get("route", "direct"),
        "context": result.get("context", ""),
    }
