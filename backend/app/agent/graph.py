from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.agent.nodes import router_node, retrieve_node, web_search_node, generate_node


def _route_decision(state: AgentState) -> str:
    route = state.get("route", "direct")
    if route == "web":
        return "web_search"
    elif route == "rag":
        return "retrieve"
    return "generate"


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", _route_decision, {
        "web_search": "web_search",
        "retrieve": "retrieve",
        "generate": "generate",
    })
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


agent_graph = build_graph()
