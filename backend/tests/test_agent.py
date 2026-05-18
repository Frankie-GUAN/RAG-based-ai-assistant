from langchain_core.messages import HumanMessage
from app.agent.graph import agent_graph
from app.agent.state import AgentState


def test_agent_direct_route():
    state: AgentState = {
        "messages": [HumanMessage(content="Python中如何读取文件？")],
        "question": "Python中如何读取文件？",
        "has_docs": False,
        "route": "",
        "context": "",
        "iteration": 0,
        "final_answer": "",
    }
    result = agent_graph.invoke(state)
    assert "final_answer" in result
    assert result["final_answer"] != ""
    assert isinstance(result["final_answer"], str)
