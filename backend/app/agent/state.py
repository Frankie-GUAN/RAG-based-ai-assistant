from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    has_docs: bool
    route: str
    context: str
    iteration: int
    final_answer: str
    summary: str
