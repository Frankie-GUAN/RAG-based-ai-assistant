"""把六个节点组装成 StateGraph。

**这张图就是「LangGraph Agent」这个宣称的证据。** 迁移前，
`requirements.txt` 钉了 `langgraph==1.1.10`、CLAUDE.md 的架构段写着
「LangGraph Agent」，但代码里只有一行 `from langgraph.graph.message import
add_messages`——那个 `AgentState` 零引用，真正的 Agent 是 `core.py` 里的
`for` 循环。现在 `graph.build_graph()` 是唯一入口，
`graph.get_graph().draw_mermaid()` 能直接导出这张图。

**为什么每次请求都重新构图。** 节点是 `AgentNodes` 实例的方法，
而实例持有请求级状态（`_seen_doc_ids` 去重集合）。若全局编译一次图、
节点绑定到共享实例，并发请求之间就会互相污染引用去重状态——
A 请求见过的 doc_id 会让 B 请求漏掉同一片段。

构图 + compile 的开销在毫秒级（实测 < 5ms），
远低于一次 LLM 往返，不值得为它引入并发风险。

## 图结构

```
START
  │
  ▼
plan ──(need_retrieval)──▶ retrieve ──▶ agent ──(有 tool_calls)──▶ tools
  │                                        ▲                        │
  └──(不需要)──────────────────────────────┘◀───────────────────────┘
                                           │
                                    (无 tool_calls)
                                           ▼
                                        reflect ──(资料不足)──▶ retrieve（最多一次）
                                           │
                                    (资料充分)
                                           ▼
                                         answer ──▶ END
```

## 怎么自己看图

```python
from app.agent.graph import build_graph
print(build_graph(deps).get_graph().draw_mermaid())
```
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    AgentDeps,
    AgentNodes,
    after_agent,
    after_plan,
    after_reflect,
)
from app.agent.state import AgentState


def build_graph(deps: AgentDeps):
    """构建并编译 Agent 图。

    返回的对象支持：
    - `astream(initial_state, stream_mode="custom")` —— 逐事件消费（生产用）
    - `ainvoke(initial_state)` —— 拿最终状态（测试与排查用）
    - `get_graph()` —— 导出结构与 Mermaid 图
    """
    nodes = AgentNodes(deps)
    graph = StateGraph(AgentState)

    graph.add_node("plan", nodes.plan)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("agent", nodes.agent)
    graph.add_node("tools", nodes.tools)
    graph.add_node("reflect", nodes.reflect)
    graph.add_node("answer", nodes.answer)

    graph.add_edge(START, "plan")

    # 规划 → 检索 or 直接推理
    graph.add_conditional_edges(
        "plan",
        after_plan,
        {"retrieve": "retrieve", "agent": "agent"},
    )

    # 预检索完成后进入推理
    graph.add_edge("retrieve", "agent")

    # 推理 → 执行工具（循环） or 反思
    # max_iterations 通过闭包注入：条件边签名固定为 (state) -> str
    graph.add_conditional_edges(
        "agent",
        lambda state: after_agent(state, deps.max_iterations),
        {"tools": "tools", "reflect": "reflect", "answer": "answer"},
    )

    # 工具执行完回到推理（这是 ReAct 循环的核心回边）
    graph.add_edge("tools", "agent")

    # 反思 → 补检索 or 作答
    graph.add_conditional_edges(
        "reflect",
        after_reflect,
        {"retrieve": "retrieve", "answer": "answer"},
    )

    graph.add_edge("answer", END)

    return graph.compile()


def mermaid(deps: AgentDeps) -> str:
    """导出 Mermaid 图，用于文档与面试演示。"""
    return build_graph(deps).get_graph().draw_mermaid()
