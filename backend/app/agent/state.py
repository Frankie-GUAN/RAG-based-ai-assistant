"""Agent 图的状态定义。

**这个文件曾经是「虚假宣称」的证据。** 旧版本定义了 8 个字段：
`messages / question / has_docs / route / context / iteration / final_answer / summary`，
但全仓库**零引用**——只有 `from langgraph.graph.message import add_messages` 这一行
被 import 过，`AgentState` 类本身从未被任何代码使用。真正的 Agent 是
`core.py` 里的一个 `for` 循环，状态存在局部变量里。

现在它是真的：`graph.py` 用这个 TypedDict 构建 StateGraph，每个节点
读它、返回增量写它。

**为什么删掉那几个字段**：

| 旧字段 | 为什么删 |
|---|---|
| `has_docs` | 从没有任何代码写入或读取过。检索是否有据由 `citations` 判断 |
| `route` | 同上。路由决策现在由**条件边**（`after_plan` / `after_reflect`）表达，不需要在状态里再存一份字符串——存了就会出现「状态说走 A、实际走了 B」的双真相 |
| `context` | `ContextEngine.build()` 产出的 messages 已经在 `messages` 里，再存一份是重复 |
| `summary` | 同 `context`：它在构图阶段被消费掉，不需要流进图里 |

**保留并新增的字段**都是真有生产者的：每个都在 `nodes.py` 里有明确的
写入点，在图里有明确的读取点。
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_citations(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """合并引用列表，按 `doc_id` 去重。

    **为什么必须在 reducer 里做去重。** 图会执行多次检索（初次 + 反思补检索），
    每次都会产出一批引用。如果直接拼接，同一个片段会因为被两轮都召回而
    出现两次——用户在来源列表里看到两条一模一样的记录。

    去重键用 `doc_id`（即 `chunk_id`），它是 loader 打的稳定标识，
    重新入库后依然指向同一段内容。

    **为什么存 dict 而不是 CitationEvent。** 状态要能跨节点传递，
    且不该绑定某个事件的序列化时机——统一在 answer 节点转成事件，
    一处负责序列化，避免「不同节点发出的同一类事件形状不一致」。
    """
    seen = {str(c.get("doc_id") or "") for c in left}
    merged = list(left)
    for item in right:
        doc_id = str(item.get("doc_id") or "")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        merged.append(item)
    return merged


class AgentState(TypedDict):
    """图状态。

    字段分四组：输入、规划、检索产物、控制流。
    """

    # --- 输入（由 run_agent_stream 构造）---
    messages: Annotated[list[BaseMessage], add_messages]
    """对话消息。`add_messages` reducer 负责追加与按 id 去重。

    节点返回 `{"messages": [某条消息]}` 即可追加，不需要自己 copy 整个列表。
    """

    question: str
    """用户原始问题。**不要拿它直接去检索**——
    检索用 `retrieval_query`，那是 plan 节点改写过的、更适合检索的表达。
    """

    # --- 规划（plan 节点写）---
    plan: str
    """规划的自然语言说明，用于外发 PlanEvent 与排查。"""

    need_retrieval: bool
    """是否需要检索。plan 节点的核心决策，`after_plan` 条件边据此分流。"""

    retrieval_query: str
    """实际用于检索的 query。plan 改写；改写失败时回落成原问题。"""

    # --- 检索产物（retrieve / tools 节点写）---
    citations: Annotated[list[dict[str, Any]], merge_citations]
    """引用列表，按 doc_id 去重合并。answer 节点统一转成 CitationEvent 外发。"""

    # --- 控制流 ---
    iteration: int
    """agent 节点已执行轮数。超过上限时 `after_agent` 强制结束，
    防止「模型反复调工具」把一次问答拖成无限循环。"""

    reflection_round: int
    """反思补检索已执行次数。**上限 1**——反思的价值在于捕获「漏检」，
    不在于反复重试到满意；允许无限重试只会让延迟不可控。"""

    need_more_retrieval: bool
    """reflect 节点写、`after_reflect` 条件边读的决策结果。

    **为什么单独存一个字段，而不是让条件边重新判断。**
    条件边是「读状态做路由」的纯函数，如果它要靠 `reflect_reason` 的字符串
    前缀去猜决策，就变成了「解析自然语言来路由」——一处文案改动就会
    悄悄改变控制流。显式布尔字段把决策点收敛在 reflect 节点里，
    条件边只做机械转发。
    """

    reflect_reason: str
    """上次反思的判断依据，用于外发与排查。"""

    final_answer: str
    """最终答案文本。answer 节点写，外发 FinalEvent 用。"""
