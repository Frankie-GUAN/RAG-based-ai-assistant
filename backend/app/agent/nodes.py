"""Agent 图的六个节点。

这张图把原来 `core.py` 里的一个 `for` 循环拆成了显式的状态机：

    START → plan →(需要检索)→ retrieve → agent →(有工具调用)→ tools ┐
                      │                     ↑                        │
                      └(不需要)→────────────┘←───────────────────────┘
                                           │
                                     (无工具调用)
                                           ↓
                                        reflect →(资料不足)→ retrieve（最多一次）
                                           │
                                       (资料充分)
                                           ↓
                                        answer → END

**为什么值得这么拆。** 旧实现的循环能跑，但它的「决策」是隐含在
`if accumulated.tool_calls:` 里的——你看代码才知道流程，改一处要读全文。
拆成图之后：

1. **流程可视化**：`graph.get_graph().draw_mermaid()` 直接出图，
   面试时能展示，排查时能对照
2. **决策显式化**：每个分叉是一个有名字的条件边函数
   （`after_plan` / `after_agent` / `after_reflect`），不是一个 if
3. **能加真正的 Planning 与 Reflection**：旧循环只会在「模型要调工具」时
   继续，永远不会主动判断「我该先检索吗」和「我答得够不够」

**流式怎么保证不丢。** 每个节点内部用 `get_stream_writer()` 拿到一个
写通道，把 `AgentEvent.to_dict()` 的结果直接写出去；外层用
`astream(stream_mode="custom")` 原样收到。实测事件的 dict 形状与
旧实现 `yield event` 完全一致，所以**前端零改动**。

**关于依赖注入。** 节点需要模型、工具客户端等运行时依赖，但 LangGraph
的节点签名固定是 `(state) -> dict`。这里用「实例持有 deps」的方式解决：
每次请求 new 一个 `AgentNodes`，再据此构图。这样请求级状态
（如 `seen_doc_ids`）天然隔离，不会在并发请求间串味。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.config import get_stream_writer

from app.agent.protocol import (
    CitationEvent,
    ErrorEvent,
    FinalEvent,
    PlanEvent,
    ReflectEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.agent.state import AgentState
from app.context.engine import ContextEngine
from app.mcp.client import MCPToolClient

logger = logging.getLogger(__name__)

# 一次回答最多外发多少条引用。检索默认取 top_k=4 条，一般不会触顶；
# 触顶时保留最靠前的那些 —— 它们是精排/融合后的前排。
MAX_CITATIONS = 8

# 每个引用写进上下文时截断到多少字符。
# 4 条 × 800 字符已经接近 retrieved 层的预算，再长会挤掉对话历史。
_EVIDENCE_CHARS_PER_CITATION = 800

SYSTEM_PROMPT = (
    "你是智能问答助手，基于私有文档知识库与联网搜索回答用户问题。"
    "你可以调用工具检索私有文档、联网搜索、解析上传的文档。\n"
    "规则：\n"
    "1. 需要事实依据时先调用工具，不要凭记忆回答文档中的内容或数据。\n"
    "2. 调用工具时不要输出解释性文字。\n"
    "3. 资料不足时明确说明不确定，不要编造。\n"
    "4. 引用检索结果时说明来源。\n"
    "5. 如果上下文中已经给出了【检索到的资料】，说明系统已经替你检索过；"
    "资料足够就直接作答，不要重复检索同一内容。"
)


@dataclass
class AgentTask:
    question: str
    history: list[BaseMessage] = field(default_factory=list)
    summary: str = ""
    # 跨会话长期记忆（用户偏好 / 已确认事实），由 chat_service 召回后注入。
    # 与 summary 分开传：前者跨会话有效，后者只属于当前会话，
    # 模型需要对它们区别对待。
    long_term_memory: str = ""


@dataclass
class AgentDeps:
    """Agent 的运行时依赖。

    `planner` 为 None 时**不做预检索规划**，由 agent 节点自行决定是否调工具
    ——这与迁移前的行为完全一致。传了 planner 才启用真正的 Planning：
    先规划、再检索、最后反思。

    为什么要可关：

    · **测试可复现**：`ScriptedChatModel` 是按调用顺序吐脚本的。
      若 plan 也调模型，所有现有脚本测试的顺序都会错位。
      默认关掉，测试无需改动就能跑。
    · **延迟可控**：每次规划都是一次额外 LLM 往返。
      对「你好」这类问题纯属浪费，可以按场景关掉。
    """

    model: BaseChatModel
    tools: MCPToolClient
    context: ContextEngine
    max_iterations: int = 8
    planner: BaseChatModel | None = None


def _as_text(content: Any) -> str:
    """兼容 str 与 LangChain 的多模态 content block 列表。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出第一个 JSON 对象。

    **模型经常把 JSON 包在 ```json 代码块里**，或前后带一句解释。
    直接 `json.loads` 会失败，而失败就意味着规划降级——
    这是会被用户感知到的（预检索没发生）。所以先按代码块抠，
    再退化到「找第一个平衡括号」。
    """
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []

    # 找第一个花括号平衡的子串
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break

    for chunk in candidates:
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError):
            continue
    return None


def _citations_from(data: Any, seen: set[str]) -> list[dict[str, Any]]:
    """把 search_documents 的返回载荷转成引用字典列表。

    载荷形状是 MCP 信封 `{"ok", "data", "error"}`（见 app/mcp/client.py），
    真正的检索结果在 `data` 里，是一个 list[dict]，每项含
    doc_id / content / source / score。

    **不做严格校验。** 载荷形状变了就返回空列表（本次回答就没有来源），
    而不是让整个工具结果解析失败 —— 溯源是增强，不该拖垮问答本身。
    """
    if isinstance(data, dict) and set(data) == {"ok", "data", "error"}:
        data = data["data"]
    if not isinstance(data, list):
        return []

    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or "")
        content = str(item.get("content") or "")
        if not doc_id or not content or doc_id in seen:
            continue
        seen.add(doc_id)
        out.append({
            "doc_id": doc_id,
            "chunk": content[:500],
            # 完整正文单独留一份：写进上下文用，展示用的 chunk 是截断版
            "_full_text": content,
            "source": str(item.get("source") or ""),
            "score": float(item.get("score") or 0.0),
            "page_title": str(item.get("page_title") or ""),
            "section_path": [str(p) for p in (item.get("section_path") or []) if p],
        })
    return out


def _format_evidence(citations: list[dict[str, Any]]) -> str:
    """把检索结果渲染成给模型看的资料块。

    **为什么不塞成 ToolMessage。** 预检索不是模型发起的，
    没有对应的 `tool_call_id`。伪造一个会让模型困惑
    （它会在自己的调用记录里找不到这次调用）。所以用一条
    HumanMessage 承载资料，并显式标注「系统已替你检索」。
    """
    if not citations:
        return "【检索结果】未在知识库中找到与该问题相关的内容。请如实告知用户资料不足，不要编造。"

    lines = ["【检索到的资料】（系统已检索，请直接基于以下内容作答，无需重复检索）"]
    for i, cite in enumerate(citations, start=1):
        where = cite.get("source") or "未知来源"
        section = " › ".join(cite.get("section_path") or [])
        if section:
            where = f"{where} › {section}"
        body = str(cite.get("_full_text") or cite.get("chunk") or "")
        lines.append(f"\n[{i}] 来源：{where}")
        lines.append(body[:_EVIDENCE_CHARS_PER_CITATION])
    return "\n".join(lines)


class AgentNodes:
    """六个节点的实现。实例级持有 deps 与请求级状态。"""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        # 引用去重用的稳定标识集合。实例级而非图级，
        # 保证并发请求之间不串味。
        self._seen_doc_ids: set[str] = set()
        schemas = [spec.to_openai_schema() for spec in deps.tools.specs]
        self._bound = deps.model.bind_tools(schemas) if schemas else deps.model

    # ------------------------------------------------------------------
    # plan
    # ------------------------------------------------------------------
    async def plan(self, state: AgentState) -> dict[str, Any]:
        """决定「要不要检索」与「用什么 query 检索」。

        **这是旧实现完全没有的一步。** 旧版把用户原话直接丢给模型，
        由模型临场决定调不调工具。问题是：模型的工具调用决策受
        system prompt 影响很大，且它看不到「这个知识库里有没有相关内容」。
        先规划一次，可以把口语问题改写成检索友好的 query
        （「这个算法咋跑的」→「worklist 算法 指针分析」），
        显著提升召回质量。
        """
        question = state["question"]
        writer = get_stream_writer()

        if self.deps.planner is None:
            # 未启用规划：不做预检索，交给 agent 自主决定（与旧实现一致）
            writer(PlanEvent(
                need_retrieval=False,
                query=question,
                reason="未启用规划，由模型自主决定是否检索",
            ).to_dict())
            return {
                "need_retrieval": False,
                "retrieval_query": question,
                "plan": "",
            }

        decision = await self._llm_plan(question)
        writer(PlanEvent(
            need_retrieval=decision["need_retrieval"],
            query=decision["query"],
            reason=decision["reason"],
        ).to_dict())
        return {
            "need_retrieval": decision["need_retrieval"],
            "retrieval_query": decision["query"],
            "plan": decision["reason"],
        }

    async def _llm_plan(self, question: str) -> dict[str, Any]:
        """让规划模型判断是否需要检索，并改写检索 query。

        **任何失败都降级为「需要检索 + 原问题」。** 理由：这个系统的
        主场景就是知识库问答，默认检索一次比默认不检索更安全——
        最坏情况是多一次检索延迟，而不是「明明有资料却说不知道」。
        """
        prompt = (
            "判断下面的问题是否需要检索私有文档库，并把问题改写成适合检索的查询。\n"
            "只输出一个 JSON 对象，不要任何解释：\n"
            '{"need_retrieval": true 或 false, "query": "检索查询", "reason": "不超过20字的理由"}\n\n'
            "判断规则：\n"
            "- 寒暄、纯逻辑推理、写代码、翻译 → need_retrieval = false\n"
            "- 需要事实、文档内容、专业概念、数据 → need_retrieval = true\n"
            "- query 要保留专业术语，去掉「请问」「能不能」这类口语词\n\n"
            f"问题：{question}"
        )
        try:
            response = await self.deps.planner.ainvoke(prompt)  # type: ignore[union-attr]
            parsed = _extract_json(_as_text(response.content))
        except Exception as exc:  # noqa: BLE001 — 规划失败不该中断问答
            logger.warning("规划节点失败，降级为直接检索：%s: %s", type(exc).__name__, exc)
            return {
                "need_retrieval": True,
                "query": question,
                "reason": "规划失败，按原问题检索",
            }

        if not parsed:
            return {
                "need_retrieval": True,
                "query": question,
                "reason": "规划输出无法解析，按原问题检索",
            }

        need = parsed.get("need_retrieval")
        if not isinstance(need, bool):
            need = True

        query = str(parsed.get("query") or "").strip() or question
        reason = str(parsed.get("reason") or "").strip()[:60]
        return {"need_retrieval": need, "query": query, "reason": reason}

    # ------------------------------------------------------------------
    # retrieve
    # ------------------------------------------------------------------
    async def retrieve(self, state: AgentState) -> dict[str, Any]:
        """按规划的 query 主动检索一次，把结果作为资料注入上下文。

        **事件仍然发 tool_call / tool_result。** 前端据此显示「正在检索」
        与耗时，体验与模型自主调用工具时一致——用户不需要知道
        这次检索是规划的产物还是模型发起的。

        ## 初次检索为什么用「原问题」而不是规划改写的 query

        这一条是**实测推翻直觉**的结论，不要凭直觉改回去。

        实测（同一语料，top_k=4）：

        | query | 命中 Week 6（常量传播所在章） |
        |---|---|
        | `常量传播分析怎么做`（原问题） | **是**，4 条全中，含「Constant Propagation – Lattice」 |
        | `常量传播分析 实现方法`（规划改写） | **否**，召回的是指针分析的「Implementing Pointer Analysis」 |

        原因有两层：

        1. **改写会丢术语**。跨语言检索靠的是「query 里出现 Constant Propagation」
           这个词面信号（见 `rag/terminology.py`）。模型把「怎么做」改写成
           「实现方法」时，很可能顺手把原措辞也动了，术语映射就失效了。
        2. **改写会引入语义偏移**。「实现方法」在语料里对应的是
           POINTER ANALYSIS 的 "Implementing"，而不是常量传播——
           改写反而把检索带到了干扰项上。

        所以：**初次检索一律用用户原话**（它自带正确术语，且术语映射就是
        基于它工作的），规划改写的 query 留给**反思补检索**——
        那时换个角度搜才是有价值的，因为它要搜的是「第一次没找到的东西」。
        """
        writer = get_stream_writer()
        rounds = state.get("reflection_round", 0)
        if rounds == 0:
            query = state["question"]
        else:
            query = state.get("retrieval_query") or state["question"]

        # 用固定 id 而非随机值：同一请求里「规划检索」只会出现 1~2 次，
        # 固定前缀足够前端区分，也便于日志里一眼认出这不是模型发起的调用
        call_id = f"plan_retrieval_{rounds}"

        writer(ToolCallEvent(
            id=call_id, name="search_documents", args={"query": query}
        ).to_dict())

        try:
            result = await self.deps.tools.call("search_documents", {"query": query})
        except Exception as exc:  # noqa: BLE001 — 检索失败不该中断问答
            logger.warning("预检索失败：%s: %s", type(exc).__name__, exc)
            writer(ToolResultEvent(
                id=call_id, ok=False, data=None, ms=0, error=f"{type(exc).__name__}: {exc}"
            ).to_dict())
            return {
                "messages": [HumanMessage(
                    content="【检索结果】检索工具调用失败，无法获取资料。"
                            "请如实告知用户当前无法访问知识库。"
                )],
            }

        writer(ToolResultEvent(
            id=call_id, ok=result.ok, data=result.data, ms=result.ms, error=result.error
        ).to_dict())

        citations = _citations_from(result.data, self._seen_doc_ids) if result.ok else []
        citations = citations[:MAX_CITATIONS]

        return {
            "messages": [HumanMessage(content=_format_evidence(citations))],
            "citations": citations,
        }

    # ------------------------------------------------------------------
    # agent
    # ------------------------------------------------------------------
    async def agent(self, state: AgentState) -> dict[str, Any]:
        """主推理节点：流式产出 token，并决定是否调用工具。

        **流式的关键。** `astream` 逐 chunk 累积成完整 AIMessage 的同时，
        把文本增量立刻写给前端。工具调用参数也在同一批 chunk 里
        （`tool_call_chunks`），流结束后 LangChain 已把它们合并为完整的
        `tool_calls`，所以这里不需要自己拼接 JSON 片段。
        """
        writer = get_stream_writer()
        messages = state["messages"]

        accumulated: AIMessage | None = None
        async for chunk in self._bound.astream(messages):
            accumulated = chunk if accumulated is None else accumulated + chunk
            text = _as_text(chunk.content)
            if text:
                writer(TokenEvent(text=text).to_dict())

        iteration = state.get("iteration", 0) + 1

        if accumulated is None:
            writer(ErrorEvent(
                code="empty_response", message="模型未返回任何内容", recoverable=False
            ).to_dict())
            return {"messages": [], "iteration": iteration, "final_answer": ""}

        return {"messages": [accumulated], "iteration": iteration}

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------
    async def tools(self, state: AgentState) -> dict[str, Any]:
        """执行模型发起的工具调用（并发），并把结果回灌给模型。

        **先外发全部 tool_call 再并发执行。** 让前端立刻看到并发意图，
        而不是等两个工具都跑完才一起显示——后者在慢工具上会让
        界面白等好几秒。
        """
        writer = get_stream_writer()

        last_ai = self._last_ai_message(state["messages"])
        calls = list(getattr(last_ai, "tool_calls", []) or []) if last_ai else []
        if not calls:
            return {"messages": []}

        for call in calls:
            writer(ToolCallEvent(
                id=call["id"], name=call["name"], args=call["args"]
            ).to_dict())

        results = await asyncio.gather(
            *(self.deps.tools.call(call["name"], call["args"]) for call in calls)
        )

        new_messages: list[BaseMessage] = []
        new_citations: list[dict[str, Any]] = []

        for call, result in zip(calls, results):
            writer(ToolResultEvent(
                id=call["id"], ok=result.ok, data=result.data, ms=result.ms, error=result.error
            ).to_dict())
            new_messages.append(ToolMessage(
                content=json.dumps(
                    {"ok": result.ok, "data": result.data, "error": result.error},
                    ensure_ascii=False,
                    default=str,
                ),
                tool_call_id=call["id"],
            ))

            # 引用溯源：只认检索工具。其它工具（联网搜索、文档解析）
            # 的返回不产生文档引用——否则来源列表里会混进非知识库的内容。
            if result.ok and call["name"] == "search_documents":
                new_citations.extend(
                    _citations_from(result.data, self._seen_doc_ids)
                )

        return {
            "messages": new_messages,
            "citations": new_citations[:MAX_CITATIONS],
        }

    # ------------------------------------------------------------------
    # reflect
    # ------------------------------------------------------------------
    async def reflect(self, state: AgentState) -> dict[str, Any]:
        """自检：本次回答的资料够不够。

        **为什么用确定性规则而不是再调一次 LLM。**

        LLM 自评的成本是多一次往返（约 1 秒），而可靠度并不高——
        模型倾向于给自己的答案打高分。而这里要捕获的失败模式是
        高度具体的：「规划说需要检索，但一条引用都没拿到」。
        这个信号用 `citations` 就能精确判断，不需要模型。

        **只补检索一次。** 反思的价值是兜住漏检，不是反复重试到满意。
        允许无限重试会让延迟不可控，而且第二次搜不到通常意味着
        知识库里真的没有。
        """
        writer = get_stream_writer()
        citations = state.get("citations") or []
        need_retrieval = state.get("need_retrieval", False)
        rounds = state.get("reflection_round", 0)

        if not need_retrieval:
            reason = "规划判定无需检索，直接作答"
            need_more = False
        elif citations and rounds == 0:
            reason = f"已获得 {len(citations)} 条引用，资料充分"
            need_more = False
        elif citations and rounds > 0:
            reason = f"补充检索后共 {len(citations)} 条引用，资料充分"
            need_more = False
        elif rounds >= 1:
            reason = "已补充检索过一次仍无结果，按资料不足作答"
            need_more = False
        else:
            reason = "规划判定需要检索，但本轮未获得任何引用，补充检索一次"
            need_more = True

        writer(ReflectEvent(need_more=need_more, reason=reason).to_dict())

        return {
            "need_more_retrieval": need_more,
            "reflect_reason": reason,
            "reflection_round": rounds + (1 if need_more else 0),
        }

    # ------------------------------------------------------------------
    # answer
    # ------------------------------------------------------------------
    async def answer(self, state: AgentState) -> dict[str, Any]:
        """收尾：外发引用与最终答案。

        **引用在答案之前发出。** 前端据此在答案下方渲染来源标签；
        顺序反了会出现「答案先到、来源后补」的闪烁。
        """
        writer = get_stream_writer()

        answer_text = ""
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage):
                answer_text = _as_text(message.content)
                break

        citations = state.get("citations") or []
        for cite in citations[:MAX_CITATIONS]:
            writer(CitationEvent(
                doc_id=str(cite.get("doc_id") or ""),
                chunk=str(cite.get("chunk") or ""),
                source=str(cite.get("source") or ""),
                score=float(cite.get("score") or 0.0),
                page_title=str(cite.get("page_title") or ""),
                section_path=list(cite.get("section_path") or []),
            ).to_dict())

        # 达到迭代上限且模型仍在反复调工具：如实报错，不假装成功。
        if not answer_text and state.get("iteration", 0) >= self.deps.max_iterations:
            writer(ErrorEvent(
                code="max_iterations",
                message=f"达到迭代上限 {self.deps.max_iterations}，未能得出结论",
                recoverable=True,
            ).to_dict())
            return {"final_answer": ""}

        writer(FinalEvent(answer=answer_text).to_dict())
        return {"final_answer": answer_text}

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None


# 下面三个是条件边函数：纯函数，只读状态、返回下一个节点名。
# 单独拿出来（而不是写在 graph.py 里内联 lambda）是为了能被单测直接调用。


def after_plan(state: AgentState) -> str:
    """规划后的分流：需要检索就先检索，否则直接进推理。

    **为什么检索要放在 agent 之前，而不是全靠模型自觉调工具。**
    模型的工具调用受 prompt 影响，且它无法预判知识库里有没有相关内容。
    先按规划好的 query 检索一次，能把「资料」直接摆到模型面前，
    比让它自己决定更稳。
    """
    return "retrieve" if state.get("need_retrieval") else "agent"


def after_agent(state: AgentState, max_iterations: int = 8) -> str:
    """推理后的分流：有工具调用就去执行，否则进入反思。

    迭代超限时直接走 answer 收尾——answer 节点会发 ErrorEvent，
    不会静默地给一个半成品答案。
    """
    if state.get("iteration", 0) >= max_iterations:
        return "answer"
    messages = state.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return "tools" if getattr(message, "tool_calls", None) else "reflect"
    return "reflect"


def after_reflect(state: AgentState) -> str:
    """反思后的分流：资料不足就补检索一次，否则作答。"""
    return "retrieve" if state.get("need_more_retrieval") else "answer"
