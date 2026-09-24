"""上下文工程引擎：分层 token 预算 + 消息拼装。

每层独立预算，层内超限先截断，**不跨层抢占**。
各层比例由 config.py 的 context_* 配置项控制。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.context.tokens import TokenCounter

# 截断时在层内容尾部补的提示，避免模型误以为内容完整
_TRUNCATION_NOTE = "\n…（内容因长度限制被截断）"


@dataclass(frozen=True)
class ContextBudget:
    total_tokens: int = 32_000
    system_ratio: float = 0.15
    memory_ratio: float = 0.10
    history_ratio: float = 0.20
    retrieved_ratio: float = 0.35
    scratchpad_ratio: float = 0.20

    def layer_budget(self, ratio: float) -> int:
        return int(self.total_tokens * ratio)

    @classmethod
    def from_settings(cls, settings) -> "ContextBudget":
        return cls(
            total_tokens=settings.context_total_tokens,
            system_ratio=settings.context_system_ratio,
            memory_ratio=settings.context_memory_ratio,
            history_ratio=settings.context_history_ratio,
            retrieved_ratio=settings.context_retrieved_ratio,
            scratchpad_ratio=settings.context_scratchpad_ratio,
        )


@dataclass(frozen=True)
class LayerStat:
    name: str
    budget: int
    used: int
    source_count: int
    kept_count: int
    truncated: bool


@dataclass(frozen=True)
class ContextResult:
    """装配结果。

    **不是 prompt 的硬性 token 上界。** 本引擎约束的是**每一层自己的 used**，而
    `question` 不过计数器、层标题与分隔符（【参考资料】、`"\\n\\n"` 等）也都不计，
    五层比例之和又恰好是 1.00。因此各层都填满时，最终 prompt 的实际 token 数会
    **超过** `ContextBudget.total_tokens`。

    需要硬上界的话得由调用方另行截断 —— Task 7 的 ReAct 循环每一轮都会重入
    `build`，更不能把它当成预算保证。
    """

    messages: list[BaseMessage]
    stats: list[LayerStat]


class ContextEngine:
    def __init__(self, budget: ContextBudget, counter: TokenCounter) -> None:
        self._budget = budget
        self._counter = counter

    def build(
        self,
        *,
        system: str,
        question: str,
        summary: str = "",
        history: Sequence[BaseMessage] = (),
        retrieved: Sequence[str] = (),
        scratchpad: str = "",
    ) -> ContextResult:
        stats: list[LayerStat] = []

        system_text, stat = self._fit_text("system", [system], self._budget.system_ratio)
        stats.append(stat)

        memory_text, stat = self._fit_text("memory", [summary] if summary else [], self._budget.memory_ratio)
        stats.append(stat)

        kept_history = self._fit_messages(list(history), self._budget.history_ratio)
        stats.append(
            LayerStat(
                name="history",
                budget=self._budget.layer_budget(self._budget.history_ratio),
                used=sum(self._counter.count(str(m.content)) for m in kept_history),
                source_count=len(history),
                kept_count=len(kept_history),
                truncated=len(kept_history) < len(history),
            )
        )

        retrieved_text, stat = self._fit_text("retrieved", list(retrieved), self._budget.retrieved_ratio)
        stats.append(stat)

        scratchpad_text, stat = self._fit_text("scratchpad", [scratchpad] if scratchpad else [], self._budget.scratchpad_ratio)
        stats.append(stat)

        messages: list[BaseMessage] = [SystemMessage(content=self._compose_system(system_text, memory_text))]

        # 历史按时间正序保留最近的部分
        messages.extend(kept_history)

        messages.append(HumanMessage(content=self._compose_user(question, retrieved_text, scratchpad_text)))

        return ContextResult(messages=messages, stats=stats)

    # --- internals ---

    def _compose_system(self, system_text: str, memory_text: str) -> str:
        parts = [system_text]
        if memory_text:
            parts.append(f"\n\n【长期记忆】\n{memory_text}")
        return "".join(parts)

    def _compose_user(self, question: str, retrieved_text: str, scratchpad_text: str) -> str:
        parts: list[str] = []
        if retrieved_text:
            parts.append(f"【参考资料】\n{retrieved_text}\n\n")
        if scratchpad_text:
            parts.append(f"【已获取的中间结果】\n{scratchpad_text}\n\n")
        parts.append(f"【用户问题】\n{question}")
        return "".join(parts)

    def _fit_text(self, name: str, items: Sequence[str], ratio: float) -> tuple[str, LayerStat]:
        """按顺序累积条目直到预算耗尽。条目本身超预算时硬截断。"""
        budget = self._budget.layer_budget(ratio)
        kept: list[str] = []
        used = 0
        truncated = False

        for item in items:
            if not item:
                continue
            remaining = budget - used
            if remaining <= 0:
                truncated = True
                break

            cost = self._counter.count(item)
            if cost <= remaining:
                kept.append(item)
                used += cost
                continue

            # 单条超预算：按比例硬截断。截断提示本身也占 token，必须一并算进预算，
            # 否则「截断后」的用量会反超该层预算。
            room = remaining - self._counter.count(_TRUNCATION_NOTE)
            if room <= 0:
                truncated = True
                break

            allowed_chars = max(1, int(len(item) * (room / cost)))
            clipped = item[:allowed_chars] + _TRUNCATION_NOTE
            # 比例估算在中英混排时与实际计数有偏差，按实测值收缩到预算内
            while allowed_chars > 1 and self._counter.count(clipped) > remaining:
                allowed_chars = max(1, allowed_chars * remaining // self._counter.count(clipped))
                clipped = item[:allowed_chars] + _TRUNCATION_NOTE

            if self._counter.count(clipped) > remaining:
                # 收缩到 1 个字符仍然塞不下，说明 count() 在这个长度区间是非单调的
                # （越短反而越贵）。TokenCounter 协议没有声明「缩短必然变便宜」这个
                # 前提，所以这里兜底：宁可整条丢掉，也不能让该层用量越界。
                # 启发式计数器与真实 BPE 都到不了这一步，属于防御性分支。
                truncated = True
                break

            kept.append(clipped)
            used += self._counter.count(clipped)
            truncated = True
            break

        return "\n\n".join(kept), LayerStat(
            name=name,
            budget=budget,
            used=used,
            source_count=len([i for i in items if i]),
            kept_count=len(kept),
            truncated=truncated or len(kept) < len([i for i in items if i]),
        )

    def _fit_messages(self, messages: list[BaseMessage], ratio: float) -> list[BaseMessage]:
        """从最近的消息往前保留，直到预算耗尽（保留时间连续的一段）。"""
        budget = self._budget.layer_budget(ratio)
        kept_reversed: list[BaseMessage] = []
        used = 0

        for message in reversed(messages):
            cost = self._counter.count(str(message.content))
            if used + cost > budget:
                break
            kept_reversed.append(message)
            used += cost

        return list(reversed(kept_reversed))
