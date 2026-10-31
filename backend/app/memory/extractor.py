"""从对话里抽取长期记忆。

## 为什么要「规则触发 + LLM 抽取」两层

**只有 LLM**：每轮对话都要多一次调用（约 1 秒 + 费用），而绝大多数轮次
根本没有值得记的东西（「这个函数怎么用」「谢谢」）。

**只有规则**：规则能判断「这句话像不像在表达偏好」，但抽不出结构化内容。
「以后回答我都要带例子」这条规则的命中，仍然需要模型翻译成
「用户偏好带代码示例的回答」。

所以两层各管一段：**规则负责「要不要抽」，LLM 负责「抽成什么」。**

## 规则怎么定

从「什么样的句子值得跨会话记住」这个角度出发，只有三类：

| 信号 | 例子 | 记忆类型 |
|---|---|---|
| 显式要求记住 | 「记住我用 MySQL 8」「以后都…」 | 看内容 |
| 表达偏好 | 「我偏好简短回答」「太长了」「我喜欢…」 | preference |
| 纠正/确认事实 | 「不是这样，应该是…」「实际上我们是…」 | fact |

**规则必须保守。** 一次误触发 = 一次白花的 LLM 调用 + 一条污染召回的记忆。
宁可漏记（用户下次再说一次），不可错记（错误记忆会被反复召回并注入上下文）。

## 为什么跑在后台

抽取要调 LLM（约 1 秒）。放在主链路里会直接把首字节延迟拉长一秒，
而用户完全不关心「记忆有没有存上」。

用 `asyncio.create_task` 起一个不等待的任务。**注意**：这依赖
`chat_service` 里那次调用的生命周期足够长（SSE 流还在跑），
否则任务可能被取消。真正生产化应该交给 Celery/后台队列——
本项目当前没有，属于已知限制，已记在 verified-facts 里。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from langchain_core.language_models import BaseChatModel

from app.memory.store import DEFAULT_USER_ID, get_memory_store

logger = logging.getLogger(__name__)

# 触发抽取的信号词。
#
# 每一条都对应一类「值得跨会话记住」的表达，没有凑数的。
# 加新词时问自己：用户说这句话，下个星期换个会话还应该生效吗？
# 答「是」才加。
_TRIGGERS: tuple[tuple[str, str], ...] = (
    # 显式要求记住
    ("记住", "fact"),
    ("记一下", "fact"),
    ("以后都", "preference"),
    ("以后回答", "preference"),
    ("下次", "preference"),
    # 表达偏好
    ("我偏好", "preference"),
    ("我喜欢", "preference"),
    ("我更希望", "preference"),
    ("我习惯", "preference"),
    ("太长了", "preference"),
    ("简短", "preference"),
    ("详细一点", "preference"),
    ("不要用", "preference"),
    # 纠正 / 确认事实
    ("不是这样", "fact"),
    ("应该是", "fact"),
    ("实际上", "fact"),
    ("我们的", "fact"),
    ("我们是", "fact"),
)

_EXTRACT_PROMPT = (
    "从下面这轮对话里抽取「值得跨会话长期记住」的信息。\n"
    "只输出 JSON 数组，不要任何解释。没有可记的就输出 []。\n\n"
    "每条格式：\n"
    '{"kind": "preference" 或 "fact" 或 "term", "content": "一句话，第三人称陈述", "confidence": 0.0~1.0}\n\n'
    "判断标准（严格）：\n"
    "- 只记**长期有效**的信息。临时请求（「这次回答短一点」不算，"
    "「以后都回答短一点」才算）不要记。\n"
    "- 记**关于用户或用户环境**的事实，不要记知识库里的知识\n"
    "  （知识库内容用检索解决，不需要记忆）。\n"
    "- content 要能独立看懂，不要出现「这个」「刚才」这类指代。\n"
    "- 最多 2 条。宁缺勿滥。\n\n"
    "对话：\n用户：{question}\n助手：{answer}\n"
)


def detect_signal(text: str) -> str | None:
    """判断这句话是否值得触发抽取，返回候选的 kind。

    返回 None 表示不触发。
    """
    if not text:
        return None
    for keyword, kind in _TRIGGERS:
        if keyword in text:
            return kind
    return None


def _parse_memories(raw: str) -> list[dict]:
    """从模型输出里抠出 JSON 数组。

    模型经常包 ```json 代码块，或前后带解释。直接 json.loads 会失败，
    而失败就等于「这轮白抽了」——用户以为说过了，系统没记住。
    """
    if not raw:
        return []

    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []

    start = raw.find("[")
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "[":
                depth += 1
            elif raw[i] == "]":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start : i + 1])
                    break

    for chunk in candidates:
        try:
            parsed = json.loads(chunk)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]

    # 模型只给了单个对象也接受
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw[start : i + 1])
                        if isinstance(obj, dict):
                            return [obj]
                    except (TypeError, ValueError):
                        pass
                    break
    return []


async def extract_and_store(
    question: str,
    answer: str,
    *,
    model: BaseChatModel,
    conversation_id: int | None = None,
    user_id: str = DEFAULT_USER_ID,
) -> int:
    """抽取并落库，返回新增条数。失败一律返回 0，不抛。

    **这是后台任务，任何异常都必须吞掉** —— 一个记忆抽取失败不该在
    服务端日志里刷出 traceback 惊动运维，更不该影响已经完成的问答。
    """
    if not detect_signal(question):
        return 0

    try:
        response = await model.ainvoke(
            _EXTRACT_PROMPT.format(question=question, answer=answer[:2000])
        )
        content = response.content
        raw = content if isinstance(content, str) else str(content)
        candidates = _parse_memories(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("记忆抽取失败：%s: %s", type(exc).__name__, exc)
        return 0

    if not candidates:
        return 0

    store = get_memory_store()
    stored = 0
    for item in candidates[:2]:  # 兜底：即使模型给了更多也只取 2 条
        text = str(item.get("content") or "").strip()
        if not text:
            continue
        kind = str(item.get("kind") or "fact").strip()
        if kind not in ("preference", "fact", "term"):
            kind = "fact"
        try:
            confidence = float(item.get("confidence") or 1.0)
        except (TypeError, ValueError):
            confidence = 1.0

        # 置信度过低的不要——错误记忆会被反复召回，比没有记忆更糟
        if confidence < 0.5:
            continue

        try:
            if store.remember(
                text,
                kind=kind,
                user_id=user_id,
                confidence=confidence,
                conversation_id=conversation_id,
            ):
                stored += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("记忆写库失败：%s: %s", type(exc).__name__, exc)

    return stored


def schedule_extraction(
    question: str,
    answer: str,
    *,
    model: BaseChatModel,
    conversation_id: int | None = None,
) -> "asyncio.Task | None":
    """在后台起一个抽取任务，不等待。

    **先做一次廉价的规则判断再起任务**：没有信号就完全不创建 Task，
    避免每轮问答都挂一个空任务。

    返回 Task 供测试断言；没有信号时返回 None。
    """
    if not detect_signal(question):
        return None

    async def _run() -> None:
        await extract_and_store(
            question, answer, model=model, conversation_id=conversation_id
        )

    try:
        return asyncio.create_task(_run())
    except RuntimeError:
        # 没有运行中的事件循环（同步调用场景）：放弃后台抽取，
        # 不能为了记忆把主流程搞崩
        logger.warning("无事件循环，跳过本次记忆抽取")
        return None
