import logging
import threading
import time
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)

# 加载失败后的重试间隔。语义与 bm25_index 的 _load_failed 一致：
# 一次瞬时故障不该让精排在进程生命周期内永久缺席，但也不该让**每次**检索
# 都去重试一次 HuggingFace 下载。
_RETRY_INTERVAL_SECONDS = 600.0

# 送进 Cross-Encoder 的候选文本最大字符数。
#
# **为什么要截断。** 精排耗时与候选文本长度成正比（CPU 推理）：
# 实测 10 个候选、每段 400~450 字符时，精排要 4.4~6.9 秒。
# 而课件片段的前 300 字符通常已经包含标题与首句——
# 标题是「这一片讲什么」的最强信号，后面的正文对「是否相关」贡献递减。
#
# 实测（同一批候选，逐档对比耗时与 top4 排序）：
#     不截断      4.39s   top4 = [p59, p60, p63, p61]
#     截断 800    4.09s   top4 = [p59, p60, p63, p61]
#     截断 500    3.44s   top4 = [p59, p60, p63, p61]
#     截断 300    3.02s   top4 = [p59, p60, p62, p63]   ← 只差一位
#     截断 200    2.07s   top4 = [p60, p59, p62, p63]   ← top1 变了，太短
#
# 取 300：省掉约 30% 耗时，排序基本不变。
# **注意这是取舍不是免费午餐**——200 字符就会改变判断结果，所以下限守在 300。
_MAX_CANDIDATE_CHARS = 300


def _clip_for_rerank(text: str) -> str:
    """截断候选文本到 `_MAX_CANDIDATE_CHARS`。

    **截断会丢信息**，所以这个值不能随便调小——
    200 字符就足以让精排把 top1 换掉（见上方实测）。
    """
    if len(text) <= _MAX_CANDIDATE_CHARS:
        return text
    return text[:_MAX_CANDIDATE_CHARS]


class Reranker:
    def __init__(self, model_name: str | None = None):
        self._model = CrossEncoder(model_name or settings.reranker_model_path)

    def rerank(
        self, query: str, candidates: List[Tuple[Document, float]], top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        if not candidates:
            return []

        pairs = [(query, _clip_for_rerank(doc.page_content)) for doc, _ in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for (doc, _), score in ranked[:top_k]]


_reranker: Reranker | None = None
_reranker_lock = threading.Lock()
# 模型不可用（没下载 / 无网络 / 磁盘满）时记下原因与时间，冷却期内直接跳过精排。
_unavailable_until: float = 0.0
_unavailable_reason: str | None = None


def get_reranker() -> Reranker:
    """Cross-Encoder 的进程内单例；模型不可用时抛 RuntimeError。

    **冷却机制。** 失败过一次之后，在 `_RETRY_INTERVAL_SECONDS` 内不再尝试加载。
    没有它，一个没下载精排模型的环境会让**每次**检索都去试一次 HuggingFace 下载
    ——而 Windows venv 里这往往以SSL 证书校验失败告终，重试 5 次要好几秒。
    也就是说，缺模型会让每一条查询都平白变慢，而不是只慢一次。

    调用方（hybrid_search._rerank）捕获异常并退回 RRF 顺序，所以这里的失败
    只意味着「这次没有精排」，检索本身照常工作。
    """
    global _reranker, _unavailable_until, _unavailable_reason

    if _reranker is not None:
        return _reranker

    now = time.monotonic()
    if now < _unavailable_until:
        raise RuntimeError(
            f"精排模型不可用（{_unavailable_reason}），"
            f"{int(_unavailable_until - now)} 秒后重试。"
            f"可用 settings.enable_reranker=false 关闭精排，"
            f"或把模型放到 {settings.reranker_model_path}。"
        )

    with _reranker_lock:
        if _reranker is not None:
            return _reranker
        if time.monotonic() < _unavailable_until:      # 等锁期间可能已被别的线程判负
            raise RuntimeError(f"精排模型不可用（{_unavailable_reason}），稍后重试。")
        try:
            _reranker = Reranker()
        except Exception as exc:  # noqa: BLE001 — 任何加载失败都归入「不可用」
            _unavailable_until = time.monotonic() + _RETRY_INTERVAL_SECONDS
            _unavailable_reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "精排模型加载失败，%d 秒内不再尝试（检索将退回 RRF 顺序）：%s",
                int(_RETRY_INTERVAL_SECONDS),
                _unavailable_reason,
            )
            raise

        _unavailable_until = 0.0
        _unavailable_reason = None
        return _reranker


def reset_reranker_state() -> None:
    """清掉「不可用」记忆（测试与运维用）。"""
    global _reranker, _unavailable_until, _unavailable_reason
    with _reranker_lock:
        _reranker = None
        _unavailable_until = 0.0
        _unavailable_reason = None
