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


class Reranker:
    def __init__(self, model_name: str | None = None):
        self._model = CrossEncoder(model_name or settings.reranker_model_path)

    def rerank(
        self, query: str, candidates: List[Tuple[Document, float]], top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc, _ in candidates]
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
