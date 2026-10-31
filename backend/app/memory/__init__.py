"""三层记忆的第二层：跨会话长期记忆。

对外只需要 `get_memory_store()` 与 `extract_and_store()`。

| 层 | 位置 | 状态 |
|---|---|---|
| 短期 | `Conversation.summary` + `history_window_size` | 已有 |
| 长期 | 本包（`store.py` 存储 / `extractor.py` 抽取） | 新增 |
| 语义缓存 | `app/cache/semantic.py` | 新增 |
"""
from app.memory.store import (
    DEFAULT_USER_ID,
    MemoryItem,
    MemoryStore,
    get_memory_store,
)

__all__ = ["DEFAULT_USER_ID", "MemoryItem", "MemoryStore", "get_memory_store"]
