"""数据模型统一出口。

**为什么必须在这里 import 全部模型。** SQLAlchemy 的 mapper 在第一次被使用时
才解析外键目标表。如果某个模块只 import 了 `UserMemory` 却没 import
`Conversation`，解析 `source_conversation_id` 的外键时就会抛
`NoReferencedTableError`——而且这个错误出现在**运行时第一次写库**，
不是导入时，很难定位。

实测踩过：单测里 `store.remember()` 直接失败，日志只有一句
「记忆入库失败：NoReferencedTableError: ... could not find table 'conversations'」。

统一在这里 import，任何 `import app.models` 的地方都能拿到完整的 mapper 注册。
"""
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.evaluation import Evaluation
from app.models.memory import UserMemory
from app.models.message import Message
from app.models.semantic_cache import SemanticCacheEntry

__all__ = [
    "Conversation",
    "Document",
    "Evaluation",
    "Message",
    "SemanticCacheEntry",
    "UserMemory",
]
