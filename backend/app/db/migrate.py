"""轻量幂等列检查。

`Base.metadata.create_all` 只建表、不改表 —— 对已存在的库新增列会**静默不生效**，
代码里 `conv.summary_upto_message_id` 读出来永远是 None，而新库是好的。这种
「新建库能用、老库不报错就是不工作」的故障最难查，所以这里在启动时补列。

仅登记**必需**的列。这是过渡方案，不是迁移框架 —— 列一多就应当换成 alembic。
"""
from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# (表名, 列名, DDL)
REQUIRED_COLUMNS: list[tuple[str, str, str]] = [
    ("conversations", "summary_upto_message_id", "INT NULL"),
    # MySQL 不接受给已有列加 NOT NULL 而不给默认值，因此补列时只能给 NULL，
    # 由应用侧把它当「未知」处理（去重时跳过空摘要的旧行，见 knowledge_service）。
    ("documents", "content_hash", "VARCHAR(64) NULL"),
    ("documents", "deleted_at", "DATETIME NULL"),
]


def ensure_columns(engine: Engine) -> None:
    """补齐缺失的列。对已存在的列是 no-op，可重复调用。

    必须在 `create_all` **之后**调用：新库由 create_all 直接带上这些列（此处 no-op），
    旧库才走 ALTER。
    """
    for table, column, ddl in REQUIRED_COLUMNS:
        # inspect() 每次循环重建：inspector 会缓存列信息，同一张表补第二列时会读到
        # 过期快照。当前只有一张表，但这个陷阱很难在补第二列时才注意到。
        inspector = inspect(engine)
        if table not in set(inspector.get_table_names()):
            continue
        if column in {c["name"] for c in inspector.get_columns(table)}:
            continue

        logger.info("补列: %s.%s", table, column)
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))