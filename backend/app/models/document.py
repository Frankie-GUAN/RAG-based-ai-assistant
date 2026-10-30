from datetime import datetime
from sqlalchemy import Integer, String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    # 内容的 sha256，用于挡住「同一份文件换个名字再传一遍」。Chroma 与 BM25 都不
    # 按内容去重，所以没有这一列就只能靠文件名猜。索引见 app/db/migrate.py：
    # create_all 不会给已存在的表加列，老库必须靠 ensure_columns 补。
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    # 软删除标记。见 services/knowledge_service.delete_document：
    # 直接删 Chroma 里的向量会毁掉整个 HNSW 段（实测三轮皆坏，隔离环境复现不了），
    # 所以向量侧只标记不删，靠检索过滤 + 索引重建清除。
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
