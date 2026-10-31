from datetime import datetime
from sqlalchemy import Integer, String, Float, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[str] = mapped_column(Text, nullable=False)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 原先还有 context_precision / context_recall 两列，从未被任何代码写入或读取。
    # 它们让模型暗示存在四层指标，实际只有两层。库里已有的列保留（删列是破坏性
    # 操作，不值得为整洁冒数据丢失的风险）。
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
