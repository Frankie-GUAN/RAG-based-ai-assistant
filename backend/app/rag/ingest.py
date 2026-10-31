"""从 data/uploads 重新入库：解析 → 分块 → 写 Chroma → 重建 BM25。

    python -m app.rag.ingest            # 增量写入 Chroma，BM25 全量重建
    python -m app.rag.ingest --fresh    # 先把旧 Chroma 目录改名备份，再重建

**为什么需要这条命令。** 此前 uploads 里的原文件与两个索引之间没有任何可重放的
路径：索引坏了就只能手工 pickle，或者把文档重新上传一遍。而以下三种情况都会让
「重新上传」变成唯一选择：

- 分词器变更后BM25 的已分词语料与新切法对不上；
- Chroma 的持久化格式随版本变化，旧数据读不出来（本次：0.5.x 写的
  `hnsw-local-persisted` 段，1.5.9 无法加载）；
- 索引文件损坏，而Chroma 里的向量还在。

**不碰 MySQL。** `DocumentModel` 行记录的是「上传过哪些文件」，由上传接口负责；
本命令只修复索引。重复跑本命令不会产生重复的 DocumentModel 行，但会**重复写入
Chroma** —— 因此索引重复时应当用 `--fresh`，而不是反复执行增量写入。
"""
from __future__ import annotations

import argparse
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from app.config import settings
from app.rag.bm25_index import BM25Index
from app.rag.exclusive import ensure_exclusive_access
from app.rag.loader import load_file, split_documents, supported_suffixes
from app.rag.vector_store import add_documents

logger = logging.getLogger(__name__)

# 支持的扩展名与 loader 保持一致（.pdf/.txt/.md/.markdown/.docx）—— 直接问
# loader，不要在这里抄一份：抄的副本一定会先于真正的支持列表漂移。


@dataclass
class IngestReport:
    files_ok: int = 0
    files_failed: int = 0
    chunks: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    backup_dir: Path | None = None

    def summary(self) -> str:
        lines = [
            f"成功 {self.files_ok} 个文件，共 {self.chunks} 个片段",
            f"失败 {self.files_failed} 个文件",
        ]
        if self.backup_dir:
            lines.append(f"旧索引已备份到 {self.backup_dir}")
        for name, reason in self.failures:
            lines.append(f"  ✗ {name}: {reason}")
        return "\n".join(lines)


def backup_chroma_dir() -> Path | None:
    """把当前 Chroma 目录改名备份，返回备份路径（不存在则 None）。

    用改名而不是删除：数据是本地知识库的全部内容，磁盘上的向量无法从
    data/uploads 的原始文件之外再生成一份不同版本 —— 万一新版本读不出来，
    至少原件还在。
    """
    chroma_dir = settings.chroma_dir
    if not chroma_dir.exists():
        return None
    backup = chroma_dir.with_name(f"{chroma_dir.name}.bak.{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.move(str(chroma_dir), str(backup))
    logger.info("旧Chroma 索引已备份到 %s", backup)
    return backup


def reingest_uploads(fresh: bool = False) -> IngestReport:
    report = IngestReport()

    if fresh:
        report.backup_dir = backup_chroma_dir()

    upload_dir: Path = settings.upload_dir
    files = sorted(
        p for p in upload_dir.iterdir()
        if p.is_file() and p.suffix.lower() in set(supported_suffixes())
    ) if upload_dir.exists() else []

    if not files:
        raise RuntimeError(
            f"{upload_dir} 下没有可入库的文件（支持 {', '.join(supported_suffixes())}）"
        )

    all_chunks: list[Document] = []

    for path in files:
        try:
            documents = load_file(path)
            chunks = split_documents(documents)
            if not chunks:
                raise ValueError("解析后没有内容（可能是扫描版 PDF 或空文件）")
            add_documents(chunks)
        except Exception as exc:  # noqa: BLE001 —— 单个文件失败不应中断整批恢复
            report.files_failed += 1
            report.failures.append((path.name, f"{type(exc).__name__}: {exc}"))
            logger.exception("入库失败：%s", path.name)
            continue

        all_chunks.extend(chunks)
        report.files_ok += 1
        report.chunks += len(chunks)
        print(f"  ✓ {path.name}（{len(chunks)} 片段）", flush=True)

    if not all_chunks:
        raise RuntimeError("没有任何片段写入成功，保留原 BM25 索引不动")

    # BM25 全量重建而不是增量 add：本次的语义就是「以 uploads 为准重建」，
    # 增量会把新切法与旧粒度再次混进同一个索引。
    index = BM25Index()
    index.build(all_chunks)
    index.save(force=True)
    logger.info("BM25 索引已重建：%d 个片段", len(all_chunks))

    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="从 uploads 重新入库")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="先把现有 Chroma 目录改名备份（索引重复或有旧格式残留时使用）",
    )
    args = parser.parse_args()

    ensure_exclusive_access("app.rag.ingest")

    print(f"扫描 {settings.upload_dir} …", flush=True)
    report = reingest_uploads(fresh=args.fresh)
    print(report.summary(), flush=True)

    if report.files_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()