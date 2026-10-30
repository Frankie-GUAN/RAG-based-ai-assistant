"""独占检查：维护类命令必须独占向量库目录。

**实测（2026-10-14）**：服务进程持有 `data/chroma` 时，另一个进程再打开同一个目录，
索引会在几秒内变成不可读 ——

    chromadb.errors.InternalError: Error executing plan: Error sending backfill
    request to compactor: Error constructing hnsw segment reader: Error loading hnsw index

磁盘上表现为段目录只剩 `index_metadata.pickle`（HNSW 文件消失），而 sqlite 里的向量
行数不变（903 条都在）。此后**任何**新进程都读不了这个索引，检索彻底不可用，
只能靠 `app.rag.ingest --fresh` 从 `data/uploads` 重建（约 6 分钟）。

值得注意的是「删除文档」曾被误判为元凶：两次损坏都发生在用 CLI 读取之后，
而删除只是恰好在前面。隔离复现（原生删除、langchain 包装层、20/40/60/903 条各规模、
优雅停止后重启）全部无法复现删除路径本身的问题 —— 真正稳定的共同因素只有
「服务运行期间有第二个进程打开该目录」。

Chroma 的 `PersistentClient` 本就是单进程独占设计，`app.rag.search` /
`app.rag.ingest` / `app.rag.bm25_index rebuild` 都是独立进程，因此这里显式拦住：
破坏索引的代价（整库重建）远高于「先停服务」。
"""
from __future__ import annotations

import socket

# 后端默认监听端口。uvicorn --port 改了端口时，这条检查会漏判，因此错误信息里
# 要提醒用户：换过端口的话请自行确认服务已停止。
DEFAULT_SERVER_PORT = 8000


def server_is_running(port: int = DEFAULT_SERVER_PORT, host: str = "127.0.0.1") -> bool:
    """本机该端口是否有服务在监听。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def ensure_exclusive_access(action: str, port: int = DEFAULT_SERVER_PORT) -> None:
    """服务在跑就抛 RuntimeError —— 宁可拦住，也不要事后重建整库。"""
    if server_is_running(port):
        raise RuntimeError(
            f"检测到 {port} 端口有服务在运行，已中止「{action}」。\n\n"
            f"服务进程持有 data/chroma 时，另一个进程再打开同一目录会让索引损坏"
            f"（之后只能重新入库恢复）。请先停止后端服务再执行维护命令：\n\n"
            f"    # 停掉 uvicorn 之后重试\n"
            f"如果你是用非默认端口启动的服务，这条检查看不到它，请自行确认服务已停止。"
        )
