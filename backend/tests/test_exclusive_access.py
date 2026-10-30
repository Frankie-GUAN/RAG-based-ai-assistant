"""独占检查的测试。

这个检查存在的理由是实测踩过的坑：服务进程持有 data/chroma 时，另一个进程再打开
同一目录，索引会在几秒内变成不可读（`Error loading hnsw index`），只能整库重建。
因此「服务是否在跑」必须是维护命令的第一道闸门。
"""
import pytest

from app.rag import exclusive


def test_server_is_running_detects_a_bound_port():
    """真的起一个监听端口，再让检查去发现它。"""
    import socket
    import threading

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    try:
        assert exclusive.server_is_running(port) is True
    finally:
        server.close()

    # 关掉之后（端口可能处于 TIME_WAIT，但不再接受连接）
    assert exclusive.server_is_running(port) is False


def test_ensure_exclusive_access_blocks_when_a_server_listens():
    import socket

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    try:
        with pytest.raises(RuntimeError) as exc:
            exclusive.ensure_exclusive_access("app.rag.ingest", port=port)

        message = str(exc.value)
        assert "app.rag.ingest" in message
        # 报错要给出可执行的下一步，而不是只说「失败了」
        assert "停止" in message
        assert "损坏" in message
    finally:
        server.close()


def test_ensure_exclusive_access_is_a_noop_when_nothing_listens(monkeypatch):
    """没服务在跑时必须放行 —— 否则维护命令永远执行不了。"""
    monkeypatch.setattr(exclusive, "server_is_running", lambda *a, **k: False)

    exclusive.ensure_exclusive_access("app.rag.search")   # 不抛异常即通过


def test_all_maintenance_entry_points_call_the_guard():
    """三个维护 CLI 都必须过这道闸门，漏一个就等于留了一条踩雷的路。"""
    import inspect

    import app.rag.bm25_index as bm25_mod
    import app.rag.ingest as ingest_mod
    import app.rag.search as search_mod

    for module in (ingest_mod, search_mod, bm25_mod):
        source = inspect.getsource(module)
        assert "ensure_exclusive_access" in source, f"{module.__name__} 缺少独占检查"
