"""精排模型「不可用」时的冷却行为测试。

不加载真实模型：Reranker 构造被替换成可控替身，因此不需要 requires_model 标记。
"""
import pytest

from app.rag import reranker as reranker_mod


@pytest.fixture(autouse=True)
def _clean_state():
    reranker_mod.reset_reranker_state()
    yield
    reranker_mod.reset_reranker_state()


def _boom_on_init(monkeypatch, error=None):
    def _factory(*_args, **_kwargs):
        raise error or OSError("模拟模型不存在")

    monkeypatch.setattr(reranker_mod, "Reranker", _factory)


def _fake_ok(monkeypatch):
    class _Fake:
        pass

    monkeypatch.setattr(reranker_mod, "Reranker", _Fake)


def test_successful_load_is_a_singleton(monkeypatch):
    _fake_ok(monkeypatch)

    first = reranker_mod.get_reranker()
    second = reranker_mod.get_reranker()

    assert first is second


def test_failure_raises_and_then_enters_cooldown(monkeypatch):
    """第一次失败抛错，随后进入冷却 —— 关键是别让每次检索都重试下载。"""
    attempts = []

    def _factory(*_args, **_kwargs):
        attempts.append(1)
        raise OSError("模型不存在")

    monkeypatch.setattr(reranker_mod, "Reranker", _factory)

    with pytest.raises(OSError):
        reranker_mod.get_reranker()

    # 冷却期内不再尝试构造
    with pytest.raises(RuntimeError, match="精排模型不可用"):
        reranker_mod.get_reranker()
    with pytest.raises(RuntimeError, match="精排模型不可用"):
        reranker_mod.get_reranker()

    assert len(attempts) == 1, "冷却期内只应构造过一次"


def test_cooldown_expires_and_retries(monkeypatch):
    attempts = []

    def _factory(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("首次失败")
        return object()

    monkeypatch.setattr(reranker_mod, "Reranker", _factory)
    monkeypatch.setattr(reranker_mod, "_RETRY_INTERVAL_SECONDS", 0.01)

    with pytest.raises(OSError):
        reranker_mod.get_reranker()

    import time
    time.sleep(0.02)

    assert reranker_mod.get_reranker() is not None
    assert len(attempts) == 2, "冷却期过后应当重试一次"


def test_recovers_after_a_transient_failure(monkeypatch):
    """瞬时故障（磁盘 momentarily 不可写）之后模型应当能用。"""
    state = {"fail": True}

    def _factory(*_args, **_kwargs):
        if state["fail"]:
            state["fail"] = False
            raise OSError("瞬时故障")
        return object()

    monkeypatch.setattr(reranker_mod, "Reranker", _factory)
    monkeypatch.setattr(reranker_mod, "_RETRY_INTERVAL_SECONDS", 0.01)

    with pytest.raises(OSError):
        reranker_mod.get_reranker()

    import time
    time.sleep(0.02)

    assert reranker_mod.get_reranker() is not None
    # 成功后冷却标记被清掉
    assert reranker_mod._unavailable_until == 0.0
    assert reranker_mod._unavailable_reason is None


def test_error_message_points_at_the_two_ways_out(monkeypatch):
    """报错要说清怎么解决：放模型，或关掉精排。"""
    _boom_on_init(monkeypatch)

    with pytest.raises(OSError):
        reranker_mod.get_reranker()

    with pytest.raises(RuntimeError) as exc:
        reranker_mod.get_reranker()

    message = str(exc.value)
    assert "enable_reranker" in message
    assert "reranker" in message.lower()