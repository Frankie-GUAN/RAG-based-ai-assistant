"""评测链路不阻塞事件循环的回归测试。

旧实现是async 端点直接调用同步的 run_evaluation（含 time.sleep(3)），
3 条 query 就能把整个服务器冻住 30~60 秒。这条测试必须**能真失败**：
把 run_evaluation 改回同步形式，它会立刻变红。
"""
import asyncio

import httpx
import pytest

from app.api import evaluation as eval_api
from app.db.session import get_db
from app.main import app
from app.services import evaluation_service as eval_svc


class _StubSession:
    """评测端点只用到 add/commit。"""

    def add(self, obj):
        pass

    def commit(self):
        pass


@pytest.mark.anyio
async def test_evaluation_does_not_block_other_requests(monkeypatch):
    started = asyncio.Event()

    async def _slow_run_evaluation(queries, tools):
        started.set()
        await asyncio.sleep(0.5)                  # 模拟慢评测
        return [
            {
                "query": q, "answer": "", "context": "",
                "faithfulness": 0.0, "answer_relevancy": 0.0, "error": None,
            }
            for q in queries
        ]

    # 注意：api 层是 `from ... import run_evaluation`，名字已绑定到 eval_api 的
    # 命名空间，只patch eval_svc 不生效，两处都要打补丁。
    monkeypatch.setattr(eval_svc, "run_evaluation", _slow_run_evaluation)
    monkeypatch.setattr(eval_api, "run_evaluation", _slow_run_evaluation)

    # ASGITransport 不跑 lifespan，因此 app.state.mcp_client 不存在，需要手工补上
    monkeypatch.setattr(app.state, "mcp_client", None, raising=False)
    app.dependency_overrides[get_db] = lambda: _StubSession()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            eval_task = asyncio.create_task(
                client.post("/api/evaluation/run", json={"queries": ["q1"]})
            )
            await asyncio.wait_for(started.wait(), timeout=2)

            # 评测还在跑，health 必须立刻回来
            health = await asyncio.wait_for(client.get("/api/health"), timeout=0.3)

            assert health.status_code == 200
            assert not eval_task.done()           # 评测尚未结束 —— health 却已经返回了
            await eval_task
    finally:
        app.dependency_overrides.pop(get_db, None)


# --- 打分失败不再计0 分 ---

@pytest.mark.anyio
async def test_unparseable_judge_output_is_recorded_as_missing(monkeypatch):
    """判官输出「抱歉，我无法评分」时，必须记为缺失而不是 0 分。

    0 分是一个有含义的结论。把「判官这次没能给数字」也记成 0，等于伪造出一条
    质量问题：统计里混进了本该缺席的数据点，平均分被拉低，而真正该修的东西
    （输出格式不稳定）反而被藏了起来。
    """
    from langchain_core.runnables import RunnableLambda

    from app.services import evaluation_service as svc

    monkeypatch.setattr(svc, "get_chat_model", lambda: RunnableLambda(lambda _p: "抱歉，我无法评分"))

    scores = await svc._evaluate_single("问题", "答案", "上下文")

    assert scores["faithfulness"] is None
    assert scores["answer_relevancy"] is None


@pytest.mark.anyio
async def test_numeric_judge_output_still_parses(monkeypatch):
    """锚点 prompt 的结尾写着「只输出一个数字」，但模型仍可能带上「0.75 分」之类的话。"""
    from langchain_core.runnables import RunnableLambda

    from app.services import evaluation_service as svc

    monkeypatch.setattr(svc, "get_chat_model", lambda: RunnableLambda(lambda _p: "0.75"))

    scores = await svc._evaluate_single("问题", "答案", "上下文")

    assert scores["faithfulness"] == 0.75
    assert scores["answer_relevancy"] == 0.75


@pytest.mark.anyio
async def test_judge_output_with_surrounding_words_is_rejected_not_zeroed(monkeypatch):
    """「0.75 分」这类输出应当被记为缺失，而不是变成 0 分混进统计。"""
    from langchain_core.runnables import RunnableLambda

    from app.services import evaluation_service as svc

    monkeypatch.setattr(svc, "get_chat_model", lambda: RunnableLambda(lambda _p: "0.75 分（良好）"))

    scores = await svc._evaluate_single("问题", "答案", "上下文")

    assert scores["faithfulness"] is None


def test_judge_prompts_carry_scoring_anchors():
    """没有标尺的评分没有信息量：所有 query 都会挤在 0.85~1.0，看不出改动好坏。"""
    from app.services import evaluation_service as svc
    import inspect

    source = inspect.getsource(svc._evaluate_single)

    assert "1.0 =" in source and "0.0 =" in source, "缺少评分锚点"
    assert "只输出一个数字" in source