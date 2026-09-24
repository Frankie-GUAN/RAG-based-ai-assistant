import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.context.engine import ContextBudget, ContextEngine
from app.context.tokens import HeuristicTokenCounter


@pytest.fixture
def counter():
    return HeuristicTokenCounter()


@pytest.fixture
def engine(counter):
    # 总预算刻意调小，便于断言截断行为
    return ContextEngine(
        budget=ContextBudget(total_tokens=1_000),
        counter=counter,
    )


def test_counter_is_monotonic_and_positive(counter):
    assert counter.count("") == 0 or counter.count("") == 1
    assert counter.count("短") < counter.count("这是一段明显更长的中文文本内容")


def test_counter_treats_cjk_as_denser_than_ascii(counter):
    """同样字符数下中文占更多 token。"""
    cjk = "中" * 60
    ascii_text = "a" * 60

    assert counter.count(cjk) > counter.count(ascii_text)


def test_build_puts_system_message_first(engine):
    result = engine.build(system="你是助手", question="你好")

    assert isinstance(result.messages[0], SystemMessage)
    assert "你是助手" in result.messages[0].content


def test_build_ends_with_the_user_question(engine):
    result = engine.build(system="sys", question="劳动合同如何解除？")

    assert isinstance(result.messages[-1], HumanMessage)
    assert "劳动合同如何解除？" in result.messages[-1].content


def test_history_is_mapped_to_message_roles(engine):
    result = engine.build(
        system="sys",
        question="继续",
        history=[HumanMessage(content="问题一"), AIMessage(content="回答一")],
    )

    roles = [type(m).__name__ for m in result.messages]
    assert "HumanMessage" in roles
    assert "AIMessage" in roles


def test_each_layer_respects_its_budget(engine):
    """retrieved 层给 50 条超长片段，必须被截到预算内。"""
    huge = ["片段" * 500] * 50

    result = engine.build(system="sys", question="q", retrieved=huge)

    retrieved_stat = next(s for s in result.stats if s.name == "retrieved")
    assert retrieved_stat.used <= retrieved_stat.budget
    assert retrieved_stat.truncated is True
    assert retrieved_stat.kept_count < retrieved_stat.source_count


def test_retrieved_layer_keeps_highest_ranked_first(engine):
    """传入顺序即相关度顺序（reranker 已排好），截断必须从头保留。"""
    retrieved = [f"排名第{i}的片段" for i in range(50)]

    result = engine.build(system="sys", question="q", retrieved=retrieved)

    joined = "\n".join(m.content for m in result.messages)
    assert "排名第0的片段" in joined


def test_summary_is_truncated_to_configured_char_limit(engine):
    result = engine.build(system="sys", question="q", summary="摘要内容" * 5_000)

    memory_stat = next(s for s in result.stats if s.name == "memory")
    assert memory_stat.truncated is True


def test_stats_cover_every_layer(engine):
    result = engine.build(system="sys", question="q", summary="s", retrieved=["r"], scratchpad="sp")

    assert {s.name for s in result.stats} == {
        "system", "memory", "history", "retrieved", "scratchpad"
    }


def test_empty_layers_are_omitted_from_messages(engine):
    result = engine.build(system="sys", question="q")

    contents = "\n".join(m.content for m in result.messages)
    assert "None" not in contents
