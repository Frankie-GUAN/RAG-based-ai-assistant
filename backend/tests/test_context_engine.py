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
    """传入顺序即相关度顺序（reranker 已排好），截断必须从头保留。

    条目刻意长到**单条就撑爆该层**（retrieved 预算 = 1000 × 0.35 = 350 token）。
    原先用 50 条短片段时加起来还不到预算，`truncated` 是 False、50 条全保留 ——
    断言对任何实现都成立，连「反向保留（从尾巴开始留）」这种 bug 都抓不住。
    """
    retrieved = [f"排名第{i}的片段" + "填充" * 200 for i in range(50)]

    result = engine.build(system="sys", question="q", retrieved=retrieved)

    stat = next(s for s in result.stats if s.name == "retrieved")
    assert stat.truncated is True
    # 只断言「确实截断了」，不写死保留几条 —— 那取决于每条的实际 token 数与预算的
    # 整除关系，写死会随 tokenizer 或预算调整而碎
    assert stat.kept_count < stat.source_count, "必须真的截断，否则下面的断言对任何实现都成立"

    joined = "\n".join(m.content for m in result.messages)
    assert "排名第0的片段" in joined
    assert "排名第49的片段" not in joined, "反向保留（从尾巴开始留）的实现会在这里失败"


def test_oversized_summary_is_truncated_within_the_memory_budget(engine):
    """摘要（memory 层）超预算时被截断，且用量不越界。

    注意用例名不再叫「char_limit」：`context_summary_max_chars` 是**生成侧**的字数
    约束，由 Task 9 的滚动摘要消费，本引擎没有这个参数。这里断言的是引擎侧的 token
    预算截断，不是字数上限。
    """
    result = engine.build(system="sys", question="q", summary="摘要内容" * 5_000)

    memory_stat = next(s for s in result.stats if s.name == "memory")
    assert memory_stat.truncated is True
    assert memory_stat.used <= memory_stat.budget


def test_stats_cover_every_layer(engine):
    result = engine.build(system="sys", question="q", summary="s", retrieved=["r"], scratchpad="sp")

    assert {s.name for s in result.stats} == {
        "system", "memory", "history", "retrieved", "scratchpad"
    }


def test_empty_layers_are_omitted_from_messages(engine):
    """没传内容的层不该留下空标题，也不该混进 "None"。

    只断言 "None" not in contents 的话，除非有人手滑 f-string 了 None，否则永远成立 ——
    空层的【长期记忆】/【参考资料】标题照样会拼进 prompt，它抓不住。
    """
    result = engine.build(system="sys", question="q")

    contents = "\n".join(m.content for m in result.messages)
    assert "None" not in contents
    for header in ("【长期记忆】", "【参考资料】", "【已获取的中间结果】"):
        assert header not in contents, f"{header} 层为空时不该出现标题"
