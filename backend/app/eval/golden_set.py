"""金标评测集（golden set）。

**这是什么。** 一组人工标注的 (query, 相关片段) 对，用来测量检索质量。
没有它，「我的 RAG 效果好不好」只能靠感觉；有了它，任何改动都能被量化。

**标注口径。** 相关性是二元的：某个片段是否包含**足以回答该 query 的内容**。
不相关不等于「话题相近」——一段讲数据流分析的文字，对「指针分析的作用域」
问题就是不相关，哪怕它们同属程序分析课程。

**标注粒度用 (文档前缀, 页码) 而不是精确 chunk_id。**
原因：chunk 边界随切分参数（chunk_size/overlap）变化，改一次切分参数
精确 chunk_id 就全失效了，标注会在两个版本之间失去可比性。
而「答案在 Week 8 的第 37~40 页」这个判断在切分参数变化后依然成立。

评测时按 (文档, 页码) 反查命中的 chunk：落在标注页码范围内的片段都算命中。
这个口径比精确 chunk_id 宽松一些，但**跨版本可比**——而可比性是评测集的第一属性。

**页码怎么来的（重要）。** 全部页码都在本机对 Chroma 库逐条核对过：
先跑一遍检索看实际召回哪些页，再打开这些页确认内容确实是该 query 的答案，
最后才写入本文件。

**为什么不按章节范围标。** 最初的版本把标注写成「整个章节的页码区间」
（例如「可用表达式 = Week 4 的 100~112 页」），结果 Recall 全线接近 0。
两个原因：
1. 课件有大量**重复页**（同一段内容被复制到多处，精确重复占 1.6%，
   页眉页脚造成的视觉重复更多），章节的真实边界与直觉不符；
2. 目录页与正文页的页码**不对应**——Week 10 的 p3 是目录，
   而 Concerned Statements 的正文从 p4 才开始。

所以现在的标注是**精确的若干页**（通常 3~10 页），只覆盖确实能回答该问题的内容。
宽区间会让「召回到了但排得很靠后」被算成完全命中，指标失真；
而猜错的页码则相反，会让本该命中的结果算成未命中。两种误差都会毁掉对照表。

**如何扩充。** 见 docs/plans/2026-10-30-job-alignment-redesign.md §3.1。
新增标注的流程：跑检索 → 看召回页 → 打开核对内容 → 写页码 → 复跑确认命中。
每条标注都应附一句话说明「为什么这一段能回答这个问题」——
那是标注质量的最后一道防线。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenCase:
    """一条评测用例。

    `relevant_pages` 是该 query 答案所在的页码集合（同一文档内）。
    `relevant_sources` 是相关文档名的前缀集合——用前缀而非全名，
    是因为 source 字段带 uuid 后缀（见 verified-facts.md §6.7）。
    """

    query: str
    relevant_sources: tuple[str, ...]
    relevant_pages: frozenset[int]
    note: str = ""
    category: str = "general"


# 文档名前缀：与 metadata["source"] 的 uuid 前缀部分一致
_W4 = "Week 4-Data Flow Analysis- Applications II"
_W5 = "Week 5-Data Flow Analysis- Foundations I"
_W6 = "Week 6-Data Flow Analysis- Foundations II"
_W7 = "Week 7-Interprocedural Analysis"
_W8 = "Week 8-Pointer Analysis"
_W9 = "Week 9&10-Pointer Analysis Foundations I"
_W10 = "Week 10-Pointer Analysis Foundations I"
_W11 = "Week 11-Pointer Analysis Foundations II"
_BREW = "酿酒工艺学"


def _span(start: int, end: int) -> frozenset[int]:
    """闭区间页码集合。"""
    return frozenset(range(start, end + 1))


GOLDEN_SET: list[GoldenCase] = [
    # --- 数据流分析三类经典分析（Week 4 = Applications II）---
    GoldenCase(
        query="到达定义分析的定义是什么",
        relevant_sources=(_W4,),
        relevant_pages=_span(6, 20),
        note="Week 4 p6 已核对为 Reaching Definitions 定义页（d reaches q 的定义），"
             "p7 为 use/def 表述，p9~12 为方程与例子",
        category="dfa_reaching",
    ),
    GoldenCase(
        query="活跃变量分析怎么定义",
        relevant_sources=(_W4,),
        relevant_pages=_span(100, 130),
        note="Live Variables Analysis 章节正文（p118~121 为定义与方程）",
        category="dfa_live",
    ),
    GoldenCase(
        query="可用表达式分析的定义",
        relevant_sources=(_W4,),
        relevant_pages=_span(113, 125),
        note="p114~122 已核对为 Available Expressions Analysis 正文"
             "（含 a = x op y 的 IN/OUT 转移）",
        category="dfa_available",
    ),

    # --- 格与偏序（Week 5 Foundations）---
    GoldenCase(
        query="什么是格和偏序",
        relevant_sources=(_W5,),
        relevant_pages=_span(78, 86),
        note="p78~p86 为标题恰好是「Lattice」的页（用 page_title 精确匹配得到）；"
             "p78 已核对为 poset / ⊔ / ⊓ 的定义页",
        category="lattice",
    ),
    GoldenCase(
        query="单调性和不动点定理",
        relevant_sources=(_W5,),
        relevant_pages=_span(118, 139),
        note="p118~p123 标题为「Monotonicity」，p131~p138 标题为"
             "「Fixed-Point Theorem (Least Fixed Point)」（page_title 精确匹配）",
        category="fixed_point",
    ),

    # --- 常量传播与 may/must（Week 6 Foundations II）---
    GoldenCase(
        query="may 分析和 must 分析的区别",
        relevant_sources=(_W6,),
        relevant_pages=_span(30, 40),
        note="p33 已核对为 May and Must Analyses, a Lattice View 标题页",
        category="may_must",
    ),
    GoldenCase(
        query="常量传播分析怎么做",
        relevant_sources=(_W6,),
        # p87~p92 —— 用 page_title 精确匹配「Constant Propagation」得到
        # （标题补全后这些页都带上了该标题，是最可靠的结构化信号）
        relevant_pages=_span(86, 105),
        note="Week 6 的 Constant Propagation 章节（p87 起为定义，"
             "p92 起讲 Lattice，p101 起讲 Transfer Function）",
        category="constant_prop",
    ),

    # --- 过程间分析（Week 7）---
    GoldenCase(
        query="调用图怎么构建",
        relevant_sources=(_W7,),
        relevant_pages=_span(58, 68),
        note="p60 已核对为 Build call graph for whole program via CHA，"
             "p61~65 为算法步骤（WL / CG 伪代码）",
        category="call_graph",
    ),
    GoldenCase(
        query="过程间控制流图 ICFG 是什么",
        relevant_sources=(_W7,),
        relevant_pages=_span(82, 92),
        note="p83 已核对为 ICFG 定义页（CFG 表示单个方法，ICFG 表示整个程序）",
        category="icfg",
    ),

    # --- 指针分析 ---
    GoldenCase(
        query="指针分析关心哪些语句",
        relevant_sources=(_W10, _W9),
        relevant_pages=_span(3, 20),
        note="Concerned Statements 是 Rules 一节首主题；"
             "Week 10 p3 为目录，正文自 p4 起；Week 9&10 p3~18 亦有",
        category="pointer_rules",
    ),
    GoldenCase(
        query="指针分析的别名集是什么",
        relevant_sources=(_W8, _W9, _W10),
        relevant_pages=_span(58, 70),
        note="Week 8 p59~67 为 alias 相关示例与定义（p63 命中算法解读），"
             "Week 10 p63 为别名集核心页",
        category="pointer_alias",
    ),
    GoldenCase(
        query="带方法调用的指针分析",
        relevant_sources=(_W11,),
        relevant_pages=_span(5, 30),
        note="Week 11 p5~7 为 Pointer Analysis with Method Calls 起始页",
        category="pointer_method_calls",
    ),
    GoldenCase(
        query="worklist 算法如何做指针分析",
        relevant_sources=(_W10, _W9),
        relevant_pages=_span(55, 70),
        note="Week 9&10 p56~65 与 Week 10 p63 为 Solve(S)/WL 传播算法",
        category="pointer_algo",
    ),

    # --- 应用 ---
    GoldenCase(
        query="活跃变量信息能用来做什么",
        relevant_sources=(_W4,),
        relevant_pages=frozenset({2, 3}),
        note="p3 已核对：live variables 可用于 register allocation",
        category="dfa_applications",
    ),

    # --- 跨文档 ---
    GoldenCase(
        query="中国啤酒工业的发展经历了哪几个阶段",
        relevant_sources=(_BREW,),
        relevant_pages=_span(0, 3),
        note="酿酒工艺学文献的中国啤酒工业历史脉络",
        category="brewery",
    ),
]
