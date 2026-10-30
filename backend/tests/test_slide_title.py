"""幻灯片标题提取的单元测试。

**为什么这些用例都是真实数据。** 每一个「不该是标题」的样本都来自
本机对 9 份课件 PDF 的实测（见 docs/tech-notes/verified-facts.md §8.4），
不是编造的。标题提取的失败模式很隐蔽——垃圾标题不会让程序报错，
只会安静地污染向量——所以必须把见过的每一种失败都钉住。

**核心原则**：宁可没有标题，也不能有坏标题。
"""
from app.rag.slide_title import (
    build_section_path,
    extract_page_title,
    looks_like_title,
    render_title_prefix,
)


class TestRealTitles:
    """实测中确认是真标题的样本，必须全部被识别。"""

    def test_english_section_titles(self):
        for title in [
            "Pointer Analysis",
            "Rules",
            "Worklist (WL)",
            "Main Algorithm",
            "Differential Propagation",
            "Pointer Flow Graph (PFG)",
            "Pointer Flow Graph: An Example",
            "Pointer-Affecting Statements",
            "Domain and Notations",
            "Reaching Definitions",
            "Understanding Available Expressions Analysis",
            "Fixed-Point Theorem (Least Fixed Point)",
            "Complete Lattice",
            "Product Lattice",
            "Monotonicity",
            "Constant Propagation – Lattice",
            "Worklist Algorithm",
            "Method Calls (Invocations) in Java",
            "Call Graph Construction for OOPLs",
            "CHA: An Example",
            "Class Hierarchy Analysis* (CHA)",
            "Allocation-Site Abstraction",
            "Flow Sensitivity",
            "Key Factors in Pointer Analysis",
            "Rule: Load",
            "Normal edges",
            "Interprocedural Control-Flow Graph",
            "Applications",
        ]:
            assert looks_like_title(title), f"真标题被误拒：{title!r}"

    def test_chinese_titles(self):
        for title in ["程序分析", "算法理解", "课程回顾"]:
            assert looks_like_title(title), f"中文标题被误拒：{title!r}"

    def test_titles_with_prepositions_are_kept(self):
        """含介词的真标题不能被句子规则误杀。

        这是「按句子特征收紧」时最容易踩的坑——
        'Pointer Flow Graph: An Example'、'Call Graph Construction for OOPLs'
        都含介词。
        """
        for title in [
            "Pointer Flow Graph: An Example",
            "Call Graph Construction for OOPLs",
            "Method Calls (Invocations) in Java",
            "Pointer Analysis in the Presence of",
            "Rule: Call",
            "The X You Need To Understand in This Lecture",
        ]:
            assert looks_like_title(title), f"含介词的真标题被误拒：{title!r}"

    def test_numbered_section_heading(self):
        """带编号的目录项也是标题。"""
        assert looks_like_title("1. Overview of Data Flow Analysis")


class TestGarbageMustBeRejected:
    """实测中确认是垃圾的样本，必须全部被拒。"""

    def test_math_symbol_titles(self):
        """花体字母标题——来自 Week 10 p17~p26。

        这些是公式 Solve(S) 被 PDF 提取器渲染成了 Unicode 花体字符。
        """
        for title in ["Solve(𝑆)", "Solve(𝑚𝑒𝑛𝑡𝑟𝑦)", "AddReachable(𝑚)", "ProcessCall(𝑥, 𝑜𝑖)"]:
            assert not looks_like_title(title), f"公式标题未被拒：{title!r}"

    def test_nfkc_must_not_launder_math_title(self):
        """**回归用例：NFKC 归一化不能把花体字母「洗白」。**

        NFKC 会把 𝑆 规整成普通 S，于是 'Solve(𝑆)' 变成 'Solve(S)'，
        数学字符检测就拦不住它了。
        这条用例钉住的是**检测顺序**：必须先判原串、再归一化。
        """
        assert extract_page_title("Solve(𝑆)\n\nWL = []") == ""
        assert extract_page_title("Solve(𝑆)\n正文内容在这里") == ""

    def test_graphic_residue(self):
        """图形残留文本——实测的多例。"""
        for title in [
            "remove                        from WLSolve(𝑆)",
            "class T … {   …",
            "class B extends A {",
            "kill b=…",
            "…",
            "S…",
            "①",
            "⊥",
        ]:
            assert not looks_like_title(title), f"图形残留未被拒：{title!r}"

    def test_code_lines(self):
        """代码/赋值语句。"""
        for title in [
            "x = p + 1",
            "v = 3 P",
            "OUT[B] = genB U (IN[B] - killB)",
            "1 class A {",
            "{ A.main() }",
            "0000 0000",
            "1011 1100 1011 1100",
        ]:
            assert not looks_like_title(title), f"代码行未被拒：{title!r}"

    def test_sentences_are_not_titles(self):
        """句子不是标题。"""
        for text in [
            "x op y by the result of its last evaluation",
            "produces a solution to a data flow analysis",
            "- A definition of a variable v is a statement that assigns a value to v",
            "This statement “generates” a definition D of variable v",
        ]:
            assert not looks_like_title(text), f"句子未被拒：{text!r}"

    def test_empty_and_too_short(self):
        assert not looks_like_title("")
        assert not looks_like_title("A")
        assert not looks_like_title("ab")

    def test_too_long(self):
        assert not looks_like_title("x" * 60)


class TestExtractPageTitle:
    def test_title_on_first_line(self):
        page = "Worklist (WL)\n\nWL = []\nPFG = {}\nforeach i: ..."
        assert extract_page_title(page) == "Worklist (WL)"

    def test_title_on_second_line(self):
        """标题偶尔被图形元素挤到第二行。"""
        page = "◦◦◦◦\nPointer Flow Graph (PFG)\n\nPFG = {}"
        assert extract_page_title(page) == "Pointer Flow Graph (PFG)"

    def test_no_title_returns_empty(self):
        """没有标题就返回空串——**调用方据此跳过补全**。"""
        page = "x = p + 1\ny = q + 2"
        assert extract_page_title(page) == ""

    def test_empty_page(self):
        assert extract_page_title("") == ""

    def test_never_returns_placeholder(self):
        """绝不能返回占位符——那会污染所有向量。"""
        for page in ["x = p + 1", "⊥", "…", ""]:
            result = extract_page_title(page)
            assert result == "", f"返回了占位符：{result!r}"

    def test_cross_line_merge_is_disabled(self):
        """**回归用例：跨行标题合并已弃用。**

        实测 20.1% 的页是「标题 + 第二行」结构，但第二行无法可靠区分：

            'Pointer Analysis' + 'Foundations (I)'   → 真标题延续
            'Rules'            + 'Kind Statement Rule' → 表头
            'Review'           + '1. Concerned Statements' → 目录项
            '算法理解'          + '可以把指针分析这个过程…'   → 中文正文

        后三类都能通过标题规则，启发式区分不了。误合并 119/890 页的代价
        （片段标题带错误语义）大于合并的收益，所以只取首行。

        这条用例钉住这个取舍，避免后来者重做一遍实验。
        """
        # 表头：不该被合并
        assert extract_page_title("Rules\nKind Statement Rule\n\n正文") == "Rules"
        # 目录项：不该被合并
        assert extract_page_title("Review\n1. Concerned Statements\n\n正文") == "Review"
        # 中文正文：不该被合并
        assert extract_page_title("算法理解\n可以把指针分析想象成物流网络\n\n正文") == "算法理解"

    def test_skips_leading_blank_lines(self):
        page = "\n\n  \nRules\n\ncontent"
        assert extract_page_title(page) == "Rules"


class TestSectionPath:
    def test_document_title_is_top_level(self):
        path = build_section_path("Worklist (WL)", "Pointer Analysis: Foundations")
        assert path == ["Pointer Analysis: Foundations", "Worklist (WL)"]

    def test_duplicate_is_not_repeated(self):
        """页标题与文档标题相同时不重复添加。"""
        path = build_section_path("Interprocedural Analysis", "Interprocedural Analysis")
        assert path == ["Interprocedural Analysis"]

    def test_empty_document_title(self):
        path = build_section_path("Rules", "")
        assert path == ["Rules"]

    def test_empty_page_title(self):
        path = build_section_path("", "Pointer Analysis")
        assert path == ["Pointer Analysis"]


class TestRenderTitlePrefix:
    def test_renders_title_with_newline(self):
        """用「标题 + 换行」而不是标记符号——嵌入模型对自然文本建模更好。"""
        assert render_title_prefix("Worklist (WL)") == "Worklist (WL)"

    def test_includes_section_path(self):
        result = render_title_prefix("Worklist (WL)", ["Pointer Analysis: Foundations"])
        assert "Worklist (WL)" in result
        assert "Pointer Analysis: Foundations" in result

    def test_empty_title_returns_empty(self):
        assert render_title_prefix("") == ""
