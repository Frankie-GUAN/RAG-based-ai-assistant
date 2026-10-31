"""幻灯片标题提取与上下文补全。

**背景。** 这份语料是 PPT 导出的 PDF，一页一张幻灯片，每页正文的第一行
就是该幻灯片的标题（实测：Week 10 的 p4=「Pointer-Affecting Statements」、
p6=「Rules」、p15=「Worklist (WL)」）。但**这个「第一行就是标题」的规律
不能无条件信任**——实测约 40% 的页首行是图形残留或公式片段：

    'Solve(𝑆)'                     ← Unicode 花体数学字母
    'remove      from WLSolve(𝑆)'  ← 图形残留文本
    'class T … {   …'              ← 代码片段 + 省略号
    'x = p + 1'                    ← 赋值语句
    '⊥'                            ← 孤立符号

把这种字符串写进 chunk 会**主动污染向量**——它既不携带章节语义，
又让片段看起来像代码而非标题。所以这里的策略是**保守**：

> 宁可没有标题，也不能有坏标题。

**判定规则**（按顺序，先命中先返回）：

1. 长度在 3~50 之间 —— 标题是短语，不是句子也不是单个符号
2. 不含 Unicode 花体/数学字母（U+1D400–U+1D7FF 等）—— 它们只出现在公式里
3. 不以省略号、`;`、`{`、`}`、`=` 结尾 —— 那是图形残留或代码的特征
4. 字母/汉字占比 ≥ 0.5 —— 代码行 `x = p + 1` 的字母占比只有 0.5 但含 `=`
5. 词密度检查：空格分隔的片段中，至少一半要含字母
   （排除 `0011 0110` 这种二进制串、`kill b=…` 这种残留）

不满足时**向后看一行**：幻灯片标题偶尔被图形元素挤到第二行。
若第二行符合规则则用之，否则这一页**不写标题**（保持原样，不写空字符串）。

**写进哪里。** 两处，作用不同：

- **chunk 文本**：标题进的是向量本身，query 的词面能直接对上。
  这是「检索得到」与「检索得准」之间的差别。
- **metadata**：`page_title` / `section_path`，供引用溯源与前端展示用，
  **不参与相似度计算**。

**为什么不做 LLM 补全。** Anthropic 的 Contextual Retrieval 是让 LLM 为每块
生成一段文档级上下文，效果更好，但**每块一次 LLM 调用**——903 个片段就是 903 次调用。
这里的标题是规则可提取的结构化信息，用规则解决更划算。
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# Unicode 花体字母区段（𝑆 𝑅 𝑚𝑒𝑛𝑡𝑟𝑦 等）。这些字符只出现在数学公式里，
# 出现在「标题」位置几乎可以肯定是把公式的第一行误当成了标题。
_MATH_ALPHANUM_RANGES = (
    (0x1D400, 0x1D7FF),   # Mathematical Alphanumeric Symbols
    (0x1F300, 0x1F5FF),   # Supplemental Symbols and Pictographs
    (0x1F600, 0x1F64F),   # Emoticons
    (0x2100, 0x214F),     # Letterlike Symbols（ℳ ℛ 等）
    (0x2200, 0x22FF),     # Mathematical Operators（⊥ ⊑ 等）
)

# 标题里允许出现的结尾标点
_ALLOWED_TRAILING = (")", ")", "）", ".", "。", ":", "：", "、", "?", "？", "]", "］")

# 明确排除的结尾：这些字符出现说明这一行是代码或图形残留
_FORBIDDEN_TRAILING = (";", "{", "}", "=", "…", "...", ",", "，")

_MIN_TITLE_LEN = 3
_MAX_TITLE_LEN = 50

# 代码/赋值符号。出现在行内任意位置都说明这是代码而非标题：
# 'v = 3 P'（实测 Week 4 p3）、'x op y by ...' 这类。
# 不含 '=' '<' '>' —— 它们在真标题里也可能出现（如 'Pointer Flow Graph (PFG)'）。
_CODE_SIGNAL = re.compile(r"[=;{}]|=>|->|\|\||&&|[<>]")


def _has_math_chars(text: str) -> bool:
    """是否含 Unicode 花体字母 / 数学符号。"""
    for ch in text:
        code = ord(ch)
        for lo, hi in _MATH_ALPHANUM_RANGES:
            if lo <= code <= hi:
                return True
    return False


def _is_cjk(ch: str) -> bool:
    """是否中日韩统一表意文字（含扩展区）。"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2A6DF
    )


def _letter_ratio(text: str) -> float:
    """字母与汉字占「非空白字符」的比例。

    `x = p + 1` → 字母 x,p 占 5 个非空白字符中的 2 个 → 0.4
    `Rules` → 1.0
    """
    meaningful = [c for c in text if not c.isspace()]
    if not meaningful:
        return 0.0
    letters = sum(1 for c in meaningful if c.isalpha() or _is_cjk(c))
    return letters / len(meaningful)


def _word_density_ok(text: str) -> bool:
    """空格分隔的片段里，至少一半要含字母。

    排除 `0011 0110`（二进制）、`kill b=…`（残留）这类
    「看起来像词但其实不是」的串。
    """
    parts = [p for p in text.split() if p]
    if len(parts) <= 1:
        # 单个片段：交给字母占比规则判断
        return True
    good = sum(1 for p in parts if any(c.isalpha() or _is_cjk(c) for c in p))
    return good * 2 >= len(parts)


def _looks_like_formula(text: str) -> bool:
    """是否是公式而非标题。

    两个信号：

    1. **Unicode 花体字母**（𝑆 𝑅 𝑚𝑒𝑛𝑡𝑟𝑦）——只在公式里出现。
    2. **归一化后的花体残留**——NFKC 会把 𝑆 规整成普通 S，
       于是 'Solve(𝑆)' 变成 'Solve(S)'，信号 1 就失效了。
       但归一化后仍保留了**数学斜体 S / R / m 这类单字母紧跟括号**的形态，
       标题不会长这样（真标题里的括号内容是词组，如 'Worklist (WL)'、
       'Pointer Flow Graph (PFG)'）。

    判据：括号里只有 1~2 个字符，且是单字母 → 公式。
    """
    if _has_math_chars(text):
        return True

    # 归一化后的残留：Solve(S) / AddReachable(m) / ProcessCall(x, oi)
    for match in re.finditer(r"\(([^()]{1,12})\)", text):
        inner = match.group(1).strip()
        if not inner:
            continue
        # 括号内是单个字母（含可能的逗号分隔的单字母）→ 数学记号
        letters_only = re.fullmatch(r"[A-Za-z](?:\s*,\s*[A-Za-z])*", inner)
        if letters_only:
            # 'Worklist (WL)' 是两个大写字母的缩写，不是数学记号。
            # 数学记号通常是小写或单个大写（求和 S、集合 R、变量 m）。
            if len(inner.replace(",", "").strip()) <= 2 and not inner.isupper():
                return True
            if len(inner.replace(",", "").strip()) == 1:
                return True
    return False


def looks_like_title(text: str) -> bool:
    """这一行文本是否像一个标题。

    **这是标题判定的唯一入口** —— 所有规则都在这里，`extract_page_title`
    只负责「取哪一行」与「取原串还是归一化串」。规则分散在两处会导致
    单测覆盖不到其中一半（这正是开发过程中实际踩到的：
    句子检测放在 extract_page_title 里，looks_like_title 单独用时拦不住句子）。

    **保守判定**：任何一条不满足就返回 False。
    """
    if not text:
        return False

    stripped = text.strip()
    if len(stripped) < _MIN_TITLE_LEN or len(stripped) > _MAX_TITLE_LEN:
        return False

    # 公式里的花体字母 / 归一化后的数学记号
    if _looks_like_formula(stripped):
        return False

    # 结尾字符：排除图形残留与代码
    if stripped.endswith(_FORBIDDEN_TRAILING):
        return False
    if not stripped.endswith(_ALLOWED_TRAILING) and not stripped[-1].isalnum():
        # 结尾既不是允许的标点也不是字母数字 → 不可信
        return False

    # 至少要有字母/汉字，且占比够高
    if _letter_ratio(stripped) < 0.5:
        return False

    # 中间出现赋值/代码符号 → 是代码行不是标题。
    # 只查结尾不够：'v = 3 P' 的 '=' 在中间，实测 Week 4 的 p3 就是这样。
    if _CODE_SIGNAL.search(stripped):
        return False

    # 词密度
    if not _word_density_ok(stripped):
        return False

    # 整行都是标点/符号的情况已被字母占比排除，这里再挡一次「①」这类
    # 单字符编号（长度已挡，但 '①.' 这种仍可能漏过）
    if not any(c.isalpha() or _is_cjk(c) for c in stripped):
        return False

    # 句子不是标题
    if _looks_like_sentence(stripped):
        return False

    return True


def _looks_like_sentence(text: str) -> bool:
    """是否是句子而不是标题。

    标题是**名词短语**，句子是主谓结构。区别在语法的可观察痕迹上：
    句子里会出现「is / are / by / of the result」这类动词或介词，
    而标题不会有。这条规则针对实测到的两类误判：

        'x op y by the result of its last evaluation'
        'produces a solution to a data flow analysis'

    注意不能简单地把所有含介词的都排除——'Pointer Flow Graph: An Example'、
    'Call Graph Construction for OOPLs' 这类真标题也含介词。
    所以只在**句子特征词**上收紧，且要求该词出现时行首不是大写词组形态。
    """
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return False
    lowered = {w.lower() for w in words}

    # 明确的动词/谓语标志：这些词出现在中间位置说明这是一个句子
    sentence_markers = {
        "is", "are", "was", "were", "be", "been", "being",
        "produces", "produce", "gives", "give", "makes", "make",
        "means", "means", "represents", "represent",
        "can", "will", "would", "should", "may", "might", "must",
    }
    if lowered & sentence_markers:
        return True

    # 「by the result of」这类介词短语尾巴 —— 句子特征
    if re.search(r"\bby the \w+ of\b", text, flags=re.IGNORECASE):
        return True

    # 单词数过多：标题一般不超过 8 个词
    if len(words) > 9:
        return True

    return False


def _normalize(text: str) -> str:
    """NFKC 归一化：全角字符转半角，避免 'Ｐｏｉｎｔｅｒ' 这类全角标题判不出字母。"""
    return unicodedata.normalize("NFKC", text)


def _is_title_continuation(text: str) -> bool:
    """这一行是否是标题的**延续**。

    **实测结论：这个判断不值得做，已弃用。**

    原本想合并「标题 + 第二行」（实测 20.1% 的页是这种结构），
    但实测发现**无法可靠区分**：

    | 首行 | 第二行 | 第二行到底是什么 |
    |---|---|---|
    | 'Pointer Analysis' | 'Foundations (I)' | 真·标题延续 |
    | '程序分析' | 'Program Analysis' | 真·标题延续（英文版） |
    | 'Rules' | 'Kind Statement Rule' | **表头** |
    | 'Review' | '1. Concerned Statements' | **目录项** |
    | 'Differential Propagation' | 'Why?' | **问句** |
    | '算法理解' | '可以把指针分析这个过程…' | **中文正文** |
    | 'Pointers in Java' | '• Local variable: x' | **正文条目** |

    后四类都不是标题延续，但它们**都能通过标题规则**——
    'Kind Statement Rule' 作为一个短语完全合法。
    靠启发式区分「表头」与「标题」需要语义理解，而规则做不到；
    靠「第二行也像标题」会误合并 119/890 页。

    **取舍**：合并的收益只覆盖约 20~30 页（其中还有一半是误合并），
    而误伤会让片段标题带上错误语义——
    比缺标题更糟。**所以只取首行，不做合并。**

    保留这个函数（返回 False）是为了把这段取舍写进代码，
    避免后来者重做一遍这个实验。
    """
    return False


def extract_page_title(page_content: str) -> str:
    """从一页 PDF 正文里提取标题。

    找不到就返回空字符串——**调用方必须据此跳过补全**，不要写入占位符。
    """
    if not page_content:
        return ""

    lines = [ln.strip() for ln in page_content.splitlines()]
    # 跳过开头的空行
    lines = [ln for ln in lines if ln]

    # 只看前两行：第一行通常是标题；偶尔被图形元素挤到第二行
    for index, line in enumerate(lines[:2]):
        # **先判原串，再归一化。** 顺序反了会漏掉公式标题：
        # NFKC 会把花体 𝑆 规整成普通 S，于是 'Solve(𝑆)' 变成 'Solve(S)'，
        # 数学字符检测就再也拦不住它了（实测 Week 10 的 p17~p26 全是这个形态）。
        if _looks_like_formula(line):
            # 首行是公式 → 这一页**没有标题**，直接放弃。
            # 不能继续往后找：公式后面跟的是正文，
            # 把正文当标题会让片段带上「正文内容在这里」这种无意义的噪声。
            return ""

        candidate = _normalize(line)
        if looks_like_title(candidate):
            # 跨行合并已弃用（见 _is_title_continuation 的 docstring）：
            # 无法可靠区分「标题延续」与「表头/目录项/正文」，误伤代价大于收益。
            return candidate
    return ""


def build_section_path(page_title: str, document_title: str = "") -> list[str]:
    """构造章节路径，供 metadata 与前端展示。

    PDF 的 `metadata["title"]` 是**文档级**的（实测 Week 10/11/9&10 全是
    "Pointer Analysis: Foundations"，Week 4/5/6 是 "PowerPoint Presentation"），
    不是章节级。所以它只能作为路径的顶层，且当它与页标题重复时不重复添加。
    """
    path: list[str] = []
    doc = _normalize(document_title).strip()
    if doc:
        path.append(doc)
    if page_title and page_title not in path:
        path.append(page_title)
    return path


def render_title_prefix(page_title: str, section_path: Iterable[str] = ()) -> str:
    """把标题渲染成写进 chunk 文本的前缀。

    格式选「标题 + 换行」而不是「【标题】」这类标记：
    嵌入模型对自然文本的建模更好，标记符号反而引入噪声 token。
    """
    if not page_title:
        return ""
    parts = [page_title]
    extra = [p for p in section_path if p and p != page_title]
    if extra:
        # 章节路径作为次级信息，附在后面
        parts.append(" / ".join(extra))
    return "\n".join(parts)
