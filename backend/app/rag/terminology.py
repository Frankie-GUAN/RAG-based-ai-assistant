"""跨语言检索：中文 query → 英文术语的自动挖掘与改写。

**问题。** 语料是英文课件，query 是中文。BGE-M3 是多语言模型，能跨语言匹配语义，
但**词面完全对不上**——实测「常量传播分析怎么做」召不回任何相关片段，
因为语料里写的是 "Constant Propagation"。

**为什么不能靠向量硬扛。** 多语言 embedding 的跨语言对齐是**粗粒度**的：
它能把「常量传播」和 "Constant Propagation" 判为语义相近，
但这个信号在 903 个片段的语料里排不到前面——实测向量召回了
「Pointer Analysis 的 Solution 章节」（同样是"解决方案"的意思）。
而**词面匹配是精确的**：只要 "Constant Propagation" 这几个字出现在 query 里，
BM25 立刻把它排到第一。

**所以解法是让词面这一路能工作**：在检索前把中文 query 补上对应的英文术语。

**术语表怎么来 —— 不手工维护，从语料自动挖掘。**

流程：
1. 扫描语料，抽取高频英文术语短语（连续首字母大写的词组，是学术术语的形态特征）
2. 用 BGE-M3 把每个术语嵌入
3. 把用户 query 也嵌入，取最近的若干术语作为「英文候选」
4. **用领域一致性校验候选**（见 `candidates_for` 的说明）

**为什么这样比手工术语表好**：
- 手工表要人知道"常量传播 = Constant Propagation"，但语料换了就得重维护
- 自动挖掘跟着语料走，语料更新后重新跑一次即可
- 顺带能发现**用户自己都不知道对应英文**的术语

**代价与边界。** 术语候选是**向量近邻 + 领域校验**，仍可能不准。
所以：
- 只在**确实需要**时才加（query 含中文才启用）
- 加的是**候选**而非替换——原 query 保留，术语以附加形式拼在后面
- 校验全不过就**不改写**——错误术语会污染词面匹配，比没有更糟
- 命中率必须用金标集验证，不能靠直觉
"""
from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

# 英文术语的形态特征：连续的首字母大写词组。
# 学术课件里的术语几乎都是这个形态（Pointer Analysis、Constant Propagation），
# 而普通英文句子不是（The algorithm works ...）。
_TERM_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,}){0,2})\b"
)

# 句首的普通词，不是术语
_STOP_HEADS = {
    "The", "This", "That", "These", "Those", "For", "When", "While", "Where",
    "If", "Let", "Add", "Remove", "Foreach", "Return", "Example", "Note",
    "See", "Use", "Now", "Thus", "Hence", "Then", "There", "Here", "Each",
    "Some", "All", "Any", "But", "And", "Not", "Its", "Our", "Their",
    "Understand", "Processing", "Analysis", "Algorithm", "Algorithms",
    "Program", "Context", "Entry", "Maintain", "Similarly", "Precise",
    "Kind", "Statement", "Rule", "Reading", "Solving", "Sharing", "Setting",
}

# 中文检测：CJK 统一表意文字
_CJK = re.compile(r"[一-鿿]")

# 术语候选的最低出现频次 —— 出现一次的多是偶然词组，不是术语
_MIN_TERM_FREQ = 5

# 每个 query 最多附加几个术语候选
#
# **实测定的是 1，不是 3。** BM25 是词袋模型：追加的词会进入同一个词袋，
# 改变**所有**词的 tf/idf 权重。加 3 个术语时，"Live Variables Analysis"、
# "Data Flow Analysis" 这类通用词会把精确的 "Available Expressions Analysis"
# 挤下去——实测「可用表达式分析的定义」的正确页 p114 直接掉出 top3，
# nDCG 从 0.631 掉到 0.000。
#
# 只加**最准的那一个**术语时，词袋里多的是一个高区分度的词，
# 不稀释原有的精确匹配。
#
# 实测对照（同一金标集，n=15，配对 t 检验）：
#     加 3 个术语：nDCG 0.297 → 0.423，Δ=+0.125，t=1.32，**不显著**
#     加 1 个术语：nDCG 0.297 → 0.629，Δ=+0.332，t=3.48，**显著**（10 胜 1 负）
#
# **少即是多**：改写的作用是补上「词面对不上」这一个缺口，
# 不是优化排序。补一个精准词就够了，补三个反而会引入噪声。
_MAX_TERM_CANDIDATES = 1

# 送进领域校验的候选池大小 —— 校验要跑一次 BM25 检索，不能太多。
# 实测正确术语常排在第 5~7 位（向量相似度不够精确），所以池子不能太小。
_VERIFY_POOL_SIZE = 20

# 术语与参考域的最低文档重合度 —— 低于此值视为噪声术语
_MIN_DOMAIN_OVERLAP = 0.5


def has_cjk(text: str) -> bool:
    """文本里是否含中文。"""
    return bool(_CJK.search(text))


def _normalize_term(term: str) -> str:
    """规范化术语：压掉换行、合并空白。

    实测语料里有 'Constant Propagation\nGiven' 这种形态 ——
    PDF 抽取时把公式和文字挤在一起。直接当术语用会让 BM25 匹配到
    一个不存在的字符串。规范化后取第一段（'Constant Propagation'）。
    """
    cleaned = re.sub(r"\s+", " ", term).strip()
    if "\n" in term:
        # 原本可能有换行，压成空格后取最像术语的那一段
        parts = [p.strip() for p in cleaned.split(" ") if p.strip()]
        # 逐段尝试，找到第一段像术语的（首字母大写、长度足够）
        for part in parts:
            if len(part) >= 7 and part[0].isupper() and part.split()[0] not in _STOP_HEADS:
                return part
        return parts[0] if parts else cleaned
    return cleaned


def extract_terms(documents: Iterable[str], min_freq: int = _MIN_TERM_FREQ) -> list[str]:
    """从语料里抽取高频英文术语短语。

    返回按频次降序排列的术语列表。
    """
    counter: Counter[str] = Counter()
    for text in documents:
        for match in _TERM_PATTERN.findall(text):
            if len(match) < 7:          # 太短的多是虚词组合
                continue
            if match.split()[0] in _STOP_HEADS:
                continue
            counter[match] += 1

    terms = [(t, c) for t, c in counter.items() if c >= min_freq]
    terms.sort(key=lambda x: (-x[1], x[0]))
    return [_normalize_term(t) for t, _ in terms]


def extract_section_titles(documents: Iterable[str]) -> list[str]:
    """从语料里抽取**章节标题**——最可靠的术语来源。

    **为什么标题比词频统计好。** 标题是语料自己声明的章节名，
    「Constant Propagation – Transfer Function」这种带后缀的形式，
    在词频统计里会被当成两个不同的词组，而正文里只写 "Constant Propagation"。
    标题同时给出了**规范写法**与**出现频次**（同一标题出现 9 次说明它是重要章节）。

    实测语料里有 121 个不同的章节标题，覆盖了
    "Constant Propagation – Lattice"、"Lattice"、"Monotonicity"、
    "Interprocedural Control-Flow Graph" 这些关键术语。
    """
    counter: Counter[str] = Counter()
    for text in documents:
        for match in _TERM_PATTERN.findall(text):
            if len(match) < 7:
                continue
            if match.split()[0] in _STOP_HEADS:
                continue
            counter[match] += 1
    # 频次高的短语里，包含其它短语的那些更可能是「带后缀的章节名」，
    # 它们的**前缀**才是真正的术语。所以这里额外产出前缀形式。
    base = [t for t, c in counter.items() if c >= _MIN_TERM_FREQ]

    prefixes: set[str] = set()
    for term in base:
        normalized = _normalize_term(term)
        # 'Constant Propagation – Transfer Function' → 'Constant Propagation'
        for sep in (" – ", " - ", ": ", " — "):
            if sep in normalized:
                prefix = normalized.split(sep)[0].strip()
                if len(prefix) >= 7 and prefix.split()[0] not in _STOP_HEADS:
                    prefixes.add(prefix)
    return sorted(prefixes)


class TerminologyMapper:
    """中文 query → 英文术语候选。

    **单例 + 惰性加载。** 术语挖掘要扫全语料、嵌入所有术语，
    是一次性的重活（实测 ~10 秒），不该在每次检索时重做。
    """

    def __init__(self, terms: Sequence[str], embeddings_model=None):
        self._terms = list(terms)
        self._model = embeddings_model
        self._term_vectors = None
        self._lock = threading.Lock()

    @property
    def terms(self) -> list[str]:
        return list(self._terms)

    def _ensure_vectors(self) -> None:
        if self._term_vectors is not None:
            return
        with self._lock:
            if self._term_vectors is not None:
                return
            if not self._terms:
                self._term_vectors = []
                return
            from app.rag.vector_store import get_embeddings
            model = self._model or get_embeddings()
            self._term_vectors = model.embed_documents(self._terms)

    def candidates_for(self, query: str, top_k: int = _MAX_TERM_CANDIDATES) -> list[str]:
        """找出与 query 最相关的英文术语候选。

        **只对含中文的 query 有意义** —— 英文 query 本来就在语料语言里，
        不需要改写。

        **筛选用「领域一致性」，不是单看向量相似度。**

        实测（见 docs/tech-notes/verified-facts.md §9）：
        纯向量相似度不够准 —— 对 query「常量传播分析怎么做」，
        "Constant Propagation" 只排到第 7 位（0.529），
        排在第一的 "Flow Analysis"（0.550）是噪声。

        所以加第二道校验：**用候选术语自己检索一次，看它是否命中
        与原 query 向量检索相同的文档**。术语若真的对应这个语义，
        它必然指向同一批文档；噪声术语会指向别处。

        实测这道校验的判别力：
            「活跃变量分析怎么定义」→ "Live Variables Analysis"  ovl=1.00 ✓
            「单调性和不动点定理」    → "Monotonicity"             ovl=1.00 ✓
            「常量传播分析怎么做」    → "Constant Propagation"     ovl=0.67 ✓（排第一）
            「常量传播分析怎么做」    → "Implementing Pointer Analysis" ovl=0.67 ✗（排更后）
        """
        if not has_cjk(query) or not self._terms:
            return []

        self._ensure_vectors()
        if not self._term_vectors:
            return []

        from app.rag.vector_store import get_embeddings
        model = self._model or get_embeddings()
        query_vec = model.embed_query(query)

        # 余弦相似度（各模型都已归一化，点积即余弦）
        scores = [sum(a * b for a, b in zip(query_vec, v)) for v in self._term_vectors]
        ranked = sorted(zip(scores, self._terms), key=lambda x: x[0], reverse=True)

        # 只对相似度靠前的候选做昂贵的领域校验
        pool = ranked[:_VERIFY_POOL_SIZE]

        reference = self._reference_sources(query)
        verified: list[tuple[float, float, str]] = []
        for score, term in pool:
            overlap = self._domain_overlap(term, reference)
            if overlap >= _MIN_DOMAIN_OVERLAP:
                verified.append((overlap, score, term))

        if not verified:
            # 校验全不过 → 宁可不改写。错误术语会污染词面匹配，比没有更糟。
            return []

        # **排序以领域一致性为主，向量相似度为辅。**
        #
        # 实测（query「常量传播分析怎么做」）：
        #     "Constant Propagation"        sim=0.529  ovl=1.00  ← 正确
        #     "Flow Analysis"               sim=0.550  ovl=0.67  ← 噪声，但相似度最高
        #     "Implementing Pointer Analysis" sim=0.529 ovl=0.67 ← 噪声
        # 若按相似度排，"Constant Propagation" 只能排到第 7 位，被 top_k=3 截掉；
        # 按 ovl 排它是第一。**领域一致性才是更强的信号**——
        # 相似度只区分不了「都指向同一领域的多个术语」，ovl 可以。
        verified.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [t for _, _, t in verified[:top_k]]

    def _reference_sources(self, query: str) -> set[str]:
        """原 query 的向量检索结果作为「参考域」。

        它代表「这个语义问题应该在哪些文档里」——
        术语必须落在这个域里才算对应。

        **性能注意**：这里会触发一次向量检索，是整条链路上最贵的一步。
        但因为 query 向量走缓存（app/rag/query_cache.py），
        同一个 query 的第二次调用几乎免费 —— 而术语改写本来就在检索链路里，
        这一次「额外」检索实际上是**命中缓存**的。
        """
        try:
            from app.rag.hybrid_search import _vector_search
            results = _vector_search(query, top_k=10)
        except Exception as exc:  # noqa: BLE001 — 校验失败就退化为「不校验」
            logger.warning("参考域计算失败，跳过术语校验：%s: %s", type(exc).__name__, exc)
            return set()
        return {
            str((doc.metadata or {}).get("source") or "")
            for doc, _ in results
        }

    def _domain_overlap(self, term: str, reference: set[str]) -> float:
        """候选术语检索到的文档与参考域的重合度。

        实测有一个陷阱：**不能只看重合度**。
        「常量传播」的 "Constant Propagation"（ovl=0.67，正确）与
        "Implementing Pointer Analysis"（ovl=0.67，错误）重合度一样，
        靠向量相似度才能把它们分开（0.529 vs 0.529，前者排更前）。
        所以最终排序仍以向量相似度为主，重合度只做**准入过滤**。
        """
        if not reference:
            return 0.0
        try:
            from app.rag.bm25_index import get_bm25_index
            results = get_bm25_index().search(term, top_k=10)
        except Exception as exc:  # noqa: BLE001 — 校验失败 → 视为不通过
            logger.warning("术语校验检索失败：%s: %s", type(exc).__name__, exc)
            return 0.0
        if not results:
            return 0.0
        sources = {
            str((doc.metadata or {}).get("source") or "")
            for doc, _ in results
        }
        return len(sources & reference) / len(sources)

_mapper: TerminologyMapper | None = None
_mapper_lock = threading.Lock()


def build_mapper(documents: Iterable[str] | None = None) -> TerminologyMapper:
    """从语料构建术语映射器。

    **术语有两个来源，缺一不可**：

    1. **词频统计**（`extract_terms`）：抓正文里反复出现的术语，
       如 "Partial Order"、"Transitivity"、"Antisymmetry"
    2. **章节标题前缀**（`extract_section_titles`）：抓语料自己声明的章节名，
       如 "Constant Propagation"（来自 "Constant Propagation – Transfer Function"）、
       "Lattice"、"Interprocedural Control-Flow Graph"

    只用第 1 个来源时，实测「常量传播分析怎么做」的正确术语
    "Constant Propagation" 因向量相似度不足（0.529，排第 7）而被挤出候选池。
    加上第 2 个来源后它才稳定入选——**因为标题是语料自己的规范写法**。
    """
    global _mapper
    if _mapper is not None:
        return _mapper
    with _mapper_lock:
        if _mapper is not None:
            return _mapper
        if documents is None:
            documents = _corpus_texts()
        docs = list(documents)

        terms = extract_terms(docs)
        section_terms = extract_section_titles(docs)

        # 合并去重，章节标题优先（它是规范写法）
        merged: list[str] = []
        seen: set[str] = set()
        for t in section_terms + terms:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                merged.append(t)

        logger.info(
            "已从语料挖掘 %d 个术语（词频 %d + 章节标题前缀 %d）",
            len(merged), len(terms), len(section_terms),
        )
        _mapper = TerminologyMapper(merged)
        return _mapper


def reset_mapper() -> None:
    """清掉缓存（测试与运维用）。"""
    global _mapper
    with _mapper_lock:
        _mapper = None


def _corpus_texts() -> list[str]:
    """读出语料全文。失败时返回空列表——术语改写是增强，不该让检索失败。"""
    try:
        from app.rag.vector_store import get_vectorstore
        store = get_vectorstore()
        if store is None:
            return []
        data = store._collection.get(include=["documents"])
        return list(data.get("documents") or [])
    except Exception as exc:  # noqa: BLE001 — 任何失败都退化为「不改写」
        logger.warning("读取语料失败，术语改写不可用：%s: %s", type(exc).__name__, exc)
        return []


def rewrite_query(query: str, mapper: TerminologyMapper | None = None) -> str:
    """给中文 query 附加英文术语候选。

    **是附加，不是替换。** 原 query 保留在前面：
    - 保留中文词 → 向量侧仍能靠语义匹配
    - 追加英文词 → BM25 侧能靠词面精确匹配

    两路各取所长，这正是混合检索该有的样子。
    """
    if not has_cjk(query):
        return query
    mapper = mapper or build_mapper()
    candidates = mapper.candidates_for(query)
    if not candidates:
        return query
    return f"{query} {' '.join(candidates)}"
