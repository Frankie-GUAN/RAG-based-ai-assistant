"""可替换的 token 计数。

P0 用启发式估算（不引入 tokenizer 依赖）；后续如需精确计数，
实现 TokenCounter 协议换成 tiktoken / DeepSeek tokenizer 即可。
"""
from __future__ import annotations

import math
from typing import Protocol


class TokenCounter(Protocol):
    """token 计数器。

    除「非负」外还有一个**隐含前提**：在**同一段文本的前缀**上，`count` 不应随长度
    减少而增大 —— 也就是 `count(text[:n])` 对 n 不增。本引擎的硬截断按这个前提做
    比例收缩；违反它不会死循环（收缩步长严格递减），但可能反超该层预算。引擎里
    有一处防御性兜底会在收缩到 1 个字符仍不合规时把该条整个丢掉。
    """

    def count(self, text: str) -> int: ...


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # 基本汉字
        or 0x3400 <= code <= 0x4DBF   # 扩展 A
        or 0x3000 <= code <= 0x303F   # CJK 标点
        or 0xFF00 <= code <= 0xFFEF   # 全角
    )


class HeuristicTokenCounter:
    """CJK 约 1 token / 1.5 字符，其余约 1 token / 4 字符。

    实测 DeepSeek 对中文的切分接近该比例，用于预算控制足够准确。
    """

    def count(self, text: str) -> int:
        if not text:
            return 0
        cjk = sum(1 for ch in text if _is_cjk(ch))
        other = len(text) - cjk
        return max(1, math.ceil(cjk / 1.5 + other / 4))
