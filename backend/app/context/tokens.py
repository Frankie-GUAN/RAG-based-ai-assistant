"""可替换的 token 计数。

P0 用启发式估算（不引入 tokenizer 依赖）；后续如需精确计数，
实现 TokenCounter 协议换成 tiktoken / DeepSeek tokenizer 即可。
"""
from __future__ import annotations

import math
from typing import Protocol


class TokenCounter(Protocol):
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
