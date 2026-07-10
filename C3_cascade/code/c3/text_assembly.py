# -*- coding: utf-8 -*-
"""Text normalization and overlap-aware assembly."""

from __future__ import annotations

import re
from typing import Sequence


def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(normalize_text(text).split())


def find_word_overlap(left_words: Sequence[str], right_words: Sequence[str], max_overlap_words: int = 8) -> int:
    max_n = min(len(left_words), len(right_words), max_overlap_words)
    for n in range(max_n, 0, -1):
        if list(left_words[-n:]) == list(right_words[:n]):
            return n
    return 0


def assemble_texts(texts: Sequence[str], max_overlap_words: int = 8) -> str:
    raw_words: list[str] = []
    normalized_words: list[str] = []
    for text in texts:
        next_raw = str(text or "").strip().split()
        next_norm = normalize_text(text).split()
        if not next_raw or not next_norm:
            continue
        overlap = find_word_overlap(normalized_words, next_norm, max_overlap_words=max_overlap_words)
        raw_words.extend(next_raw[overlap:])
        normalized_words.extend(next_norm[overlap:])
    return " ".join(raw_words).strip()


def assemble_translations_zh(texts: Sequence[str]) -> str:
    return "".join(str(text or "").strip() for text in texts if str(text or "").strip())
