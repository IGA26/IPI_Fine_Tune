#!/usr/bin/env python3
"""
text_quality.py

Heuristics for assessing basic text quality/noise before running SIL/emotion models.
The checker provides a quick gibberish filter so we can short-circuit and ask the user
to rephrase instead of routing nonsense through the full inference stack.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

try:
    from spellchecker import SpellChecker  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SpellChecker = None  # type: ignore[misc]


@dataclass
class TextQualityResult:
    """Container for quality check metrics."""

    word_count: int
    valid_word_ratio: float
    non_alnum_ratio: float
    repeated_char_ratio: float
    is_gibberish: bool


class TextQualityChecker:
    """
    Lightweight heuristic checker for nonsense input.

    The defaults are conservative—finish inference unless the text is extremely noisy.
    """

    def __init__(
        self,
        min_length: int = 3,
        min_valid_ratio: float = 0.25,
        max_non_alnum_ratio: float = 0.6,
        max_repeated_char_ratio: float = 0.4,
        stopwords: Optional[Sequence[str]] = None,
    ) -> None:
        self.min_length = min_length
        self.min_valid_ratio = min_valid_ratio
        self.max_non_alnum_ratio = max_non_alnum_ratio
        self.max_repeated_char_ratio = max_repeated_char_ratio

        self.stopwords = {s.lower() for s in (stopwords or [])}
        self._spell = SpellChecker(language="en") if SpellChecker else None  # type: ignore

        self._word_pattern = re.compile(r"\b\w+\b", re.UNICODE)
        self._non_alnum_pattern = re.compile(r"[^\w\s]", re.UNICODE)
        self._repeat_pattern = re.compile(r"(.)\1{2,}")

    def _count_valid_words(self, words: Sequence[str]) -> int:
        if not words:
            return 0

        if self._spell is None:
            return sum(1 for w in words if w.lower() not in self.stopwords)

        valid_count = 0
        for word in words:
            lower = word.lower()
            if lower in self.stopwords:
                valid_count += 1
                continue
            if not self._spell.unknown([lower]):  # type: ignore
                valid_count += 1
        return valid_count

    def score(self, text: str) -> TextQualityResult:
        if not text or len(text.strip()) < self.min_length:
            return TextQualityResult(0, 0.0, 1.0, 1.0, True)

        words = self._word_pattern.findall(text)
        word_count = len(words)
        valid_word_ratio = (
            self._count_valid_words(words) / word_count if word_count else 0.0
        )

        non_alnum_matches = self._non_alnum_pattern.findall(text)
        non_alnum_ratio = (
            len(non_alnum_matches) / max(len(text), 1) if text else 0.0
        )

        repeated_spans = list(self._repeat_pattern.finditer(text))
        repeated_char_ratio = (
            sum(match.end() - match.start() for match in repeated_spans)
            / max(len(text), 1)
            if repeated_spans
            else 0.0
        )

        triggers = [
            valid_word_ratio < self.min_valid_ratio,
            non_alnum_ratio > self.max_non_alnum_ratio,
            repeated_char_ratio > self.max_repeated_char_ratio,
            word_count == 0,
        ]

        is_gibberish = any(triggers)

        return TextQualityResult(
            word_count=word_count,
            valid_word_ratio=valid_word_ratio,
            non_alnum_ratio=non_alnum_ratio,
            repeated_char_ratio=repeated_char_ratio,
            is_gibberish=is_gibberish,
        )

    def is_gibberish(self, text: str) -> bool:
        return self.score(text).is_gibberish


_DEFAULT_CHECKER: Optional[TextQualityChecker] = None


def is_gibberish(text: str) -> bool:
    global _DEFAULT_CHECKER
    if _DEFAULT_CHECKER is None:
        _DEFAULT_CHECKER = TextQualityChecker()
    return _DEFAULT_CHECKER.is_gibberish(text)

