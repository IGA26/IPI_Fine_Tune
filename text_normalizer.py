#!/usr/bin/env python3
"""
text_normalizer.py

Utility helper for lightweight spelling normalization prior to model inference.

The helper uses pyspellchecker (if available) with a small domain whitelist so that
financial terms are not inadvertently corrected. If the dependency is missing, the
normalizer becomes a no-op and emits a warning once.
"""

from __future__ import annotations

import re
import sys
from typing import Iterable, Optional, Set

try:
    from spellchecker import SpellChecker  # type: ignore
except ImportError:  # pragma: no cover - dependency is optional at runtime
    SpellChecker = None  # type: ignore[misc]


class SpellNormalizer:
    """Normalize obvious spelling mistakes while preserving domain-specific terms."""

    DEFAULT_WHITELIST: Set[str] = {
        "isa",
        "lisa",
        "sipp",
        "pension",
        "pensions",
        "mortgage",
        "mortgages",
        "overdraft",
        "overdrafts",
        "isa",
        "isas",
        "lifetime",
        "isa",
        "gilts",
        "annuity",
        "annuities",
        "savings",
        "investment",
        "investments",
        "debt",
        "debts",
        "bond",
        "bonds",
        "credit",
        "credits",
        "loan",
        "loans",
    }

    _warning_emitted: bool = False

    def __init__(
        self,
        distance: int = 1,
        whitelist: Optional[Iterable[str]] = None,
        enabled: bool = True,
    ) -> None:
        """
        Args:
            distance: Maximum edit distance for corrections (1 keeps it fast/safe).
            whitelist: Additional tokens to preserve verbatim.
            enabled: Allow opting out programmatically.
        """
        self.enabled = bool(enabled) and SpellChecker is not None
        self._spell: Optional[SpellChecker] = None  # type: ignore[assignment]

        if not self.enabled:
            if SpellChecker is None and not SpellNormalizer._warning_emitted:
                print(
                    "⚠️  pyspellchecker not installed - spelling normalization disabled",
                    file=sys.stderr,
                )
                SpellNormalizer._warning_emitted = True
            return

        self._spell = SpellChecker(distance=distance)
        self._whitelist: Set[str] = set(self.DEFAULT_WHITELIST)

        if whitelist:
            self._whitelist.update(token.lower() for token in whitelist)

        for token in self._whitelist:
            self._spell.word_frequency.add(token)

    def normalize(self, text: str) -> str:
        """Return text with token-level corrections applied."""
        if not text or not self.enabled or self._spell is None:
            return text

        word_pattern = re.compile(r"\b\w+\b", re.UNICODE)

        def _replace(match: re.Match) -> str:
            word = match.group(0)
            lower_word = word.lower()

            if lower_word in self._whitelist:
                return word

            suggestion = self._spell.correction(lower_word)
            if not suggestion or suggestion == lower_word:
                return word

            if word.isupper():
                return suggestion.upper()
            if word[0].isupper():
                return suggestion.capitalize()
            return suggestion

        return word_pattern.sub(_replace, text)


_DEFAULT_NORMALIZER: Optional[SpellNormalizer] = None


def normalize_spelling(text: str) -> str:
    """
    Convenience function for one-off normalization without manual instantiation.
    """
    global _DEFAULT_NORMALIZER
    if _DEFAULT_NORMALIZER is None:
        _DEFAULT_NORMALIZER = SpellNormalizer()
    return _DEFAULT_NORMALIZER.normalize(text)

