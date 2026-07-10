# -*- coding: utf-8 -*-
"""Backend protocols used by correction and translation modules."""

from __future__ import annotations

from typing import Protocol


class CorrectionBackend(Protocol):
    def correct(self, prompt: str) -> str:
        ...


class TranslationBackend(Protocol):
    def translate(self, prompt: str) -> str:
        ...
