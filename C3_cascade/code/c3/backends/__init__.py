# -*- coding: utf-8 -*-
"""Model backends for C3."""

from .hf_causal_lm import LocalCorrectionBackend, LocalTranslationBackend
from .openai_compatible import OpenAICompatibleCorrectionBackend

__all__ = ["LocalCorrectionBackend", "LocalTranslationBackend", "OpenAICompatibleCorrectionBackend"]
