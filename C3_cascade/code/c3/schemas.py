# -*- coding: utf-8 -*-
"""Shared C3 data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CorrectionUnitConfig:
    min_unit_ms: int = 6000
    min_unit_words: int = 10
    target_max_ms: int = 25000
    hard_max_ms: int = 29500
    max_unit_words: int = 80
    max_merge_gap_ms: int = 1200
    fallback_min_ms: int = 3000
    fallback_min_words: int = 5
    max_overlap_words: int = 8
    respect_speaker_boundary: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        fields = {
            "min_unit_ms": self.min_unit_ms,
            "min_unit_words": self.min_unit_words,
            "target_max_ms": self.target_max_ms,
            "hard_max_ms": self.hard_max_ms,
            "max_unit_words": self.max_unit_words,
            "max_merge_gap_ms": self.max_merge_gap_ms,
            "fallback_min_ms": self.fallback_min_ms,
            "fallback_min_words": self.fallback_min_words,
            "max_overlap_words": self.max_overlap_words,
        }
        for name, value in fields.items():
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.hard_max_ms < self.target_max_ms:
            raise ValueError("hard_max_ms must be greater than or equal to target_max_ms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_unit_ms": self.min_unit_ms,
            "min_unit_words": self.min_unit_words,
            "target_max_ms": self.target_max_ms,
            "hard_max_ms": self.hard_max_ms,
            "max_unit_words": self.max_unit_words,
            "max_merge_gap_ms": self.max_merge_gap_ms,
            "fallback_min_ms": self.fallback_min_ms,
            "fallback_min_words": self.fallback_min_words,
            "max_overlap_words": self.max_overlap_words,
            "respect_speaker_boundary": self.respect_speaker_boundary,
        }


@dataclass
class CascadeResult:
    predictions_path: str
    details_path: str
    summary_path: str
    correction_prompts_path: str
    translation_prompts_path: str
