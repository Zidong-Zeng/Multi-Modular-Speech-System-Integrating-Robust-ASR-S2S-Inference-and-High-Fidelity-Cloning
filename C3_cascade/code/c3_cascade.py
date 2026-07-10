# -*- coding: utf-8 -*-
"""Compatibility entrypoint for the modular C3 cascade package."""

from __future__ import annotations

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from c3.backends.hf_causal_lm import LocalCorrectionBackend, LocalTranslationBackend
from c3.backends.openai_compatible import OpenAICompatibleCorrectionBackend
from c3.cli import build_arg_parser, config_from_args, main
from c3.compat import load_stage2_predictions, load_stage3_details, run_stage2, run_stage3, run_stage4
from c3.correction import (
    correct_unit,
    summarize_correction as summarize_stage3,
    transformer_compatible_prediction,
)
from c3.correction_units import (
    build_c3_prediction,
    build_c3_predictions,
    build_correction_units,
    build_unit,
    can_merge_next,
    crosses_speaker_boundary,
    nbest_by_chunk,
    normalized_chunk,
    sorted_valid_chunks,
    summarize_unit_construction as summarize_predictions,
    unit_duration_ms,
    unit_is_ready,
    unit_text,
)
from c3.io import load_c2_predictions
from c3.pipeline import run_cascade
from c3.prompts.correction import build_correction_prompt, clean_correction_output
from c3.prompts.translation import build_translation_prompt, clean_translation_output
from c3.schemas import CascadeResult, CorrectionUnitConfig
from c3.text_assembly import (
    assemble_texts,
    assemble_translations_zh,
    find_word_overlap,
    normalize_text,
    word_count,
)
from c3.translation import (
    summarize_translation as summarize_stage4,
    translate_unit,
    translation_prediction,
)


__all__ = [
    "CascadeResult",
    "CorrectionUnitConfig",
    "LocalCorrectionBackend",
    "LocalTranslationBackend",
    "OpenAICompatibleCorrectionBackend",
    "assemble_texts",
    "assemble_translations_zh",
    "build_arg_parser",
    "build_c3_prediction",
    "build_c3_predictions",
    "build_correction_prompt",
    "build_correction_units",
    "build_translation_prompt",
    "build_unit",
    "can_merge_next",
    "clean_correction_output",
    "clean_translation_output",
    "config_from_args",
    "correct_unit",
    "crosses_speaker_boundary",
    "find_word_overlap",
    "load_c2_predictions",
    "load_stage2_predictions",
    "load_stage3_details",
    "main",
    "nbest_by_chunk",
    "normalize_text",
    "normalized_chunk",
    "run_cascade",
    "run_stage2",
    "run_stage3",
    "run_stage4",
    "sorted_valid_chunks",
    "summarize_predictions",
    "summarize_stage3",
    "summarize_stage4",
    "transformer_compatible_prediction",
    "translate_unit",
    "translation_prediction",
    "unit_duration_ms",
    "unit_is_ready",
    "unit_text",
    "word_count",
]


if __name__ == "__main__":
    main()
