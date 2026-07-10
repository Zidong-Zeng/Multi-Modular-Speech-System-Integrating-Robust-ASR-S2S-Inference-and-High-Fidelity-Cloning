# -*- coding: utf-8 -*-
"""Compatibility wrappers for earlier stage-oriented scripts."""

from __future__ import annotations

import os

from .correction import correct_details, transformer_compatible_prediction
from .correction_units import build_c3_predictions, summarize_unit_construction
from .io import load_c2_predictions, load_c3_details, write_json, write_jsonl
from .schemas import CorrectionUnitConfig
from .translation import translate_details, translation_prediction


def load_stage2_predictions(stage2_json: str) -> list[dict]:
    return load_c3_details(stage2_json)


def load_stage3_details(stage3_json: str) -> list[dict]:
    return load_c3_details(stage3_json)


def run_stage2(c2_json: str, outdir: str, config: CorrectionUnitConfig | None = None) -> tuple[str, str]:
    config = config or CorrectionUnitConfig()
    c2_predictions = load_c2_predictions(c2_json)
    predictions = build_c3_predictions(c2_predictions, config)
    summary = summarize_unit_construction(predictions, c2_json, config)
    summary["stage"] = "stage2_correction_unit_construction"

    predictions_path = write_json(os.path.join(outdir, "c3_predictions.json"), predictions)
    summary_path = write_json(os.path.join(outdir, "c3_summary.json"), summary)
    return predictions_path, summary_path


def run_stage3(
    stage2_json: str,
    outdir: str,
    correction_backend=None,
    correction_model: str | None = None,
    max_new_tokens: int = 128,
) -> tuple[str, str, str]:
    stage2_predictions = load_stage2_predictions(stage2_json)
    details, prompt_records, summary = correct_details(
        stage2_predictions,
        stage2_json,
        correction_backend=correction_backend,
        correction_model=correction_model,
        max_new_tokens=max_new_tokens,
    )
    predictions = [transformer_compatible_prediction(detail) for detail in details]
    summary["stage"] = "stage3_correction"
    summary["source_stage2_json"] = summary.pop("source_json")

    prompts_dir = os.path.join(outdir, "prompts")
    predictions_path = write_json(os.path.join(outdir, "c3_predictions.json"), predictions)
    details_path = write_json(os.path.join(outdir, "c3_correction_details.json"), details)
    summary_path = write_json(os.path.join(outdir, "c3_summary.json"), summary)
    write_jsonl(os.path.join(prompts_dir, "correction_prompts.jsonl"), prompt_records)
    return predictions_path, details_path, summary_path


def run_stage4(
    stage3_json: str,
    outdir: str,
    translation_backend=None,
    translation_model: str | None = None,
    max_new_tokens: int = 256,
) -> tuple[str, str, str]:
    stage3_details = load_stage3_details(stage3_json)
    details, prompt_records, summary = translate_details(
        stage3_details,
        stage3_json,
        translation_backend=translation_backend,
        translation_model=translation_model,
        max_new_tokens=max_new_tokens,
    )
    translations = [translation_prediction(detail) for detail in details]
    summary["stage"] = "stage4_translation"
    summary["source_stage3_json"] = summary.pop("source_json")

    prompts_dir = os.path.join(outdir, "prompts")
    translations_path = write_json(os.path.join(outdir, "c3_translations.json"), translations)
    details_path = write_json(os.path.join(outdir, "c3_translation_details.json"), details)
    summary_path = write_json(os.path.join(outdir, "c3_summary.json"), summary)
    write_jsonl(os.path.join(prompts_dir, "translation_prompts.jsonl"), prompt_records)
    return translations_path, details_path, summary_path
