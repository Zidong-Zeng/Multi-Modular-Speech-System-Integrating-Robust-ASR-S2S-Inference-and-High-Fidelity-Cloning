# -*- coding: utf-8 -*-
"""End-to-end C3 cascade composition."""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import Callable

from .correction import correct_details
from .correction_units import build_c3_predictions
from .io import load_c2_predictions, write_json, write_jsonl
from .metrics import correction_error_rate_summary
from .schemas import CascadeResult, CorrectionUnitConfig
from .translation import translate_details, translation_prediction


def build_pipeline_summary(
    details: list[dict],
    c2_json: str,
    correction_model: str,
    translation_model: str,
    elapsed_sec: float,
) -> dict:
    warning_counts: Counter[str] = Counter()
    num_corrected_units = 0
    num_fallback_units = 0
    num_translated_units = 0
    for detail in details:
        for unit in detail.get("correction_units", []):
            warning_counts.update(unit.get("warnings", []))
            if unit.get("correction_mode") == "fallback_top1":
                num_fallback_units += 1
            else:
                num_corrected_units += 1
            if unit.get("translation_zh"):
                num_translated_units += 1

    summary = {
        "pipeline": "c3_cascade",
        "source_c2_json": os.path.abspath(c2_json),
        "correction_model": correction_model,
        "translation_model": translation_model,
        "num_samples": len(details),
        "num_chunks": int(sum(detail.get("num_chunks", 0) for detail in details)),
        "num_correction_units": int(sum(detail.get("num_correction_units", 0) for detail in details)),
        "num_corrected_units": num_corrected_units,
        "num_fallback_units": num_fallback_units,
        "num_translated_units": num_translated_units,
        "num_warnings": int(sum(warning_counts.values())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "elapsed_sec": round(elapsed_sec, 3),
    }
    summary.update(correction_error_rate_summary(details))
    return summary


def run_cascade(
    c2_json: str,
    outdir: str,
    correction_backend=None,
    translation_backend=None,
    correction_model: str | None = None,
    translation_model: str | None = None,
    correction_max_new_tokens: int = 128,
    translation_max_new_tokens: int = 256,
    config: CorrectionUnitConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> CascadeResult:
    def progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    started = time.time()
    config = config or CorrectionUnitConfig()
    progress(f"[C3] Loading C2 predictions: {os.path.abspath(c2_json)}")
    c2_predictions = load_c2_predictions(c2_json)
    progress(f"[C3] Loaded {len(c2_predictions)} samples")
    unit_details = build_c3_predictions(c2_predictions, config)
    num_chunks = int(sum(detail.get("num_chunks", 0) for detail in unit_details))
    num_units = int(sum(detail.get("num_correction_units", 0) for detail in unit_details))
    num_fallback_units = int(
        sum(
            1
            for detail in unit_details
            for unit in detail.get("correction_units", [])
            if unit.get("correction_mode") == "fallback_top1"
        )
    )
    progress(f"[C3] Built {num_units} correction units from {num_chunks} chunks ({num_fallback_units} fallback)")

    progress("[C3] Starting correction")
    corrected_details, correction_prompt_records, _ = correct_details(
        unit_details,
        c2_json,
        correction_backend=correction_backend,
        correction_model=correction_model,
        max_new_tokens=correction_max_new_tokens,
        progress_callback=progress,
    )
    progress(f"[C3] Correction complete: {len(correction_prompt_records)} model calls")
    progress("[C3] Starting translation")
    translated_details, translation_prompt_records, _ = translate_details(
        corrected_details,
        c2_json,
        translation_backend=translation_backend,
        translation_model=translation_model,
        max_new_tokens=translation_max_new_tokens,
        progress_callback=progress,
    )
    progress(f"[C3] Translation complete: {len(translation_prompt_records)} model calls")

    predictions = [translation_prediction(detail) for detail in translated_details]
    summary = build_pipeline_summary(
        translated_details,
        c2_json,
        correction_model or "",
        translation_model or "",
        time.time() - started,
    )
    summary["config"] = config.to_dict()

    progress(f"[C3] Writing outputs: {os.path.abspath(outdir)}")
    prompts_dir = os.path.join(outdir, "prompts")
    predictions_path = write_json(os.path.join(outdir, "c3_predictions.json"), predictions)
    details_path = write_json(os.path.join(outdir, "c3_details.json"), translated_details)
    summary_path = write_json(os.path.join(outdir, "c3_summary.json"), summary)
    correction_prompts_path = write_jsonl(os.path.join(prompts_dir, "correction_prompts.jsonl"), correction_prompt_records)
    translation_prompts_path = write_jsonl(os.path.join(prompts_dir, "translation_prompts.jsonl"), translation_prompt_records)

    result = CascadeResult(
        predictions_path=predictions_path,
        details_path=details_path,
        summary_path=summary_path,
        correction_prompts_path=correction_prompts_path,
        translation_prompts_path=translation_prompts_path,
    )
    progress(f"[C3] Done in {round(time.time() - started, 3)}s")
    return result
