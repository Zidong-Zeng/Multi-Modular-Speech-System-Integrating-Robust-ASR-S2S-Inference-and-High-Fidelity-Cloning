# -*- coding: utf-8 -*-
"""Apply the correction model to correction units."""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import Callable, Sequence

from .backends.hf_causal_lm import LocalCorrectionBackend
from .metrics import add_correction_error_rates, correction_error_rate_summary
from .prompts.correction import build_correction_prompt, clean_correction_output
from .text_assembly import assemble_texts


def correct_unit(unit: dict, sample_id: str, correction_backend) -> tuple[dict, dict | None]:
    row = dict(unit)
    warnings = list(row.get("warnings", []))
    row["warnings"] = warnings

    if row.get("correction_mode") == "fallback_top1":
        row["corrected_en"] = str(row.get("corrected_en") or row.get("asr_top1_en", "")).strip()
        row.setdefault("correction_prompt", "")
        row.setdefault("correction_raw_output", "")
        return row, None

    prompt = build_correction_prompt(row)
    raw_output = correction_backend.correct(prompt)
    corrected = clean_correction_output(raw_output)
    if not corrected:
        corrected = str(row.get("asr_top1_en", "")).strip()
        row["correction_mode"] = "fallback_top1"
        warnings.append("empty_correction_output")

    row["corrected_en"] = corrected
    row["correction_prompt"] = prompt
    row["correction_raw_output"] = raw_output
    prompt_record = {
        "id": sample_id,
        "unit_id": row.get("unit_id"),
        "source_chunk_ids": row.get("source_chunk_ids", []),
        "prompt": prompt,
        "raw_output": raw_output,
        "corrected_en": corrected,
    }
    return row, prompt_record


def correct_samples(
    details: Sequence[dict],
    correction_backend,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[dict]]:
    corrected_details = []
    prompt_records = []
    total_units = sum(len(sample.get("correction_units", [])) for sample in details)
    current_unit = 0
    for sample in details:
        corrected_units = []
        for unit in sample.get("correction_units", []):
            current_unit += 1
            if progress_callback is not None:
                progress_callback(
                    f"[C3] Correction {current_unit}/{total_units}: "
                    f"id={sample.get('id')} unit={unit.get('unit_id')} mode={unit.get('correction_mode', 'generate')}"
                )
            corrected_unit, prompt_record = correct_unit(unit, str(sample.get("id", "")), correction_backend)
            corrected_units.append(corrected_unit)
            if prompt_record is not None:
                prompt_records.append(prompt_record)

        corrected_transcript = assemble_texts(
            [unit.get("corrected_en", "") for unit in corrected_units],
            max_overlap_words=8,
        )
        detail = dict(sample)
        detail["correction_units"] = corrected_units
        detail["corrected_transcript_en"] = corrected_transcript
        detail["num_correction_units"] = len(corrected_units)
        corrected_details.append(add_correction_error_rates(detail))
    return corrected_details, prompt_records


def transformer_compatible_prediction(detail: dict) -> dict:
    row = {
        "id": detail.get("id"),
        "audio": detail.get("audio"),
        "reference": detail.get("reference", ""),
        "hypothesis": detail.get("corrected_transcript_en", ""),
    }
    if "emotion" in detail:
        row["emotion"] = detail.get("emotion")
    return row


def summarize_correction(
    details: Sequence[dict],
    source_json: str,
    correction_model: str,
    elapsed_sec: float,
) -> dict:
    warning_counts: Counter[str] = Counter()
    num_corrected_units = 0
    num_fallback_units = 0
    for detail in details:
        for unit in detail.get("correction_units", []):
            warning_counts.update(unit.get("warnings", []))
            if unit.get("correction_mode") == "fallback_top1":
                num_fallback_units += 1
            else:
                num_corrected_units += 1
    summary = {
        "phase": "correction",
        "source_json": os.path.abspath(source_json),
        "correction_model": correction_model,
        "num_samples": len(details),
        "num_chunks": int(sum(detail.get("num_chunks", 0) for detail in details)),
        "num_correction_units": int(sum(detail.get("num_correction_units", 0) for detail in details)),
        "num_corrected_units": num_corrected_units,
        "num_fallback_units": num_fallback_units,
        "num_warnings": int(sum(warning_counts.values())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "correction_time_sec": round(elapsed_sec, 3),
    }
    summary.update(correction_error_rate_summary(details))
    return summary


def correct_details(
    details: Sequence[dict],
    source_json: str,
    correction_backend=None,
    correction_model: str | None = None,
    max_new_tokens: int = 128,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    if correction_backend is None:
        if not correction_model:
            raise ValueError("correction_model is required when correction_backend is not provided")
        correction_backend = LocalCorrectionBackend(correction_model, max_new_tokens=max_new_tokens)

    started = time.time()
    corrected_details, prompt_records = correct_samples(details, correction_backend, progress_callback=progress_callback)
    summary = summarize_correction(corrected_details, source_json, correction_model or "", time.time() - started)
    return corrected_details, prompt_records, summary
