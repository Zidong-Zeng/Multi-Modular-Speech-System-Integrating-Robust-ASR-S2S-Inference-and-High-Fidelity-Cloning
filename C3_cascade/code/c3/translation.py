# -*- coding: utf-8 -*-
"""Translate corrected English units into Chinese."""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import Callable, Sequence

from .backends.hf_causal_lm import LocalTranslationBackend
from .prompts.translation import build_translation_prompt, clean_translation_output
from .text_assembly import assemble_translations_zh


def translate_unit(unit: dict, sample_id: str, translation_backend) -> tuple[dict, dict | None]:
    row = dict(unit)
    warnings = list(row.get("warnings", []))
    row["warnings"] = warnings
    corrected_en = str(row.get("corrected_en") or row.get("asr_top1_en", "")).strip()
    if not corrected_en:
        row["translation_zh"] = ""
        row.setdefault("translation_prompt", "")
        row.setdefault("translation_raw_output", "")
        warnings.append("empty_corrected_en_for_translation")
        return row, None

    prompt = build_translation_prompt(corrected_en)
    raw_output = translation_backend.translate(prompt)
    translation = clean_translation_output(raw_output)
    if not translation:
        warnings.append("empty_translation_output")

    row["translation_zh"] = translation
    row["translation_prompt"] = prompt
    row["translation_raw_output"] = raw_output
    prompt_record = {
        "id": sample_id,
        "unit_id": row.get("unit_id"),
        "source_chunk_ids": row.get("source_chunk_ids", []),
        "prompt": prompt,
        "raw_output": raw_output,
        "translation_zh": translation,
    }
    return row, prompt_record


def translate_samples(
    details: Sequence[dict],
    translation_backend,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[dict]]:
    translated_details = []
    prompt_records = []
    total_units = sum(len(sample.get("correction_units", [])) for sample in details)
    current_unit = 0
    for sample in details:
        translated_units = []
        for unit in sample.get("correction_units", []):
            current_unit += 1
            if progress_callback is not None:
                progress_callback(
                    f"[C3] Translation {current_unit}/{total_units}: "
                    f"id={sample.get('id')} unit={unit.get('unit_id')}"
                )
            translated_unit, prompt_record = translate_unit(unit, str(sample.get("id", "")), translation_backend)
            translated_units.append(translated_unit)
            if prompt_record is not None:
                prompt_records.append(prompt_record)

        translation_zh = assemble_translations_zh([unit.get("translation_zh", "") for unit in translated_units])
        detail = dict(sample)
        detail["correction_units"] = translated_units
        detail["translation_zh"] = translation_zh
        detail["num_correction_units"] = len(translated_units)
        translated_details.append(detail)
    return translated_details, prompt_records


def translation_prediction(detail: dict) -> dict:
    row = {
        "id": detail.get("id"),
        "audio": detail.get("audio"),
        "reference": detail.get("reference", ""),
        "hypothesis": detail.get("corrected_transcript_en", ""),
        "translation_zh": detail.get("translation_zh", ""),
    }
    if "emotion" in detail:
        row["emotion"] = detail.get("emotion")
    return row


def summarize_translation(
    details: Sequence[dict],
    source_json: str,
    translation_model: str,
    elapsed_sec: float,
) -> dict:
    warning_counts: Counter[str] = Counter()
    num_translated_units = 0
    for detail in details:
        for unit in detail.get("correction_units", []):
            warning_counts.update(unit.get("warnings", []))
            if unit.get("translation_zh"):
                num_translated_units += 1
    return {
        "phase": "translation",
        "source_json": os.path.abspath(source_json),
        "translation_model": translation_model,
        "num_samples": len(details),
        "num_chunks": int(sum(detail.get("num_chunks", 0) for detail in details)),
        "num_correction_units": int(sum(detail.get("num_correction_units", 0) for detail in details)),
        "num_translated_units": num_translated_units,
        "num_warnings": int(sum(warning_counts.values())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "translation_time_sec": round(elapsed_sec, 3),
    }


def translate_details(
    details: Sequence[dict],
    source_json: str,
    translation_backend=None,
    translation_model: str | None = None,
    max_new_tokens: int = 256,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    if translation_backend is None:
        if not translation_model:
            raise ValueError("translation_model is required when translation_backend is not provided")
        translation_backend = LocalTranslationBackend(translation_model, max_new_tokens=max_new_tokens)

    started = time.time()
    translated_details, prompt_records = translate_samples(details, translation_backend, progress_callback=progress_callback)
    summary = summarize_translation(translated_details, source_json, translation_model or "", time.time() - started)
    return translated_details, prompt_records, summary
