# -*- coding: utf-8 -*-
"""Build correction units from C2 per-chunk N-best outputs."""

from __future__ import annotations

import os
from collections import Counter
from typing import Sequence

from .schemas import CorrectionUnitConfig
from .text_assembly import assemble_texts, word_count


def normalized_chunk(chunk: dict, sample_id: str) -> dict:
    if "nbest" not in chunk or not isinstance(chunk.get("nbest"), list) or not chunk.get("nbest"):
        raise ValueError(f"sample {sample_id} chunk {chunk.get('chunk_id')} missing nbest")

    row = dict(chunk)
    row["chunk_id"] = int(row.get("chunk_id", 0))
    row["start_ms"] = int(row.get("start_ms", 0) or 0)
    if "end_ms" in row:
        row["end_ms"] = int(row.get("end_ms", 0) or 0)
    else:
        row["end_ms"] = row["start_ms"] + int(row.get("duration_ms", 0) or 0)
    row["duration_ms"] = max(0, row["end_ms"] - row["start_ms"])
    row["text"] = str(row.get("text") or row["nbest"][0].get("text", "")).strip()
    return row


def sorted_valid_chunks(sample: dict) -> list[dict]:
    chunks = sample.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"sample {sample.get('id')} chunks must be a list")
    sample_id = str(sample.get("id", ""))
    rows = [normalized_chunk(chunk, sample_id) for chunk in chunks]
    return sorted(rows, key=lambda row: (row["start_ms"], row["chunk_id"]))


def unit_duration_ms(chunks: Sequence[dict]) -> int:
    if not chunks:
        return 0
    return max(int(chunk["end_ms"]) for chunk in chunks) - min(int(chunk["start_ms"]) for chunk in chunks)


def unit_text(chunks: Sequence[dict], config: CorrectionUnitConfig) -> str:
    return assemble_texts([str(chunk.get("text", "")) for chunk in chunks], max_overlap_words=config.max_overlap_words)


def unit_is_ready(chunks: Sequence[dict], config: CorrectionUnitConfig) -> bool:
    return unit_duration_ms(chunks) >= config.min_unit_ms and word_count(unit_text(chunks, config)) >= config.min_unit_words


def crosses_speaker_boundary(current_chunks: Sequence[dict], next_chunk: dict, config: CorrectionUnitConfig) -> bool:
    if not config.respect_speaker_boundary:
        return False
    current_speakers = {str(chunk.get("speaker")) for chunk in current_chunks if chunk.get("speaker")}
    next_speaker = str(next_chunk.get("speaker")) if next_chunk.get("speaker") else None
    return bool(current_speakers and next_speaker and next_speaker not in current_speakers)


def can_merge_next(current_chunks: Sequence[dict], next_chunk: dict, config: CorrectionUnitConfig) -> bool:
    current_end_ms = max(int(chunk["end_ms"]) for chunk in current_chunks)
    gap_ms = int(next_chunk["start_ms"]) - current_end_ms
    if gap_ms > config.max_merge_gap_ms:
        return False
    if crosses_speaker_boundary(current_chunks, next_chunk, config):
        return False

    candidate_chunks = list(current_chunks) + [next_chunk]
    candidate_duration = unit_duration_ms(candidate_chunks)
    if candidate_duration > config.hard_max_ms:
        return False
    if word_count(unit_text(candidate_chunks, config)) > config.max_unit_words:
        return False
    if unit_is_ready(current_chunks, config) and candidate_duration > config.target_max_ms:
        return False
    return True


def nbest_by_chunk(chunks: Sequence[dict]) -> list[dict]:
    rows = []
    for chunk in chunks:
        row = {
            "chunk_id": chunk.get("chunk_id"),
            "start_ms": chunk.get("start_ms"),
            "end_ms": chunk.get("end_ms"),
            "nbest": chunk.get("nbest", []),
        }
        if chunk.get("speaker"):
            row["speaker"] = chunk.get("speaker")
        rows.append(row)
    return rows


def build_unit(unit_id: int, chunks: Sequence[dict], config: CorrectionUnitConfig) -> dict:
    text = unit_text(chunks, config)
    duration_ms = unit_duration_ms(chunks)
    words = word_count(text)
    warnings = []
    correction_mode = "generate"
    corrected_en = ""

    if duration_ms < config.fallback_min_ms or words < config.fallback_min_words:
        correction_mode = "fallback_top1"
        corrected_en = text
        warnings.append("too_short_for_correction")

    speakers = sorted({str(chunk.get("speaker")) for chunk in chunks if chunk.get("speaker")})
    row = {
        "unit_id": unit_id,
        "source_chunk_ids": [int(chunk.get("chunk_id", 0)) for chunk in chunks],
        "start_ms": min(int(chunk["start_ms"]) for chunk in chunks),
        "end_ms": max(int(chunk["end_ms"]) for chunk in chunks),
        "duration_ms": duration_ms,
        "word_count": words,
        "asr_top1_en": text,
        "corrected_en": corrected_en,
        "translation_zh": "",
        "asr_nbest_by_chunk": nbest_by_chunk(chunks),
        "correction_mode": correction_mode,
        "warnings": warnings,
    }
    if speakers:
        row["speakers"] = speakers
    return row


def build_correction_units(sample: dict, config: CorrectionUnitConfig | None = None) -> list[dict]:
    config = config or CorrectionUnitConfig()
    chunks = sorted_valid_chunks(sample)
    units = []
    index = 0
    while index < len(chunks):
        current = [chunks[index]]
        index += 1
        while index < len(chunks) and can_merge_next(current, chunks[index], config):
            current.append(chunks[index])
            index += 1
        units.append(build_unit(len(units), current, config))
    return units


def build_c3_prediction(sample: dict, config: CorrectionUnitConfig) -> dict:
    units = build_correction_units(sample, config)
    chunks = sorted_valid_chunks(sample)
    asr_top1 = assemble_texts([chunk.get("text", "") for chunk in chunks], max_overlap_words=config.max_overlap_words)
    corrected_units = [unit["corrected_en"] for unit in units if unit.get("corrected_en")]
    corrected_transcript = assemble_texts(corrected_units, max_overlap_words=config.max_overlap_words)
    return {
        "id": sample.get("id"),
        "audio": sample.get("audio"),
        "reference": sample.get("reference", sample.get("text", "")),
        "emotion": sample.get("emotion"),
        "engine": sample.get("engine"),
        "model": sample.get("model"),
        "asr_top1_transcript_en": asr_top1,
        "corrected_transcript_en": corrected_transcript,
        "translation_zh": "",
        "num_chunks": len(chunks),
        "num_correction_units": len(units),
        "correction_units": units,
    }


def build_c3_predictions(c2_predictions: Sequence[dict], config: CorrectionUnitConfig | None = None) -> list[dict]:
    config = config or CorrectionUnitConfig()
    return [build_c3_prediction(sample, config) for sample in c2_predictions]


def summarize_unit_construction(predictions: Sequence[dict], c2_json: str, config: CorrectionUnitConfig) -> dict:
    warning_counts: Counter[str] = Counter()
    num_fallback_units = 0
    for prediction in predictions:
        for unit in prediction.get("correction_units", []):
            warning_counts.update(unit.get("warnings", []))
            if unit.get("correction_mode") == "fallback_top1":
                num_fallback_units += 1

    return {
        "phase": "correction_unit_construction",
        "source_c2_json": os.path.abspath(c2_json),
        "num_samples": len(predictions),
        "num_chunks": int(sum(prediction.get("num_chunks", 0) for prediction in predictions)),
        "num_correction_units": int(sum(prediction.get("num_correction_units", 0) for prediction in predictions)),
        "num_fallback_units": num_fallback_units,
        "num_warnings": int(sum(warning_counts.values())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "config": config.to_dict(),
    }
