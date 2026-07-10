# -*- coding: utf-8 -*-
"""Stage 3: assemble chunk ASR results and compute full-audio metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Sequence

from asr_interfaces import ASRBackend, TranscriptionResult, WhisperASRBackend


# Text normalization keeps WER/CER comparisons stable across casing and punctuation.
def normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Edit distance is the shared engine behind lightweight WER and CER metrics.
def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    prev = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        curr = [i]
        for j, hyp_item in enumerate(hyp, start=1):
            cost = 0 if ref_item == hyp_item else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


# WER measures word-level recognition quality for the assembled long transcript.
def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


# CER complements WER by measuring character-level recognition differences.
def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


# Overlap-aware assembly removes duplicated boundary words from adjacent chunks.
def assemble_chunk_texts(chunks: Sequence[dict], max_overlap_words: int = 8) -> str:
    ordered = sorted(chunks, key=lambda chunk: (int(chunk.get("start_ms", 0)), int(chunk.get("chunk_id", 0))))
    words: list[str] = []
    for chunk in ordered:
        chunk_words = normalize_text(chunk.get("text", "")).split()
        if not chunk_words:
            continue
        overlap = find_word_overlap(words, chunk_words, max_overlap_words=max_overlap_words)
        words.extend(chunk_words[overlap:])
    return " ".join(words)


# Boundary duplicate detection finds the longest suffix-prefix word match.
def find_word_overlap(left_words: Sequence[str], right_words: Sequence[str], max_overlap_words: int = 8) -> int:
    max_n = min(len(left_words), len(right_words), max_overlap_words)
    for n in range(max_n, 0, -1):
        if list(left_words[-n:]) == list(right_words[:n]):
            return n
    return 0


# Stage 3 main runner consumes Stage 2 chunks, fills missing text, and writes full results.
def run_stage3(
    stage2_json: str,
    output_json: str,
    asr_backend: ASRBackend | None = None,
    reference_json: str | None = None,
    max_overlap_words: int = 8,
) -> dict:
    with open(stage2_json, encoding="utf-8") as f:
        stage2 = json.load(f)
    references = load_references(reference_json) if reference_json else {}
    items = stage2.get("items", [])
    results = []
    failures = []

    for item in items:
        try:
            processed_chunks = transcribe_missing_chunks(item.get("chunks", []), asr_backend=asr_backend)
            full_text = assemble_chunk_texts(processed_chunks, max_overlap_words=max_overlap_words)
            reference = item.get("reference") or references.get(str(item.get("id")), "")
            infer_time_sec = sum(int(chunk.get("latency_ms", 0) or 0) for chunk in processed_chunks) / 1000.0
            audio_duration_sec = resolve_audio_duration_sec(item)
            results.append(
                {
                    "id": item.get("id"),
                    "audio": item.get("audio"),
                    "audio_path": item.get("audio_path"),
                    "strategy": "vad_dynamic_chunk_whisper",
                    "reference": reference,
                    "full_text": full_text,
                    "wer": round(compute_wer(reference, full_text), 6) if reference else None,
                    "cer": round(compute_cer(reference, full_text), 6) if reference else None,
                    "rtf": round(infer_time_sec / audio_duration_sec, 6) if audio_duration_sec > 0 else None,
                    "audio_duration_sec": round(audio_duration_sec, 3),
                    "infer_time_sec": round(infer_time_sec, 3),
                    "num_chunks": len(processed_chunks),
                    "chunks": processed_chunks,
                }
            )
        except Exception as exc:
            failures.append({"id": item.get("id"), "error": str(exc)})

    result = {
        "stage": "stage3_full_asr_assembly",
        "stage2_json": os.path.abspath(stage2_json),
        "reference_json": os.path.abspath(reference_json) if reference_json else None,
        "summary": summarize_results(results, failures),
        "items": results,
        "failures": failures,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


# Missing chunk text is filled through an injected ASR backend without hard-coding Whisper.
def transcribe_missing_chunks(chunks: Sequence[dict], asr_backend: ASRBackend | None = None) -> list[dict]:
    processed = []
    for chunk in chunks:
        row = dict(chunk)
        if not str(row.get("text", "")).strip():
            if asr_backend is None:
                raise ValueError(f"chunk {row.get('chunk_id')} has no text and no ASR backend was provided")
            audio_path = row.get("chunk_audio") or row.get("audio_path")
            if not audio_path:
                raise ValueError(f"chunk {row.get('chunk_id')} has no chunk_audio/audio_path for ASR")
            result = asr_backend.transcribe(audio_path)
            if isinstance(result, TranscriptionResult):
                row["text"] = result.text
                row["latency_ms"] = result.latency_ms
            else:
                row["text"] = str(result)
                row.setdefault("latency_ms", 0)
        processed.append(row)
    return processed


# Reference loader maps dataset rows to text for WER/CER evaluation.
def load_references(reference_json: str) -> dict[str, str]:
    with open(reference_json, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("reference_json must be a list of dataset rows")
    refs = {}
    for row in rows:
        if "id" in row and "text" in row:
            refs[str(row["id"])] = row["text"]
    return refs


# Audio duration is read from Stage 2 summaries to avoid reopening audio files.
def resolve_audio_duration_sec(item: dict) -> float:
    summary = item.get("summary", {})
    audio_ms = int(summary.get("audio_ms", 0) or 0)
    if audio_ms <= 0:
        chunks = item.get("chunks", [])
        audio_ms = max((int(chunk.get("end_ms", 0) or 0) for chunk in chunks), default=0)
    return audio_ms / 1000.0


# Final summary aggregates item-level quality and efficiency metrics.
def summarize_results(items: Sequence[dict], failures: Sequence[dict]) -> dict:
    wers = [item["wer"] for item in items if item.get("wer") is not None]
    cers = [item["cer"] for item in items if item.get("cer") is not None]
    rtfs = [item["rtf"] for item in items if item.get("rtf") is not None]
    return {
        "num_items": len(items),
        "num_failed_items": len(failures),
        "num_chunks": int(sum(item.get("num_chunks", 0) for item in items)),
        "mean_wer": round(sum(wers) / len(wers), 6) if wers else None,
        "mean_cer": round(sum(cers) / len(cers), 6) if cers else None,
        "mean_rtf": round(sum(rtfs) / len(rtfs), 6) if rtfs else None,
        "total_audio_duration_sec": round(sum(float(item.get("audio_duration_sec", 0.0)) for item in items), 3),
        "total_infer_time_sec": round(sum(float(item.get("infer_time_sec", 0.0)) for item in items), 3),
    }


# CLI keeps local assembly and server Whisper inference behind the same Stage 3 entrypoint.
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 3 full ASR assembly and WER/CER/RTF evaluation")
    ap.add_argument("--stage2_json", required=True, help="Stage 2 long_asr_chunks JSON")
    ap.add_argument("--out", required=True, help="Output long_asr_full JSON")
    ap.add_argument("--reference_json", default=None, help="Optional dataset JSON with id/text references")
    ap.add_argument("--max_overlap_words", type=int, default=8)
    ap.add_argument("--run_asr", action="store_true", help="Transcribe chunks missing text with Whisper")
    ap.add_argument("--model", default="/root/siton-tmp/assignment_C/model/whisper-large-v3")
    ap.add_argument("--batch_size", type=int, default=1)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    backend = WhisperASRBackend(args.model, batch_size=args.batch_size) if args.run_asr else None
    result = run_stage3(
        stage2_json=args.stage2_json,
        output_json=args.out,
        asr_backend=backend,
        reference_json=args.reference_json,
        max_overlap_words=args.max_overlap_words,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
