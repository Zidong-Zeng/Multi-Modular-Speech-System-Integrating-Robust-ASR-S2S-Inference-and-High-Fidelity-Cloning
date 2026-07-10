# -*- coding: utf-8 -*-
"""Evaluation metrics for C3 outputs."""

from __future__ import annotations

from typing import Sequence

from .text_assembly import normalize_text


def normalize_for_wer(text: str) -> list[str]:
    return normalize_text(text).split()


def normalize_for_cer(text: str) -> str:
    return normalize_text(text).replace(" ", "")


def edit_distance(ref, hyp) -> int:
    ref_len = len(ref)
    hyp_len = len(hyp)
    previous = list(range(hyp_len + 1))
    for i in range(1, ref_len + 1):
        current = [i] + [0] * hyp_len
        ref_char = ref[i - 1]
        for j in range(1, hyp_len + 1):
            cost = 0 if ref_char == hyp[j - 1] else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[hyp_len]


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    if not ref:
        return None
    return edit_distance(ref, hyp) / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float | None:
    ref = normalize_for_cer(reference)
    hyp = normalize_for_cer(hypothesis)
    if not ref:
        return None
    return edit_distance(ref, hyp) / len(ref)


def correction_error_rates(reference: str, hypothesis: str) -> dict:
    ref_words = normalize_for_wer(reference)
    hyp_words = normalize_for_wer(hypothesis)
    ref_chars = normalize_for_cer(reference)
    hyp_chars = normalize_for_cer(hypothesis)

    wer_edits = edit_distance(ref_words, hyp_words) if ref_words else 0
    cer_edits = edit_distance(ref_chars, hyp_chars) if ref_chars else 0
    return {
        "wer_after_correction": round(wer_edits / len(ref_words), 6) if ref_words else None,
        "wer_edits_after_correction": wer_edits,
        "wer_ref_words_after_correction": len(ref_words),
        "cer_after_correction": round(cer_edits / len(ref_chars), 6) if ref_chars else None,
        "cer_edits_after_correction": cer_edits,
        "cer_ref_chars_after_correction": len(ref_chars),
    }


def add_correction_error_rates(detail: dict) -> dict:
    row = dict(detail)
    row.update(
        correction_error_rates(
            str(row.get("reference", "")),
            str(row.get("corrected_transcript_en", "")),
        )
    )
    return row


def correction_error_rate_summary(details: Sequence[dict]) -> dict:
    total_wer_edits = 0
    total_ref_words = 0
    total_cer_edits = 0
    total_ref_chars = 0
    num_eval_samples = 0
    for detail in details:
        reference_words = normalize_for_wer(str(detail.get("reference", "")))
        reference_chars = normalize_for_cer(str(detail.get("reference", "")))
        hypothesis_words = normalize_for_wer(str(detail.get("corrected_transcript_en", "")))
        hypothesis_chars = normalize_for_cer(str(detail.get("corrected_transcript_en", "")))
        if not reference_words and not reference_chars:
            continue
        if reference_words:
            total_wer_edits += edit_distance(reference_words, hypothesis_words)
            total_ref_words += len(reference_words)
        if reference_chars:
            total_cer_edits += edit_distance(reference_chars, hypothesis_chars)
            total_ref_chars += len(reference_chars)
        num_eval_samples += 1

    return {
        "wer_after_correction": round(total_wer_edits / total_ref_words, 6) if total_ref_words else None,
        "num_wer_eval_samples": num_eval_samples if total_ref_words else 0,
        "wer_total_edits": total_wer_edits,
        "wer_total_ref_words": total_ref_words,
        "cer_after_correction": round(total_cer_edits / total_ref_chars, 6) if total_ref_chars else None,
        "num_cer_eval_samples": num_eval_samples if total_ref_chars else 0,
        "cer_total_edits": total_cer_edits,
        "cer_total_ref_chars": total_ref_chars,
    }


def correction_cer_summary(details: Sequence[dict]) -> dict:
    return correction_error_rate_summary(details)
