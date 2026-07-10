# -*- coding: utf-8 -*-
"""Translation prompt construction and output cleaning."""

from __future__ import annotations


def build_translation_prompt(corrected_en: str) -> str:
    return "\n".join(
        [
            "Translate the English transcript into Chinese.",
            "Preserve the meaning of the target transcript.",
            "Do not add explanations.",
            "Do not continue the transcript.",
            "Return only the Chinese translation.",
            "",
            "English transcript:",
            str(corrected_en or "").strip(),
            "",
            "Chinese translation:",
        ]
    ).strip()


def clean_translation_output(raw_output: str) -> str:
    text = str(raw_output or "").strip()
    markers = [
        "Chinese translation:",
        "Translation:",
        "涓枃缈昏瘧锛?,
        "涓枃缈昏瘧:",
        "缈昏瘧锛?,
        "缈昏瘧:",
    ]
    lowered = text.lower()
    for marker in markers:
        marker_lower = marker.lower()
        if marker_lower in lowered:
            index = lowered.rfind(marker_lower)
            text = text[index + len(marker) :].strip()
            break
    return text.strip().strip('"').strip()
