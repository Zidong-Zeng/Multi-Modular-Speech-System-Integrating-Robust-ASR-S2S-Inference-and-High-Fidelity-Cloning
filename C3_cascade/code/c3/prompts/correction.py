# -*- coding: utf-8 -*-
"""Correction prompt construction and output cleaning."""

from __future__ import annotations

import json
import re


ASR_CORRECTION_PROMPT = [
    "You are an ASR error correction model.",
    "",
    "You will receive an n-best transcription list produced by an ASR system for one audio segment.",
    "",
    "Core fact about the input:",
    "The rank order of the candidates does NOT reflect correctness. The ASR model's confidence ranking is unreliable 鈥?,
    "the correct transcript may be the lowest-ranked candidate, a candidate that appears only once, or may not exist verbatim in any candidate at all.",
    "Treat all candidates as equally weighted evidence, not a ranked list to pick from.",
    "",
    "Your task: infer the single most likely correct transcript by reasoning across all candidates together, using:",
    "- Words, phrases, or homophones that recur across candidates. But be aware that a shared pattern can also be a systematic ASR error repeated across the beam search, not real evidence 鈥?weigh plausibility over raw frequency.",
    "- General grammar, common collocations, and semantic plausibility.",
    "- Likely ASR failure modes: homophones, word-boundary splits or merges, and dropped or inserted function words.",
    "",
    "You are not limited to copying any single candidate verbatim.",
    "If no candidate is fully correct, synthesize the best transcript by combining fragments from different candidates, as long as every fragment is attested in at least one candidate.",
    "Do not introduce content words, facts, or meaning not supported by at least one candidate.",
    "If nothing in the candidates points to a better answer than the top-ranked one, output the top-ranked candidate as is 鈥?do not \"correct\" it based on outside world knowledge or what a sentence \"probably\" should say.",
    "Do not continue the sentence, add commentary, or translate.",
    "",
    "Formatting rule - spell out numbers:",
    "Any number that would be spoken aloud (times, counts, quantities, etc.) must appear as English words, never as digits.",
    "Example: '11 o'clock' -> 'eleven o'clock', '7' -> 'seven'.",
    "Apply this even if none of the candidates already contain a spelled-out form 鈥?generate it yourself.",
    "",
    "Final requirement - no grammatical errors:",
    "corrected_text must be a complete, natural, grammatically correct English sentence.",
    "Before finalizing your answer, check it against this requirement. If it fails, revise it until it passes 鈥?even if that means deviating further from the literal candidate wording.",
    "",
    'Output exactly one JSON object and nothing else: {"corrected_text": "...", "rule_applied": "consensus" | "plausibility" | "both" | "synthesis"}',
    "",
    "Example 1 - consensus overrides rank 1:",
    "Input n-best:",
    '1. "Give me an aim."',
    '2. "Give me a name."',
    '3. "Give me the name."',
    '4. "Give me a Name."',
    '5. "Give me an name."',
    'Output: {"corrected_text": "Give me a name.", "rule_applied": "consensus"}',
    "",
    "Example 2 - both consensus and plausibility agree:",
    "Input n-best:",
    '1. "She\'s going to the the store."',
    '2. "She\'s going to the store."',
    '3. "She\'s going to store."',
    '4. "She is going to the store."',
    '5. "Shes going to the store."',
    'Output: {"corrected_text": "She\'s going to the store.", "rule_applied": "both"}',
    "",
    "Example 3 - rank and vote count are misleading; the correct answer is a low-frequency candidate:",
    "Input n-best:",
    '1. "I would like to do a large clock."',
    '2. "I would like to do a hard clock!"',
    '3. "I would like to do an art clock."',
    '4. "I would like to do a dark clock."',
    '5. "I would like a new alarm clock."',
    'Output: {"corrected_text": "I would like a new alarm clock.", "rule_applied": "plausibility"}',
    "",
    "Example 4 - synthesize from fragments scattered across multiple candidates:",
    "Input n-best:",
    '1. "Maybe tomorrow will be cold."',
    '2. "Maybe tomorrow will be cool."',
    '3. "It\'d be tomorrow it would be cool."',
    '4. "It will be tomorrow. Be cold."',
    '5. "And maybe tomorrow will be cold"',
    'Output: {"corrected_text": "Maybe tomorrow it will be cold.", "rule_applied": "synthesis"}',
    "",
    "Example 5 - spell out numbers even when no candidate already uses the spelled-out form:",
    "Input n-best:",
    '1. "It\'s 11 o\'clock."',
    '2. "It\'s 11 oclock,"',
    '3. "It\'s live in a clock."',
    '4. "It\'s 11 a clock."',
    '5. "It\'s 7 o\'clock."',
    'Output: {"corrected_text": "It\'s eleven o\'clock.", "rule_applied": "consensus"}',
    "",
    "Now correct the following content:",
    "Input n-best:",
]


def build_correction_prompt(unit: dict) -> str:
    lines = list(ASR_CORRECTION_PROMPT)
    for chunk in unit.get("asr_nbest_by_chunk", []):
        for candidate in chunk.get("nbest", []):
            rank = candidate.get("rank", "")
            text = str(candidate.get("text", "")).strip()
            if text:
                lines.append(f"{rank}. {text}")
    lines.append("Output:")
    return "\n".join(lines).strip()


def clean_correction_output(raw_output: str) -> str:
    text = strip_thinking_blocks(str(raw_output or "").strip())
    parsed_text = _extract_corrected_text_from_json(text)
    if parsed_text:
        return parsed_text

    marker = "Corrected English transcript:"
    if marker.lower() in text.lower():
        index = text.lower().rfind(marker.lower())
        text = text[index + len(marker) :].strip()
    parsed_text = _extract_corrected_text_from_json(text)
    if parsed_text:
        return parsed_text
    return text.strip().strip('"').strip()


def strip_thinking_blocks(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    open_match = re.search(r"<think\b[^>]*>", text, flags=re.IGNORECASE)
    if open_match:
        text = text[: open_match.start()].strip()
    text = re.sub(r"</think>", "", text, flags=re.IGNORECASE).strip()
    return text


def _extract_corrected_text_from_json(text: str) -> str:
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            corrected = str(data.get("corrected_text", "")).strip()
            if corrected:
                return corrected.strip().strip('"').strip()
    return ""
