# -*- coding: utf-8 -*-
"""JSON IO helpers for C3 artifacts."""

from __future__ import annotations

import json
import os
from typing import Sequence


def load_json_list(path: str, description: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{description} must be a JSON list")
    return rows


def load_c2_predictions(c2_json: str) -> list[dict]:
    return load_json_list(c2_json, "c2_json produced by c2_asr.py --engine whisper_nbest")


def load_c3_details(details_json: str) -> list[dict]:
    return load_json_list(details_json, "C3 details JSON")


def write_json(path: str, data) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def write_jsonl(path: str, rows: Sequence[dict]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
