# -*- coding: utf-8 -*-
"""Build a Stage 1-compatible dataset manifest from AVA-style directories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
LABEL_EXTS = (".json", ".txt", ".csv", ".tsv")


# Directory scanning converts AVA's paired files into the project's shared dataset schema.
def build_ava_manifest(
    data_root: str,
    split: str | None = None,
    max_items: int = 0,
    path_base: str | None = None,
) -> list[dict]:
    data_root = os.path.abspath(data_root)
    split = split or Path(data_root).name
    path_base = os.path.abspath(path_base or data_root)
    items = []
    for audio_path in iter_audio_files(data_root):
        label_path = find_label_path(audio_path)
        if not label_path:
            continue
        rel_audio = os.path.relpath(audio_path, path_base).replace("\\", "/")
        rel_label = os.path.relpath(label_path, path_base).replace("\\", "/")
        item_id = Path(audio_path).stem
        items.append(
            {
                "id": item_id,
                "audio": rel_audio,
                "audio_path": rel_audio,
                "text": "",
                "label_path": rel_label,
                "source_dataset": "AVA-Speech",
                "split": split,
            }
        )
        if max_items and max_items > 0 and len(items) >= max_items:
            break
    return items


# Audio iteration is deterministic so manifests are stable across runs.
def iter_audio_files(data_root: str):
    for dirpath, _, filenames in os.walk(data_root):
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() in AUDIO_EXTS:
                yield os.path.join(dirpath, filename)


# Label pairing follows the AVA convention of same-stem annotation side files.
def find_label_path(audio_path: str) -> str | None:
    stem = os.path.splitext(audio_path)[0]
    for ext in LABEL_EXTS:
        candidate = stem + ext
        if os.path.exists(candidate):
            return candidate
    return None


# CLI writes the manifest once so Stage 1/2/3 can reuse the same dataset index.
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build an AVA-Speech dataset manifest for Stage 1 VAD pipelines")
    ap.add_argument("--data_root", default="/root/siton-tmp/assignment_C/C2_ASR/data/libris-merge", help="Root directory of an AVA split with paired audio/label files")
    ap.add_argument("--out", default="/root/siton-tmp/assignment_C/C2_ASR/data/libris_manifest.json", help="Output dataset JSON path")
    ap.add_argument("--split", default=None, help="Optional split name; defaults to the data_root directory name")
    ap.add_argument("--max_items", type=int, default=0, help="Limit manifest size for debugging; 0 means all")
    ap.add_argument("--path_base", default="/root/siton-tmp/assignment_C/C2_ASR/data", help="Base directory used to write relative audio/label paths")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = build_ava_manifest(args.data_root, split=args.split, max_items=args.max_items, path_base=args.path_base)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({"num_items": len(manifest), "data_root": os.path.abspath(args.data_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
