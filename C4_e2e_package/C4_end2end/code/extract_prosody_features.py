# -*- coding: utf-8 -*-
"""Extract lightweight prosody features from the shared C dataset."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

from prosody_utils import EMOTION_LABELS, extract_prosody_features, resolve_audio_path


HERE = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(HERE, "..", "common_data", "dataset.json"))
    parser.add_argument("--out", default=os.path.join(HERE, "..", "C4_end2end", "emotion_controller", "prosody_features.json"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=0, help="0 means all remaining samples.")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    dataset = load_json(args.dataset)
    data_root = os.path.dirname(os.path.abspath(args.dataset))
    end = len(dataset) if args.count <= 0 else min(len(dataset), args.start + args.count)
    samples = dataset[args.start:end]

    rows = []
    failures = []
    t0 = time.time()
    for local_idx, sample in enumerate(samples):
        global_idx = args.start + local_idx
        audio_file = resolve_audio_path(data_root, sample)
        try:
            features = extract_prosody_features(audio_file, target_sr=args.sample_rate)
            rows.append({
                "index": global_idx,
                "id": sample.get("id", ""),
                "audio": sample.get("audio", ""),
                "speaker": sample.get("speaker", ""),
                "emotion": sample.get("emotion", ""),
                "features": features,
            })
        except Exception as exc:
            failures.append({
                "index": global_idx,
                "id": sample.get("id", ""),
                "audio": sample.get("audio", ""),
                "error": repr(exc),
            })

        if (local_idx + 1) % 100 == 0 or local_idx + 1 == len(samples):
            print(f"[INFO] extracted {local_idx + 1}/{len(samples)}")

    summary = {
        "dataset": os.path.abspath(args.dataset),
        "num_samples": len(rows),
        "num_failures": len(failures),
        "range": [args.start, end],
        "sample_rate": args.sample_rate,
        "elapsed_sec": round(time.time() - t0, 1),
        "emotion_counts": dict(Counter(row["emotion"] for row in rows if row["emotion"] in EMOTION_LABELS)),
        "failures": failures[:20],
    }
    save_json({"summary": summary, "items": rows}, args.out)
    print(f"[DONE] saved {args.out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

