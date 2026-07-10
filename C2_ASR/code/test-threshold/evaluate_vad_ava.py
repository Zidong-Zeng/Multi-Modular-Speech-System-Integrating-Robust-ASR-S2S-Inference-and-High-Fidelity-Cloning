# -*- coding: utf-8 -*-
"""Evaluate Stage 1 VAD on AVA-Speech style speech-activity labels."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from vad_stage1 import EnergyVAD, FrameScore, compute_binary_metrics, compute_dcf, compute_roc_auc, read_wav_mono


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
MANIFEST_EXTS = {".json", ".jsonl", ".csv", ".tsv", ".txt"}
SPEECH_LABELS = {"speech", "clean_speech", "cleanspeech", "speech+music", "speech+noise", "human"}


def run_ava_vad_evaluation(
    data_root: str,
    output_json: str,
    manifest_path: str | None = None,
    thresholds: Sequence[float] | None = None,
    frame_ms: int = 30,
    sample_rate: int = 16000,
    max_items: int = 0,
    p_speech: float = 0.5,
    c_miss: float = 2.0,
    c_fa: float = 1.0,
) -> dict:
    thresholds = list(thresholds or [0.01, 0.03, 0.05, 0.1, 0.2, 0.35, 0.5])
    records = load_ava_records(data_root, manifest_path=manifest_path)
    if max_items and max_items > 0:
        records = records[:max_items]

    scores = []
    labels = []
    item_rows = []
    failures = []
    for record in records:
        try:
            audio, sr = read_wav_mono(record["audio_path"], target_sample_rate=sample_rate)
            vad = EnergyVAD(sample_rate=sr, frame_ms=frame_ms)
            frames = vad.score_frames(audio)
            frame_labels = frame_labels_from_segments(frames, record["segments"])
            scores.extend(frame.score for frame in frames)
            labels.extend(frame_labels)
            item_rows.append(
                {
                    "id": record.get("id") or Path(record["audio_path"]).stem,
                    "audio": record["audio_path"],
                    "num_frames": len(frames),
                    "speech_frames": int(sum(frame_labels)),
                    "audio_ms": frames[-1].end_ms if frames else 0,
                    "num_reference_segments": len(record["segments"]),
                }
            )
        except Exception as exc:
            failures.append({"id": record.get("id"), "audio": record.get("audio_path"), "error": str(exc)})

    roc = compute_roc_auc(scores, labels)
    threshold_rows = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(scores, labels, threshold)
        threshold_rows.append(
            {
                "threshold": float(threshold),
                **metrics,
                "dcf": round(compute_dcf(metrics, p_speech=p_speech, c_miss=c_miss, c_fa=c_fa), 6),
            }
        )
    best = min(threshold_rows, key=lambda row: row["dcf"]) if threshold_rows else {"threshold": None, "dcf": None}

    result = {
        "dataset": os.path.abspath(data_root),
        "manifest": os.path.abspath(manifest_path) if manifest_path else None,
        "backend": "energy",
        "params": {
            "frame_ms": frame_ms,
            "sample_rate": sample_rate,
            "thresholds": thresholds,
            "p_speech": p_speech,
            "c_miss": c_miss,
            "c_fa": c_fa,
        },
        "summary": {
            "num_records": len(records),
            "num_processed_records": len(item_rows),
            "num_failed_records": len(failures),
            "num_frames": len(scores),
            "speech_frame_ratio": round(sum(labels) / len(labels), 6) if labels else 0.0,
            "auc": roc["auc"],
            "best_threshold": best["threshold"],
            "best_dcf": best["dcf"],
        },
        "roc": roc,
        "threshold_metrics": threshold_rows,
        "items": item_rows,
        "failures": failures,
    }

    write_outputs(result, output_json)
    return result


def load_ava_records(data_root: str, manifest_path: str | None = None) -> list[dict]:
    data_root = os.path.abspath(data_root)
    if manifest_path is None:
        paired_records = load_paired_audio_label_records(data_root)
        if paired_records:
            return paired_records
    manifest_path = manifest_path or find_manifest(data_root)
    if not manifest_path:
        raise ValueError(f"No AVA manifest found under {data_root}; pass --manifest explicitly")
    manifest_path = os.path.abspath(manifest_path)
    ext = Path(manifest_path).suffix.lower()
    if ext == ".json":
        rows = load_json_rows(manifest_path)
    elif ext == ".jsonl":
        rows = load_jsonl_rows(manifest_path)
    elif ext in {".csv", ".tsv"}:
        rows = load_csv_rows(manifest_path, delimiter="\t" if ext == ".tsv" else ",")
    elif ext == ".txt":
        rows = load_txt_rows(manifest_path)
    else:
        raise ValueError(f"Unsupported manifest extension: {ext}")
    return normalize_manifest_rows(rows, data_root=os.path.dirname(manifest_path))


def load_paired_audio_label_records(data_root: str) -> list[dict]:
    records = []
    for audio_path in find_audio_files(data_root):
        label_path = find_paired_label_path(audio_path)
        if not label_path:
            continue
        with open(label_path, encoding="utf-8") as f:
            label_data = json.load(f)
        segments = extract_segments_from_label_data(label_data)
        if segments:
            records.append(
                {
                    "id": Path(audio_path).stem,
                    "audio_path": os.path.abspath(audio_path),
                    "segments": sorted(segments, key=lambda seg: (seg["start_ms"], seg["end_ms"])),
                }
            )
    return records


def find_audio_files(data_root: str) -> list[str]:
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(data_root, f"**/*{ext}"), recursive=True))
    return sorted(files)


def find_paired_label_path(audio_path: str) -> str | None:
    stem = os.path.splitext(audio_path)[0]
    for ext in (".json", ".txt", ".csv", ".tsv"):
        candidate = stem + ext
        if os.path.exists(candidate):
            return candidate
    return None


def extract_segments_from_label_data(data) -> list[dict]:
    if isinstance(data, list):
        segments = []
        for item in data:
            if isinstance(item, dict):
                segments.extend(extract_segments(item))
        return segments
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("segments"), list) or isinstance(data.get("speech_segments"), list):
        return extract_segments(data)
    starts = first_present(data, ["onset", "onsets", "start", "starts", "start_time", "start_times", "start_sec"])
    ends = first_present(data, ["offset", "offsets", "end", "ends", "end_time", "end_times", "end_sec"])
    labels = first_present(data, ["label", "labels", "class", "classes", "cluster", "clusters", "activity", "activities"])
    if starts is None or ends is None:
        return extract_segments(data)
    if not isinstance(starts, list):
        starts = [starts]
    if not isinstance(ends, list):
        ends = [ends]
    if labels is None:
        labels = ["speech"] * min(len(starts), len(ends))
    elif not isinstance(labels, list):
        labels = [labels] * min(len(starts), len(ends))
    segments = []
    for start, end, label in zip(starts, ends, labels):
        label = str(label or "speech").lower()
        if is_speech_label(label):
            segments.append(make_segment(start, end, scale=1000, label=label))
    return segments


def find_manifest(data_root: str) -> str | None:
    candidates = []
    for ext in MANIFEST_EXTS:
        candidates.extend(glob.glob(os.path.join(data_root, f"**/*{ext}"), recursive=True))
    candidates = [path for path in candidates if not Path(path).name.startswith(".")]
    if not candidates:
        return None
    # Prefer files that look like labels/manifests over metadata side files.
    keywords = ("manifest", "label", "annotation", "ava", "vad", "train", "test", "valid")
    candidates.sort(key=lambda p: (0 if any(k in Path(p).stem.lower() for k in keywords) else 1, len(p)))
    return candidates[0]


def load_json_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "records", "annotations"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("JSON manifest must be a list or contain data/items/records/annotations")


def load_jsonl_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv_rows(path: str, delimiter: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def load_txt_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.replace(",", " ").split()]
            if len(parts) >= 3:
                rows.append({"audio": parts[0], "start": parts[1], "end": parts[2], "label": parts[3] if len(parts) > 3 else "speech"})
    return rows


def normalize_manifest_rows(rows: Sequence[dict], data_root: str) -> list[dict]:
    grouped = {}
    for row in rows:
        audio = first_present(row, ["audio", "audio_path", "path", "file", "wav", "filename", "clip"])
        if not audio:
            continue
        audio_path = resolve_audio_path(str(audio), data_root)
        row_id = str(first_present(row, ["id", "utt_id", "clip_id", "video_id"]) or Path(audio_path).stem)
        segments = extract_segments(row)
        if not segments:
            continue
        key = audio_path
        if key not in grouped:
            grouped[key] = {"id": row_id, "audio_path": audio_path, "segments": []}
        grouped[key]["segments"].extend(segments)
    records = list(grouped.values())
    for record in records:
        record["segments"] = sorted(record["segments"], key=lambda seg: (seg["start_ms"], seg["end_ms"]))
    return records


def first_present(row: dict, keys: Sequence[str]):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def resolve_audio_path(audio: str, data_root: str) -> str:
    if os.path.isabs(audio):
        return audio
    direct = os.path.abspath(os.path.join(data_root, audio))
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(data_root, "**", audio), recursive=True)
    if matches:
        return os.path.abspath(matches[0])
    stem = Path(audio).stem
    for ext in AUDIO_EXTS:
        matches = glob.glob(os.path.join(data_root, "**", stem + ext), recursive=True)
        if matches:
            return os.path.abspath(matches[0])
    return direct


def extract_segments(row: dict) -> list[dict]:
    if isinstance(row.get("segments"), list):
        return [normalize_segment(seg) for seg in row["segments"] if normalize_segment(seg)]
    if isinstance(row.get("speech_segments"), list):
        return [normalize_segment(seg) for seg in row["speech_segments"] if normalize_segment(seg)]

    start = first_present(row, ["start_ms", "onset_ms", "begin_ms"])
    end = first_present(row, ["end_ms", "offset_ms", "stop_ms"])
    scale = 1
    if start is None or end is None:
        start = first_present(row, ["start", "onset", "begin", "start_time", "start_sec"])
        end = first_present(row, ["end", "offset", "stop", "end_time", "end_sec"])
        scale = 1000
    label = str(first_present(row, ["label", "class", "activity", "speech_label"]) or "speech").lower()
    if start is None or end is None or not is_speech_label(label):
        return []
    return [make_segment(start, end, scale=scale, label=label)]


def normalize_segment(segment: dict) -> dict | None:
    start = first_present(segment, ["start_ms", "onset_ms", "begin_ms"])
    end = first_present(segment, ["end_ms", "offset_ms", "stop_ms"])
    scale = 1
    if start is None or end is None:
        start = first_present(segment, ["start", "onset", "begin", "start_time", "start_sec"])
        end = first_present(segment, ["end", "offset", "stop", "end_time", "end_sec"])
        scale = 1000
    if start is None or end is None:
        return None
    label = str(first_present(segment, ["label", "class", "activity", "speech_label"]) or "speech").lower()
    if not is_speech_label(label):
        return None
    return make_segment(start, end, scale=scale, label=label)


def make_segment(start, end, scale: int, label: str) -> dict:
    start_ms = int(round(float(start) * scale))
    end_ms = int(round(float(end) * scale))
    return {"start_ms": start_ms, "end_ms": end_ms, "label": label}


def is_speech_label(label: str) -> bool:
    label = label.strip().lower().replace(" ", "_")
    if label in {"nospeech", "no_speech", "non_speech", "silence", "noise", "music"}:
        return False
    return label in SPEECH_LABELS or "speech" in label


def frame_labels_from_segments(
    frames: Sequence[FrameScore],
    segments: Sequence[dict],
    min_overlap_ratio: float = 0.5,
) -> list[int]:
    labels = []
    for frame in frames:
        frame_len = max(1, frame.end_ms - frame.start_ms)
        label = 0
        for segment in segments:
            overlap = min(frame.end_ms, int(segment["end_ms"])) - max(frame.start_ms, int(segment["start_ms"]))
            if overlap / frame_len >= min_overlap_ratio:
                label = 1
                break
        labels.append(label)
    return labels


def write_outputs(result: dict, output_json: str) -> None:
    outdir = os.path.dirname(os.path.abspath(output_json))
    os.makedirs(outdir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    write_threshold_csv(result["threshold_metrics"], os.path.join(outdir, "ava_vad_threshold_metrics.csv"))
    write_roc_plot(result["roc"]["points"], os.path.join(outdir, "ava_vad_roc.png"))
    write_dcf_plot(result["threshold_metrics"], os.path.join(outdir, "ava_vad_dcf.png"))


def write_threshold_csv(rows: Sequence[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_roc_plot(points: Sequence[dict], path: str) -> None:
    if not points:
        return
    ordered = sorted(points, key=lambda p: p["fpr"])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([p["fpr"] for p in ordered], [p["tpr"] for p in ordered], marker="o", linewidth=1.6)
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1)
    ax.set_title("AVA-Speech VAD ROC")
    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_dcf_plot(rows: Sequence[dict], path: str) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: row["threshold"])
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot([row["threshold"] for row in rows], [row["dcf"] for row in rows], marker="o", linewidth=1.8, color="#dc2626")
    ax.set_title("AVA-Speech VAD DCF by threshold")
    ax.set_xlabel("VAD threshold")
    ax.set_ylabel("DCF")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_thresholds(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate EnergyVAD on AVA-Speech style labels")
    ap.add_argument("--data_root", default="/root/siton-tmp/assignment_C/C2_ASR/data/AVA-Speech")
    ap.add_argument("--manifest", default=None, help="Optional manifest path; auto-detected if omitted")
    ap.add_argument("--out", required=True, help="Output evaluation JSON")
    ap.add_argument("--thresholds", default="0.01,0.03,0.05,0.1,0.2,0.35,0.5")
    ap.add_argument("--frame_ms", type=int, default=30)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--p_speech", type=float, default=0.5)
    ap.add_argument("--c_miss", type=float, default=2.0)
    ap.add_argument("--c_fa", type=float, default=1.0)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_ava_vad_evaluation(
        data_root=args.data_root,
        output_json=args.out,
        manifest_path=args.manifest,
        thresholds=parse_thresholds(args.thresholds),
        frame_ms=args.frame_ms,
        sample_rate=args.sample_rate,
        max_items=args.max_items,
        p_speech=args.p_speech,
        c_miss=args.c_miss,
        c_fa=args.c_fa,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
