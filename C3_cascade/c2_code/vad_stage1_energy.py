# -*- coding: utf-8 -*-
"""Stage 1 VAD detection and evaluation utilities.

This module intentionally keeps a lightweight energy-based backend so the
pipeline can be tested without downloading VAD models. The public functions are
backend-agnostic and can later wrap Silero VAD with the same segment format.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import wave
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

@dataclass(frozen=True)
class FrameScore:
    start_ms: int
    end_ms: int
    score: float


class EnergyVAD:
    """Simple frame-level VAD based on normalized RMS energy."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, threshold: float = 0.2):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.threshold = threshold

    def score_frames(self, audio: np.ndarray) -> List[FrameScore]:
        audio = _as_float_mono(audio)
        frame_len = max(1, int(self.sample_rate * self.frame_ms / 1000))
        rms_values = []
        spans: List[Tuple[int, int]] = []

        for start in range(0, len(audio), frame_len):
            end = min(len(audio), start + frame_len)
            frame = audio[start:end]
            rms = float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0
            rms_values.append(rms)
            spans.append((start, end))

        max_rms = max(rms_values) if rms_values else 0.0
        denom = max(max_rms, 1e-9)
        frames = []
        for (start, end), rms in zip(spans, rms_values):
            frames.append(
                FrameScore(
                    start_ms=round(start / self.sample_rate * 1000),
                    end_ms=round(end / self.sample_rate * 1000),
                    score=min(1.0, rms / denom),
                )
            )
        return frames


def read_wav_mono(path: str, target_sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """Read a PCM WAV file, convert to mono float32, and optionally resample."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if target_sample_rate and sample_rate != target_sample_rate:
        data = resample_linear(data, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    return _as_float_mono(data), sample_rate


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")
    if orig_sr == target_sr:
        return _as_float_mono(audio)
    audio = _as_float_mono(audio)
    if len(audio) == 0:
        return audio
    old_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    new_len = max(1, int(round(len(audio) * target_sr / orig_sr)))
    new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def merge_speech_frames(
    frames: Sequence[FrameScore],
    threshold: float,
    min_speech_ms: int = 250,
    min_silence_ms: int = 200,
    speech_pad_ms: int = 80,
) -> List[dict]:
    """Convert frame scores into merged speech segments."""
    raw_segments = []
    active_start = None
    active_scores = []

    for frame in frames:
        is_speech = frame.score >= threshold
        if is_speech and active_start is None:
            active_start = frame.start_ms
            active_scores = []
        if active_start is not None:
            active_scores.append(frame.score)
        if not is_speech and active_start is not None:
            raw_segments.append((active_start, frame.start_ms, active_scores[:-1]))
            active_start = None
            active_scores = []

    if active_start is not None and frames:
        raw_segments.append((active_start, frames[-1].end_ms, active_scores))

    filtered = []
    for start, end, scores in raw_segments:
        if end - start >= min_speech_ms:
            filtered.append([start, end, scores])

    merged = []
    for start, end, scores in filtered:
        if merged and start - merged[-1][1] <= min_silence_ms:
            merged[-1][1] = end
            merged[-1][2].extend(scores)
        else:
            merged.append([start, end, list(scores)])

    segments = []
    max_end = frames[-1].end_ms if frames else 0
    for start, end, scores in merged:
        start = max(0, start - speech_pad_ms)
        end = min(max_end, end + speech_pad_ms)
        confidence = float(np.mean(scores)) if scores else 0.0
        segments.append(
            {
                "start_ms": int(start),
                "end_ms": int(end),
                "duration_ms": int(max(0, end - start)),
                "confidence": round(confidence, 4),
            }
        )
    return segments


def compute_binary_metrics(scores: Sequence[float], labels: Sequence[int], threshold: float) -> dict:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    tp = tn = fp = fn = 0
    for score, label in zip(scores, labels):
        pred = 1 if score >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        else:
            fn += 1
    positives = tp + fn
    negatives = tn + fp
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "miss_rate": fn / positives if positives else 0.0,
        "false_alarm_rate": fp / negatives if negatives else 0.0,
        "tpr": tp / positives if positives else 0.0,
        "fpr": fp / negatives if negatives else 0.0,
    }


def compute_dcf(metrics: dict, p_speech: float = 0.5, c_miss: float = 2.0, c_fa: float = 1.0) -> float:
    return float(c_miss * metrics["miss_rate"] * p_speech + c_fa * metrics["false_alarm_rate"] * (1.0 - p_speech))


def compute_roc_auc(scores: Sequence[float], labels: Sequence[int]) -> dict:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        return {"auc": 0.0, "points": []}

    thresholds = sorted(set(float(s) for s in scores), reverse=True)
    thresholds = [math.inf] + thresholds + [-math.inf]
    points = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(scores, labels, threshold)
        points.append({"threshold": threshold, "tpr": metrics["tpr"], "fpr": metrics["fpr"]})

    ordered = sorted(points, key=lambda p: p["fpr"])
    auc = 0.0
    for left, right in zip(ordered, ordered[1:]):
        width = right["fpr"] - left["fpr"]
        height = (right["tpr"] + left["tpr"]) / 2.0
        auc += width * height
    return {"auc": round(float(auc), 6), "points": _json_safe_points(points)}


def run_vad_file(
    audio_path: str,
    output_json: str | None = None,
    threshold: float = 0.2,
    frame_ms: int = 30,
    min_speech_ms: int = 250,
    min_silence_ms: int = 200,
    speech_pad_ms: int = 80,
    sample_rate: int = 16000,
) -> dict:
    audio, sr = read_wav_mono(audio_path, target_sample_rate=sample_rate)
    vad = EnergyVAD(sample_rate=sr, frame_ms=frame_ms, threshold=threshold)
    frames = vad.score_frames(audio)
    segments = merge_speech_frames(
        frames,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )

    result = {
        "audio": os.path.abspath(audio_path),
        "sample_rate": sr,
        "backend": "energy",
        "params": {
            "threshold": threshold,
            "frame_ms": frame_ms,
            "min_speech_ms": min_speech_ms,
            "min_silence_ms": min_silence_ms,
            "speech_pad_ms": speech_pad_ms,
        },
        "segments": segments,
        "summary": {
            "num_segments": len(segments),
            "speech_ms": int(sum(seg["duration_ms"] for seg in segments)),
            "audio_ms": int(round(len(audio) / sr * 1000)),
        },
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def filter_dataset_items(items: Sequence[dict], where: Sequence[str] | None = None, start: int = 0, n: int = 0) -> List[dict]:
    """Filter dataset rows with simple key=value predicates, then slice."""
    predicates = [_parse_where(expr) for expr in (where or [])]
    selected = []
    for item in items:
        if all(str(item.get(key, "")) == value for key, value in predicates):
            selected.append(item)
    start = max(0, start)
    if n and n > 0:
        return list(selected[start : start + n])
    return list(selected[start:])


def run_vad_dataset(
    dataset_json: str,
    output_json: str,
    where: Sequence[str] | None = None,
    start: int = 0,
    n: int = 0,
    threshold: float = 0.2,
    frame_ms: int = 30,
    min_speech_ms: int = 250,
    min_silence_ms: int = 200,
    speech_pad_ms: int = 80,
    sample_rate: int = 16000,
) -> dict:
    with open(dataset_json, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("dataset JSON must be a list of objects")

    selected = filter_dataset_items(items, where=where, start=start, n=n)
    dataset_dir = os.path.dirname(os.path.abspath(dataset_json))
    results = []
    failures = []

    for item in selected:
        try:
            audio_path = resolve_audio_path(item, dataset_dir)
            vad_result = run_vad_file(
                audio_path,
                output_json=None,
                threshold=threshold,
                frame_ms=frame_ms,
                min_speech_ms=min_speech_ms,
                min_silence_ms=min_silence_ms,
                speech_pad_ms=speech_pad_ms,
                sample_rate=sample_rate,
            )
            results.append(
                {
                    "id": item.get("id"),
                    "audio": item.get("audio"),
                    "audio_path": os.path.abspath(audio_path),
                    "metadata": {
                        key: item.get(key)
                        for key in (
                            "speaker",
                            "emotion",
                            "emotion_code",
                            "sentence_code",
                            "intensity",
                            "speaker_sex",
                            "speaker_age",
                        )
                        if key in item
                    },
                    "segments": vad_result["segments"],
                    "summary": vad_result["summary"],
                }
            )
        except Exception as exc:  # Keep batch jobs inspectable instead of aborting all rows.
            failures.append({"id": item.get("id"), "error": str(exc)})

    batch = {
        "dataset": os.path.abspath(dataset_json),
        "filters": list(where or []),
        "params": {
            "threshold": threshold,
            "frame_ms": frame_ms,
            "min_speech_ms": min_speech_ms,
            "min_silence_ms": min_silence_ms,
            "speech_pad_ms": speech_pad_ms,
            "sample_rate": sample_rate,
        },
        "summary": {
            "num_input_items": len(items),
            "num_selected_items": len(selected),
            "num_processed_items": len(results),
            "num_failed_items": len(failures),
            "total_speech_ms": int(sum(row["summary"]["speech_ms"] for row in results)),
            "total_audio_ms": int(sum(row["summary"]["audio_ms"] for row in results)),
        },
        "items": results,
        "failures": failures,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
    return batch


def resolve_audio_path(item: dict, dataset_dir: str) -> str:
    rel_path = item.get("audio_path") or item.get("audio")
    if not rel_path:
        raise ValueError(f"dataset item {item.get('id')} has no audio/audio_path field")
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.abspath(os.path.join(dataset_dir, rel_path))


def _as_float_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def _json_safe_points(points: Iterable[dict]) -> List[dict]:
    safe = []
    for point in points:
        threshold = point["threshold"]
        if threshold == math.inf:
            threshold_value = "inf"
        elif threshold == -math.inf:
            threshold_value = "-inf"
        else:
            threshold_value = threshold
        safe.append({"threshold": threshold_value, "tpr": point["tpr"], "fpr": point["fpr"]})
    return safe


def _parse_where(expr: str) -> Tuple[str, str]:
    if "=" not in expr:
        raise ValueError(f"Invalid --where expression {expr!r}; expected key=value")
    key, value = expr.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"Invalid --where expression {expr!r}; key is empty")
    return key, value


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 1 VAD segmentation")
    ap.add_argument("--dataset",default=os.path.join(HERE, "..", "data", "dataset.json"),help="Dataset JSON list with audio/audio_path fields")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--where", action="append", default=[], help="Filter dataset rows by key=value; can be repeated")
    ap.add_argument("--start", type=int, default=0, help="Start offset after filtering dataset rows")
    ap.add_argument("--n", type=int, default=0, help="Number of dataset rows to process; 0 means all after start")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--frame_ms", type=int, default=30)
    ap.add_argument("--min_speech_ms", type=int, default=250)
    ap.add_argument("--min_silence_ms", type=int, default=200)
    ap.add_argument("--speech_pad_ms", type=int, default=80)
    ap.add_argument("--sample_rate", type=int, default=16000)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_vad_dataset(
        args.dataset,
        args.out,
        where=args.where,
        start=args.start,
        n=args.n,
        threshold=args.threshold,
        frame_ms=args.frame_ms,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        sample_rate=args.sample_rate,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
