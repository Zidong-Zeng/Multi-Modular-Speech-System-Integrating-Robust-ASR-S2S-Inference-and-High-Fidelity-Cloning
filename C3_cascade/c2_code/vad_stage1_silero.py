# -*- coding: utf-8 -*-
"""Stage 1 Silero VAD backend with Stage 1-compatible dataset output."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Sequence

import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from vad_stage1_energy import filter_dataset_items, read_wav_mono, resolve_audio_path


# Silero segment normalization keeps Stage 2 compatible with the existing Stage 1 schema.
def normalize_silero_segments(
    raw_segments: Sequence[dict],
    sample_rate: int,
    confidences: Sequence[float] | None = None,
) -> list[dict]:
    normalized = []
    confidences = list(confidences or [])
    for index, raw in enumerate(raw_segments):
        start = int(raw.get("start", 0))
        end = int(raw.get("end", 0))
        if end <= start:
            continue
        confidence = raw.get("prob")
        if confidence is None and index < len(confidences):
            confidence = confidences[index]
        confidence = 1.0 if confidence is None else float(confidence)
        start_ms = int(round(start / sample_rate * 1000))
        end_ms = int(round(end / sample_rate * 1000))
        normalized.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": int(max(0, end_ms - start_ms)),
                "confidence": round(float(confidence), 4),
            }
        )
    return normalized


# Device resolution keeps CLI behavior explicit while preserving an auto mode for servers.
def resolve_device(device: str, cuda_available: bool | None = None) -> str:
    device = str(device).lower().strip()
    cuda_available = detect_cuda_available() if cuda_available is None else bool(cuda_available)
    if device == "auto":
        return "cuda" if cuda_available else "cpu"
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not cuda_available:
            raise ValueError("CUDA was requested but is not available")
        return "cuda"
    raise ValueError(f"Unsupported device {device!r}; expected auto, cpu, or cuda")


# Progress logging keeps long-running server jobs observable in the terminal.
def log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[vad_stage1_silero] {message}", file=sys.stderr, flush=True)


# CUDA detection is isolated for testability and to avoid importing torch at module import time.
def detect_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


# Installed-package loading keeps the server path aligned with the user's current setup.
def load_silero_package_api():
    try:
        module = importlib.import_module("silero_vad")
    except Exception as exc:
        raise RuntimeError(
            "silero_vad is not installed in the current Python environment. "
            "Please install the local wheel first, for example: "
            "pip install --no-index --find-links=/root/siton-tmp/assignment_C/model/silero-vad-wheels silero-vad"
        ) from exc
    if not hasattr(module, "load_silero_vad") or not hasattr(module, "get_speech_timestamps"):
        raise RuntimeError(
            "Installed silero_vad package does not expose load_silero_vad/get_speech_timestamps as expected."
        )
    return module


# SileroSegmenter wraps the real model or an injected fake backend behind one stable API.
class SileroSegmenter:
    def __init__(
        self,
        backend=None,
        sample_rate: int = 16000,
        threshold: float = 0.35,
        min_speech_ms: int = 250,
        min_silence_ms: int = 200,
        speech_pad_ms: int = 80,
        device: str = "cpu",
    ):
        self.backend = backend
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.device = device
        if self.backend is not None and hasattr(self.backend, "set_device"):
            self.backend.set_device(device)

    def run_file(self, audio_path: str, output_json: str | None = None) -> dict:
        audio, sr = read_wav_mono(audio_path, target_sample_rate=self.sample_rate)
        result = self.run_array(audio, sample_rate=sr)
        result["audio"] = os.path.abspath(audio_path)
        if output_json:
            os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    def run_array(self, audio: np.ndarray, sample_rate: int | None = None) -> dict:
        sample_rate = sample_rate or self.sample_rate
        raw_segments = self._get_backend().get_speech_timestamps(
            audio,
            sampling_rate=sample_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )
        segments = normalize_silero_segments(raw_segments, sample_rate=sample_rate)
        return {
            "audio": None,
            "sample_rate": sample_rate,
            "backend": "silero",
            "params": {
                "threshold": self.threshold,
                "min_speech_ms": self.min_speech_ms,
                "min_silence_ms": self.min_silence_ms,
                "speech_pad_ms": self.speech_pad_ms,
                "sample_rate": sample_rate,
            },
            "segments": segments,
            "summary": {
                "num_segments": len(segments),
                "speech_ms": int(sum(seg["duration_ms"] for seg in segments)),
                "audio_ms": int(round(len(audio) / sample_rate * 1000)),
            },
        }

    def _get_backend(self):
        if self.backend is None:
            self.backend = load_silero_backend(
                device=self.device,
            )
        elif hasattr(self.backend, "set_device"):
            self.backend.set_device(self.device)
        return self.backend


# The real Silero loader uses the installed package instead of torch.hub repo loading.
def load_silero_backend(device: str = "cpu"):
    import torch

    silero_vad = load_silero_package_api()
    try:
        model = silero_vad.load_silero_vad()
    except Exception as exc:
        message = (
            f"Failed to initialize silero_vad from the installed package: {exc}. "
            "Please verify that silero-vad is installed correctly in the current environment."
        )
        raise RuntimeError(message) from exc
    get_speech_timestamps = silero_vad.get_speech_timestamps
    if hasattr(model, "to"):
        model = model.to(device)

    class _Backend:
        def __init__(self, model_obj, target_device: str):
            self.model = model_obj
            self.device = target_device

        def set_device(self, device_name: str):
            self.device = device_name
            if hasattr(self.model, "to"):
                self.model = self.model.to(device_name)

        def get_speech_timestamps(self, audio, sampling_rate, threshold, min_speech_duration_ms, min_silence_duration_ms, speech_pad_ms):
            audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(self.device)
            return get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=sampling_rate,
                threshold=threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
            )

    return _Backend(model, device)


# Batch dataset processing keeps output structure identical to the EnergyVAD Stage 1 JSON.
def run_vad_dataset_silero(
    dataset_json: str,
    output_json: str,
    where: Sequence[str] | None = None,
    start: int = 0,
    n: int = 0,
    threshold: float = 0.35,
    min_speech_ms: int = 250,
    min_silence_ms: int = 200,
    speech_pad_ms: int = 80,
    sample_rate: int = 16000,
    silero_backend=None,
    device: str = "auto",
    verbose: bool = False,
    log_every: int = 50,
) -> dict:
    with open(dataset_json, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("dataset JSON must be a list of objects")

    selected = filter_dataset_items(items, where=where, start=start, n=n)
    dataset_dir = os.path.dirname(os.path.abspath(dataset_json))
    resolved_device = resolve_device(device)
    segmenter = SileroSegmenter(
        backend=silero_backend,
        sample_rate=sample_rate,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        device=resolved_device,
    )
    results = []
    failures = []
    log_every = max(1, int(log_every))
    log_progress(verbose, f"Using Silero device={resolved_device} for {len(selected)} selected items")
    try:
        segmenter._get_backend()
    except Exception as exc:
        message = str(exc)
        failures = [{"id": item.get("id"), "error": message} for item in selected]
        batch = {
            "dataset": os.path.abspath(dataset_json),
            "backend": "silero",
            "filters": list(where or []),
            "params": {
                "threshold": threshold,
                "min_speech_ms": min_speech_ms,
                "min_silence_ms": min_silence_ms,
                "speech_pad_ms": speech_pad_ms,
                "sample_rate": sample_rate,
                "device": resolved_device,
            },
            "summary": {
                "num_input_items": len(items),
                "num_selected_items": len(selected),
                "num_processed_items": 0,
                "num_failed_items": len(failures),
                "total_speech_ms": 0,
                "total_audio_ms": 0,
            },
            "items": [],
            "failures": failures,
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
        return batch

    for index, item in enumerate(selected, start=1):
        try:
            audio_path = resolve_audio_path(item, dataset_dir)
            vad_result = segmenter.run_file(audio_path, output_json=None)
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
        except Exception as exc:
            failures.append({"id": item.get("id"), "error": str(exc)})
        if verbose and (index == 1 or index % log_every == 0 or index == len(selected)):
            log_progress(
                verbose,
                f"Processed {index}/{len(selected)} items (ok={len(results)}, failed={len(failures)})",
            )

    batch = {
        "dataset": os.path.abspath(dataset_json),
        "backend": "silero",
        "filters": list(where or []),
        "params": {
            "threshold": threshold,
            "min_speech_ms": min_speech_ms,
            "min_silence_ms": min_silence_ms,
            "speech_pad_ms": speech_pad_ms,
            "sample_rate": sample_rate,
            "device": resolved_device,
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


# CLI arguments mirror the existing Stage 1 script so Silero can be swapped in cleanly.
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 1 Silero VAD segmentation")
    ap.add_argument("--dataset", default=os.path.join(HERE, "..", "data", "ava_test_manifest.json"), help="Dataset JSON list with audio/audio_path fields")
    ap.add_argument("--out", default="/root/siton-tmp/assignment_C/C2_ASR/outputs/vad_stage1_silero_ava_test.json", help="Output JSON path")
    ap.add_argument("--where", action="append", default=[], help="Filter dataset rows by key=value; can be repeated")
    ap.add_argument("--start", type=int, default=0, help="Start offset after filtering dataset rows")
    ap.add_argument("--n", type=int, default=0, help="Number of dataset rows to process; 0 means all after start")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--min_speech_ms", type=int, default=250)
    ap.add_argument("--min_silence_ms", type=int, default=200)
    ap.add_argument("--speech_pad_ms", type=int, default=80)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--verbose", action="store_false", help="Print progress logs during batch processing")
    ap.add_argument("--log_every", type=int, default=50, help="Log progress every N items when --verbose is enabled")
    return ap
def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_vad_dataset_silero(
        dataset_json=args.dataset,
        output_json=args.out,
        where=args.where,
        start=args.start,
        n=args.n,
        threshold=args.threshold,
        min_speech_ms=args.min_speech_ms,
        min_silence_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        sample_rate=args.sample_rate,
        device=args.device,
        verbose=args.verbose,
        log_every=args.log_every,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
