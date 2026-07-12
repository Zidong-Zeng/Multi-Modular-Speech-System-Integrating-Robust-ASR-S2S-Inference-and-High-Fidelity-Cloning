# -*- coding: utf-8 -*-
"""Lightweight prosody feature utilities for C4 emotion/style experiments."""
from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Tuple

import librosa
import numpy as np


EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

FEATURE_NAMES = [
    "duration",
    "pitch_mean",
    "pitch_std",
    "pitch_min",
    "pitch_max",
    "pitch_range",
    "rms_mean",
    "rms_std",
    "rms_max",
    "pause_ratio",
    "voiced_ratio",
    "zcr_mean",
    "zcr_std",
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "bandwidth_std",
    "onset_rate",
] + [f"mfcc{i}_mean" for i in range(1, 14)] + [f"mfcc{i}_std" for i in range(1, 14)]


STYLE_PROMPTS = {
    "angry": "Preserve a strong, firm, high-intensity speaking tone.",
    "disgust": "Preserve a low, displeased, slightly tense speaking tone.",
    "fear": "Preserve a tense, cautious, and slightly unstable speaking tone.",
    "happy": "Preserve a bright, energetic, and upbeat speaking tone.",
    "neutral": "Preserve a natural and calm speaking tone.",
    "sad": "Preserve a low-energy, slower, and subdued speaking tone.",
}


def resolve_audio_path(data_root: str, sample: dict) -> str:
    audio = sample["audio"]
    return audio if os.path.isabs(audio) else os.path.join(data_root, audio)


def safe_float(value: float) -> float:
    if value is None:
        return 0.0
    try:
        value = float(value)
    except Exception:
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def _stats(values: np.ndarray) -> Tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        safe_float(np.mean(values)),
        safe_float(np.std(values)),
        safe_float(np.min(values)),
        safe_float(np.max(values)),
    )


def extract_prosody_features(audio_file: str, target_sr: int = 16000) -> Dict[str, float]:
    y, sr = librosa.load(audio_file, sr=target_sr, mono=True)
    if y.size == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    y = librosa.util.normalize(y)
    duration = librosa.get_duration(y=y, sr=sr)
    hop_length = 512
    frame_length = 1024

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean, rms_std, _, rms_max = _stats(rms)
    if rms.size and rms_max > 0:
        voiced_frames = rms > max(0.01, 0.15 * rms_max)
        pause_ratio = 1.0 - float(np.mean(voiced_frames))
        voiced_ratio = float(np.mean(voiced_frames))
    else:
        pause_ratio = 1.0
        voiced_ratio = 0.0

    try:
        pitch = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=frame_length, hop_length=hop_length)
    except Exception:
        pitch = np.asarray([], dtype=np.float64)
    pitch = pitch[np.isfinite(pitch)]
    pitch = pitch[(pitch >= 50) & (pitch <= 500)]
    pitch_mean, pitch_std, pitch_min, pitch_max = _stats(pitch)

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr_mean, zcr_std, _, _ = _stats(zcr)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    centroid_mean, centroid_std, _, _ = _stats(centroid)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    bandwidth_mean, bandwidth_std, _, _ = _stats(bandwidth)

    try:
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
        onset_rate = len(onset_frames) / max(duration, 1e-6)
    except Exception:
        onset_rate = 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
    mfcc_means = np.mean(mfcc, axis=1) if mfcc.size else np.zeros(13)
    mfcc_stds = np.std(mfcc, axis=1) if mfcc.size else np.zeros(13)

    features = {
        "duration": duration,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_range": pitch_max - pitch_min if pitch_max > pitch_min else 0.0,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_max": rms_max,
        "pause_ratio": pause_ratio,
        "voiced_ratio": voiced_ratio,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "centroid_mean": centroid_mean,
        "centroid_std": centroid_std,
        "bandwidth_mean": bandwidth_mean,
        "bandwidth_std": bandwidth_std,
        "onset_rate": onset_rate,
    }
    for idx, value in enumerate(mfcc_means, start=1):
        features[f"mfcc{idx}_mean"] = safe_float(value)
    for idx, value in enumerate(mfcc_stds, start=1):
        features[f"mfcc{idx}_std"] = safe_float(value)

    return {name: safe_float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def vectorize_features(features: Dict[str, float], feature_names: Iterable[str] = FEATURE_NAMES) -> List[float]:
    return [safe_float(features.get(name, 0.0)) for name in feature_names]


def style_prompt_for_emotion(emotion: str) -> str:
    return STYLE_PROMPTS.get((emotion or "").lower(), STYLE_PROMPTS["neutral"])

