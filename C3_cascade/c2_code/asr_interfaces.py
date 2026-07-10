# -*- coding: utf-8 -*-
"""Shared VAD/ASR interfaces used by the staged long-audio pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


# Unified ASR return object so Stage 3 is independent from a specific model.
@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    latency_ms: int = 0


# Replaceable ASR backend contract; Whisper is one implementation, tests use fakes.
class ASRBackend(Protocol):
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        raise NotImplementedError


# Replaceable VAD backend contract reserved for EnergyVAD, Silero, WebRTC, etc.
class VADBackend(Protocol):
    def detect(self, audio_path: str) -> list[dict]:
        raise NotImplementedError


# Whisper ASR backend is lazy-loaded so local tests do not require torch/transformers.
class WhisperASRBackend:
    def __init__(self, model_path: str, batch_size: int = 1, language: str = "english", task: str = "transcribe"):
        self.model_path = model_path
        self.batch_size = batch_size
        self.language = language
        self.task = task
        self._pipeline = None

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        import time

        asr = self._load_pipeline()
        start = time.time()
        result = asr(
            audio_path,
            batch_size=self.batch_size,
            generate_kwargs={"language": self.language, "task": self.task},
        )
        latency_ms = int(round((time.time() - start) * 1000))
        return TranscriptionResult(text=str(result.get("text", "")).strip(), latency_ms=latency_ms)

    def _load_pipeline(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_path,
                torch_dtype=dtype,
                device=device,
            )
        return self._pipeline
