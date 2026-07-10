# -*- coding: utf-8 -*-
"""Stage 2: build VAD-driven dynamic ASR chunks.

The core chunking functions are dependency-light and testable. Whisper is
loaded only when --run_asr is requested, so local tests do not need a model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import wave
from typing import Callable, Sequence


def build_dynamic_chunks(
    segments: Sequence[dict],
    audio_ms: int,
    max_chunk_ms: int = 30000,
    min_chunk_ms: int = 1000,
    merge_gap_ms: int = 500,
    overlap_ms: int = 500,
) -> list[dict]:
    """Merge VAD segments and split them into ASR-friendly windows."""
    if max_chunk_ms <= 0:
        raise ValueError("max_chunk_ms must be positive")
    if min_chunk_ms < 0 or merge_gap_ms < 0 or overlap_ms < 0:
        raise ValueError("chunking parameters must be non-negative")

    merged = merge_vad_segments(segments, merge_gap_ms=merge_gap_ms, min_chunk_ms=min_chunk_ms)
    chunks = []
    for group_index, segment in enumerate(merged):
        source_start = int(segment["start_ms"])
        source_end = int(segment["end_ms"])
        split_index = 0
        cursor = source_start
        while cursor < source_end:
            split_end = min(source_end, cursor + max_chunk_ms)
            if split_end - cursor >= min_chunk_ms:
                chunks.append(
                    make_chunk(
                        chunk_id=len(chunks),
                        group_index=group_index,
                        split_index=split_index,
                        source_start_ms=cursor,
                        source_end_ms=split_end,
                        audio_ms=audio_ms,
                        overlap_ms=overlap_ms,
                        confidence=segment["confidence"],
                        num_source_segments=segment["num_source_segments"],
                    )
                )
            cursor = split_end
            split_index += 1
    return chunks


def merge_vad_segments(segments: Sequence[dict], merge_gap_ms: int, min_chunk_ms: int) -> list[dict]:
    valid = sorted(
        [
            {
                "start_ms": int(seg["start_ms"]),
                "end_ms": int(seg["end_ms"]),
                "confidence": float(seg.get("confidence", 0.0)),
                "num_source_segments": 1,
            }
            for seg in segments
            if int(seg.get("end_ms", 0)) > int(seg.get("start_ms", 0))
        ],
        key=lambda seg: seg["start_ms"],
    )
    merged = []
    for seg in valid:
        if merged and seg["start_ms"] - merged[-1]["end_ms"] <= merge_gap_ms:
            prev = merged[-1]
            prev_count = prev["num_source_segments"]
            new_count = prev_count + 1
            prev["end_ms"] = max(prev["end_ms"], seg["end_ms"])
            prev["confidence"] = (prev["confidence"] * prev_count + seg["confidence"]) / new_count
            prev["num_source_segments"] = new_count
        else:
            merged.append(dict(seg))
    return [seg for seg in merged if seg["end_ms"] - seg["start_ms"] >= min_chunk_ms]


def make_chunk(
    chunk_id: int,
    group_index: int,
    split_index: int,
    source_start_ms: int,
    source_end_ms: int,
    audio_ms: int,
    overlap_ms: int,
    confidence: float,
    num_source_segments: int,
) -> dict:
    start_ms = max(0, source_start_ms - overlap_ms)
    end_ms = min(audio_ms, source_end_ms + overlap_ms)
    return {
        "chunk_id": chunk_id,
        "group_index": group_index,
        "split_index": split_index,
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "duration_ms": int(max(0, end_ms - start_ms)),
        "source_start_ms": int(source_start_ms),
        "source_end_ms": int(source_end_ms),
        "confidence": round(float(confidence), 4),
        "num_source_segments": int(num_source_segments),
    }


def run_stage2(
    vad_json: str,
    output_json: str,
    chunk_dir: str | None = None,
    max_chunk_ms: int = 30000,
    min_chunk_ms: int = 1000,
    merge_gap_ms: int = 500,
    overlap_ms: int = 500,
    export_audio: bool = False,
    transcriber: Callable[[str], str] | None = None,
) -> dict:
    with open(vad_json, encoding="utf-8") as f:
        vad_data = json.load(f)
    items = vad_data.get("items") if isinstance(vad_data, dict) else None
    if not isinstance(items, list):
        raise ValueError("Stage 2 expects a Stage 1 dataset JSON with an items list")

    output_dir = os.path.dirname(os.path.abspath(output_json))
    chunk_dir = chunk_dir or os.path.join(output_dir, "stage2_chunks")
    results = []
    failures = []

    for item in items:
        try:
            audio_path = item.get("audio_path") or item.get("audio")
            if not audio_path:
                raise ValueError(f"item {item.get('id')} has no audio path")
            audio_path = os.path.abspath(audio_path)
            audio_ms = int(item.get("summary", {}).get("audio_ms", 0) or get_wav_duration_ms(audio_path))
            chunks = build_dynamic_chunks(
                item.get("segments", []),
                audio_ms=audio_ms,
                max_chunk_ms=max_chunk_ms,
                min_chunk_ms=min_chunk_ms,
                merge_gap_ms=merge_gap_ms,
                overlap_ms=overlap_ms,
            )
            enrich_chunks_with_optional_asr(
                chunks=chunks,
                item_id=str(item.get("id") or "item"),
                audio_path=audio_path,
                chunk_dir=chunk_dir,
                export_audio=export_audio or transcriber is not None,
                transcriber=transcriber,
            )
            results.append(
                {
                    "id": item.get("id"),
                    "audio": item.get("audio"),
                    "audio_path": audio_path,
                    "summary": {
                        "audio_ms": audio_ms,
                        "num_vad_segments": len(item.get("segments", [])),
                        "num_chunks": len(chunks),
                        "chunk_audio_ms": int(sum(chunk["duration_ms"] for chunk in chunks)),
                    },
                    "chunks": chunks,
                }
            )
        except Exception as exc:
            failures.append({"id": item.get("id"), "error": str(exc)})

    result = {
        "stage": "stage2_dynamic_chunk_asr",
        "vad_json": os.path.abspath(vad_json),
        "params": {
            "max_chunk_ms": max_chunk_ms,
            "min_chunk_ms": min_chunk_ms,
            "merge_gap_ms": merge_gap_ms,
            "overlap_ms": overlap_ms,
            "export_audio": export_audio,
            "run_asr": transcriber is not None,
        },
        "summary": {
            "num_items": len(items),
            "num_processed_items": len(results),
            "num_failed_items": len(failures),
            "num_chunks": int(sum(row["summary"]["num_chunks"] for row in results)),
            "total_audio_ms": int(sum(row["summary"]["audio_ms"] for row in results)),
            "total_chunk_audio_ms": int(sum(row["summary"]["chunk_audio_ms"] for row in results)),
        },
        "items": results,
        "failures": failures,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def enrich_chunks_with_optional_asr(
    chunks: list[dict],
    item_id: str,
    audio_path: str,
    chunk_dir: str,
    export_audio: bool,
    transcriber: Callable[[str], str] | None,
) -> None:
    for chunk in chunks:
        chunk_audio = None
        if export_audio:
            chunk_audio = export_chunk_audio(audio_path, chunk, chunk_dir, item_id)
            chunk["chunk_audio"] = chunk_audio
        if transcriber is not None:
            start = time.time()
            chunk["text"] = transcriber(chunk_audio or audio_path).strip()
            chunk["latency_ms"] = int(round((time.time() - start) * 1000))


def export_chunk_audio(audio_path: str, chunk: dict, chunk_dir: str, item_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", item_id)
    os.makedirs(chunk_dir, exist_ok=True)
    out_path = os.path.join(chunk_dir, f"{safe_id}_chunk{chunk['chunk_id']:04d}.wav")
    with wave.open(audio_path, "rb") as src:
        sample_rate = src.getframerate()
        channels = src.getnchannels()
        sample_width = src.getsampwidth()
        start_frame = int(round(chunk["start_ms"] * sample_rate / 1000))
        end_frame = int(round(chunk["end_ms"] * sample_rate / 1000))
        src.setpos(max(0, start_frame))
        frames = src.readframes(max(0, end_frame - start_frame))
    with wave.open(out_path, "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(sample_width)
        dst.setframerate(sample_rate)
        dst.writeframes(frames)
    return os.path.abspath(out_path)


def get_wav_duration_ms(audio_path: str) -> int:
    with wave.open(audio_path, "rb") as wf:
        return int(round(wf.getnframes() / wf.getframerate() * 1000))


def build_whisper_transcriber(model_path: str, batch_size: int = 1) -> Callable[[str], str]:
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr = pipeline("automatic-speech-recognition", model=model_path, torch_dtype=dtype, device=device)

    def transcribe(audio_path: str) -> str:
        result = asr(audio_path, batch_size=batch_size, generate_kwargs={"language": "english", "task": "transcribe"})
        return result.get("text", "")

    return transcribe


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stage 2 VAD dynamic chunking and optional chunk ASR")
    ap.add_argument("--vad_json", required=True, help="Stage 1 dataset output JSON")
    ap.add_argument("--out", required=True, help="Output long_asr_chunks JSON")
    ap.add_argument("--chunk_dir", default=None, help="Directory for exported chunk WAV files")
    ap.add_argument("--max_chunk_s", type=float, default=30.0)
    ap.add_argument("--min_chunk_s", type=float, default=1.0)
    ap.add_argument("--merge_gap_ms", type=int, default=500)
    ap.add_argument("--overlap_s", type=float, default=0.5)
    ap.add_argument("--export_audio", action="store_true")
    ap.add_argument("--run_asr", action="store_true", help="Run Whisper on exported chunks")
    ap.add_argument("--model", default="/root/siton-tmp/assignment_C/model/whisper-large-v3")
    ap.add_argument("--batch_size", type=int, default=1)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    transcriber = build_whisper_transcriber(args.model, batch_size=args.batch_size) if args.run_asr else None
    result = run_stage2(
        vad_json=args.vad_json,
        output_json=args.out,
        chunk_dir=args.chunk_dir,
        max_chunk_ms=int(round(args.max_chunk_s * 1000)),
        min_chunk_ms=int(round(args.min_chunk_s * 1000)),
        merge_gap_ms=args.merge_gap_ms,
        overlap_ms=int(round(args.overlap_s * 1000)),
        export_audio=args.export_audio,
        transcriber=transcriber,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
