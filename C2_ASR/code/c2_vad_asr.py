# -*- coding: utf-8 -*-
"""
C2 ASR entrypoint with explicit VAD -> chunk -> ASR flow.

Supported engines:
1. transformers pipeline
2. WhisperX with alignment and optional diarization
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from typing import Any, Callable


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _maybe_offline() -> None:
    argv = sys.argv
    offline = "--offline" in argv
    for flag in ("--model", "--asr_model", "--llm"):
        if flag in argv:
            index = argv.index(flag) + 1
            if index < len(argv) and os.path.isdir(argv[index]):
                offline = True
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


_maybe_offline()

HERE = os.path.dirname(os.path.abspath(__file__))


def normalize(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(left: list[str], right: list[str]) -> int:
    prev = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        curr = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    ref_words = [word for text in refs for word in normalize(text).split()]
    hyp_words = [word for text in hyps for word in normalize(text).split()]
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def compute_cer(refs: list[str], hyps: list[str]) -> float:
    ref_chars = list("".join(normalize(text).replace(" ", "") for text in refs))
    hyp_chars = list("".join(normalize(text).replace(" ", "") for text in hyps))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def load_torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("torch is required for ASR inference but is not installed.") from exc


def load_transformers_pipeline():
    try:
        module = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Install transformers before using the transformers ASR engine."
        ) from exc
    return module.pipeline


def load_whisperx_module(importer: Callable[[], Any] | None = None):
    importer = importer or (lambda: importlib.import_module("whisperx"))
    try:
        return importer()
    except ImportError as exc:
        raise RuntimeError(
            "whisperx is not installed. Install whisperx and its alignment dependencies before using this engine."
        ) from exc


def load_silero_segmenter():
    module = importlib.import_module("vad_stage1_silero")
    return module.SileroSegmenter


def load_chunking_helpers():
    module = importlib.import_module("vad_stage2")
    return module.build_dynamic_chunks, module.export_chunk_audio


def resolve_hf_token(cli_token: str | None, diarize: bool) -> str | None:
    token = cli_token or os.environ.get("HF_TOKEN")
    if diarize and not token:
        raise ValueError("HF token is required when diarization is enabled. Use --hf_token or HF_TOKEN.")
    return token


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/root/siton-tmp/assignment_C/C2_ASR/data/libris_manifest.json")
    ap.add_argument("--model", default="/root/siton-tmp/assignment_C/model/whisper-large-v3")
    ap.add_argument("--engine", default="transformers", choices=["transformers", "whisperx"])
    ap.add_argument("--start", type=int, default=0, help="璧峰涓嬫爣(榛樿0)")
    ap.add_argument("--n", type=int, default=0, help="鍙窇鍓嶅嚑鏉★紱榛樿 0 = 璺戞暟鎹泦鍏ㄩ儴")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--language", default=None, help="璇█浠ｇ爜锛涗笉浼犲垯鐢辨ā鍨嬭嚜鍔ㄥ鐞?)
    ap.add_argument("--compute_type", default="float16", help="WhisperX/faster-whisper compute type")
    ap.add_argument("--diarize", action="store_true", help="鍚敤 PyAnnote 璇磋瘽浜鸿仛绫?)
    ap.add_argument("--hf_token", default=None, help="PyAnnote/WhisperX 璇磋瘽浜鸿仛绫绘墍闇€ Hugging Face token")
    ap.add_argument("--vad_threshold", type=float, default=0.35)
    ap.add_argument("--vad_min_speech_ms", type=int, default=250)
    ap.add_argument("--vad_min_silence_ms", type=int, default=200)
    ap.add_argument("--vad_speech_pad_ms", type=int, default=80)
    ap.add_argument("--chunk_max_ms", type=int, default=30000)
    ap.add_argument("--chunk_min_ms", type=int, default=1000)
    ap.add_argument("--chunk_merge_gap_ms", type=int, default=500)
    ap.add_argument("--chunk_overlap_ms", type=int, default=500)
    ap.add_argument("--offline", action="store_false", help="绂荤嚎妯″紡锛氫笉鑱旂綉锛岀敤鏈湴/缂撳瓨宸蹭笅杞界殑妯″瀷")
    ap.add_argument("--outdir", default="/root/siton-tmp/assignment_C/C2_ASR/outputs/c2_vad_asr/libris-v3")
    return ap


def load_dataset_rows(dataset_path: str, start: int = 0, n: int = 0) -> tuple[list[dict], str]:
    with open(dataset_path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("dataset JSON must be a list of objects")
    data_root = os.path.dirname(os.path.abspath(dataset_path))
    selected = rows[start:] if n <= 0 else rows[start : start + n]
    return selected, data_root


def resolve_audio_paths(samples: list[dict], data_root: str) -> list[str]:
    return [os.path.join(data_root, sample["audio"]) for sample in samples]


def resolve_generation_kwargs(language: str | None) -> dict[str, str]:
    kwargs = {"task": "transcribe"}
    if language:
        kwargs["language"] = language
    return kwargs


def build_transformers_prediction_row(sample: dict, hypothesis: str) -> dict:
    return {
        "id": sample["id"],
        "audio": sample["audio"],
        "reference": sample.get("text", ""),
        "hypothesis": hypothesis,
        "emotion": sample.get("emotion"),
        "engine": "transformers",
    }


def build_chunked_prediction_row(sample: dict, chunks: list[dict], hypothesis: str, engine: str) -> dict:
    return {
        "id": sample["id"],
        "audio": sample["audio"],
        "reference": sample.get("text", ""),
        "hypothesis": hypothesis,
        "emotion": sample.get("emotion"),
        "engine": engine,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }


def build_whisperx_prediction_row(sample: dict, result: dict, stage_times: dict[str, float]) -> dict:
    segments = result.get("segments", [])
    texts = [str(segment.get("text", "")).strip() for segment in segments if str(segment.get("text", "")).strip()]
    words = [word for segment in segments for word in segment.get("words", [])]
    speakers = sorted({str(segment.get("speaker")) for segment in segments if segment.get("speaker")})
    return {
        "id": sample["id"],
        "audio": sample["audio"],
        "reference": sample.get("text", ""),
        "hypothesis": " ".join(texts).strip(),
        "emotion": sample.get("emotion"),
        "engine": "whisperx",
        "language": result.get("language"),
        "segments": segments,
        "words": words,
        "speakers": speakers,
        "stage_times": stage_times,
    }


def build_chunks_from_segments(
    segments: list[dict],
    audio_ms: int,
    max_chunk_ms: int,
    min_chunk_ms: int,
    merge_gap_ms: int,
    overlap_ms: int,
    chunk_builder: Callable[..., list[dict]] | None = None,
) -> list[dict]:
    if chunk_builder is None:
        chunk_builder, _ = load_chunking_helpers()
    return chunk_builder(
        segments,
        audio_ms=audio_ms,
        max_chunk_ms=max_chunk_ms,
        min_chunk_ms=min_chunk_ms,
        merge_gap_ms=merge_gap_ms,
        overlap_ms=overlap_ms,
    )


def cap_chunk_durations(chunks: list[dict], max_duration_ms: int) -> list[dict]:
    capped = []
    for chunk in chunks:
        row = dict(chunk)
        duration_ms = int(row["end_ms"]) - int(row["start_ms"])
        if duration_ms > max_duration_ms:
            row["end_ms"] = int(row["start_ms"]) + max_duration_ms
            row["duration_ms"] = max_duration_ms
        else:
            row["duration_ms"] = duration_ms
        capped.append(row)
    return capped


def build_vad_chunks_for_audio(
    audio_path: str,
    args,
    segmenter_factory: Callable[[], Any] | None = None,
    chunk_builder: Callable[..., list[dict]] | None = None,
) -> tuple[list[dict], int]:
    segmenter_factory = segmenter_factory or (
        lambda: load_silero_segmenter()(
            sample_rate=16000,
            threshold=args.vad_threshold,
            min_speech_ms=args.vad_min_speech_ms,
            min_silence_ms=args.vad_min_silence_ms,
            speech_pad_ms=args.vad_speech_pad_ms,
            device="cpu",
        )
    )
    segmenter = segmenter_factory()
    vad_result = segmenter.run_file(audio_path, output_json=None)
    audio_ms = int(vad_result.get("summary", {}).get("audio_ms", 0))
    chunks = build_chunks_from_segments(
        segments=vad_result.get("segments", []),
        audio_ms=audio_ms,
        max_chunk_ms=args.chunk_max_ms,
        min_chunk_ms=args.chunk_min_ms,
        merge_gap_ms=args.chunk_merge_gap_ms,
        overlap_ms=args.chunk_overlap_ms,
        chunk_builder=chunk_builder,
    )
    # Keep exported chunk audio below Whisper's 30s long-form threshold.
    chunks = cap_chunk_durations(chunks, max_duration_ms=min(args.chunk_max_ms - 500, 29500))
    return chunks, audio_ms


def run_transformers_engine(
    samples: list[dict],
    audio_paths: list[str],
    args,
    torch_module=None,
    pipeline_factory=None,
    segmenter_factory: Callable[[], Any] | None = None,
    chunk_exporter: Callable[[str, dict, str, str], str] | None = None,
    chunk_builder: Callable[..., list[dict]] | None = None,
) -> tuple[list[dict], float]:
    torch = torch_module or load_torch()
    pipeline = pipeline_factory or load_transformers_pipeline()
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"鍔犺浇妯″瀷: {args.model}  (engine=transformers, device={'cuda:0' if device == 0 else 'cpu'}, dtype={dtype})")
    t0 = time.time()
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        torch_dtype=dtype,
        device=device,
    )
    print(f"妯″瀷鍔犺浇鑰楁椂: {time.time() - t0:.1f}s")
    if device == 0 and hasattr(torch.cuda, "get_device_name") and hasattr(torch.cuda, "memory_allocated"):
        print(f"GPU: {torch.cuda.get_device_name(0)}  鏄惧瓨鍗犵敤: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"\n寮€濮嬫壒閲忚瘑鍒?{len(audio_paths)} 鏉￠煶棰?(batch_size={args.batch_size}) ...")
    started = time.time()
    predictions = []
    if chunk_exporter is None:
        _, chunk_exporter = load_chunking_helpers()

    for sample, audio_path in zip(samples, audio_paths):
        chunks, _audio_ms = build_vad_chunks_for_audio(
            audio_path,
            args,
            segmenter_factory=segmenter_factory,
            chunk_builder=chunk_builder,
        )
        chunk_dir = os.path.join(args.outdir, "stage2_chunks")
        chunk_paths = [
            chunk_exporter(audio_path, chunk, chunk_dir, str(sample.get("id") or "item"))
            for chunk in chunks
        ]
        for chunk, chunk_audio in zip(chunks, chunk_paths):
            result = asr(
                chunk_audio,
                batch_size=1,
                generate_kwargs=resolve_generation_kwargs(args.language),
            )
            chunk["text"] = str(result["text"]).strip()
            chunk["chunk_audio"] = chunk_audio
        hypothesis = " ".join(chunk.get("text", "") for chunk in chunks).strip()
        predictions.append(build_chunked_prediction_row(sample, chunks, hypothesis, engine="transformers"))

    infer_time = time.time() - started
    print(f"鎵归噺鎺ㄧ悊瀹屾垚锛岃€楁椂 {infer_time:.1f}s锛屽钩鍧?{infer_time / len(audio_paths) * 1000:.0f} ms/鏉?)
    return predictions, infer_time


def run_whisperx_engine(
    samples: list[dict],
    audio_paths: list[str],
    args,
    torch_module=None,
    whisperx_module=None,
    segmenter_factory: Callable[[], Any] | None = None,
    chunk_exporter: Callable[[str, dict, str, str], str] | None = None,
    chunk_builder: Callable[..., list[dict]] | None = None,
) -> tuple[list[dict], float]:
    torch = torch_module or load_torch()
    whisperx = whisperx_module or load_whisperx_module()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    token = resolve_hf_token(args.hf_token, args.diarize)
    language = args.language or None

    print(f"鍔犺浇妯″瀷: {args.model}  (engine=whisperx, device={device}, compute_type={args.compute_type})")
    t0 = time.time()
    model = whisperx.load_model(args.model, device, compute_type=args.compute_type, language=language)
    print(f"妯″瀷鍔犺浇鑰楁椂: {time.time() - t0:.1f}s")

    diarizer = None
    if args.diarize:
        try:
            diarizer = whisperx.DiarizationPipeline(use_auth_token=token, device=device)
        except TypeError:
            diarizer = whisperx.DiarizationPipeline(token=token, device=device)

    align_cache: dict[str, tuple[Any, Any]] = {}
    predictions = []
    started = time.time()
    if chunk_exporter is None:
        _, chunk_exporter = load_chunking_helpers()

    for sample, audio_path in zip(samples, audio_paths):
        stage_times: dict[str, float] = {}
        chunks, _audio_ms = build_vad_chunks_for_audio(
            audio_path,
            args,
            segmenter_factory=segmenter_factory,
            chunk_builder=chunk_builder,
        )
        chunk_dir = os.path.join(args.outdir, "stage2_chunks")
        aligned_segments = []

        for chunk in chunks:
            chunk_audio = chunk_exporter(audio_path, chunk, chunk_dir, str(sample.get("id") or "item"))
            audio = whisperx.load_audio(chunk_audio)

            t_stage = time.time()
            result = model.transcribe(audio, batch_size=args.batch_size)
            stage_times["transcribe"] = stage_times.get("transcribe", 0.0) + round(time.time() - t_stage, 3)

            lang = result.get("language") or language or "en"
            if lang not in align_cache:
                align_cache[lang] = whisperx.load_align_model(language_code=lang, device=device)
            model_a, metadata = align_cache[lang]

            t_stage = time.time()
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            stage_times["align"] = stage_times.get("align", 0.0) + round(time.time() - t_stage, 3)

            offset_s = chunk["start_ms"] / 1000.0
            chunk_texts = []
            for segment in result.get("segments", []):
                shifted = dict(segment)
                shifted["start"] = float(shifted.get("start", 0.0)) + offset_s
                shifted["end"] = float(shifted.get("end", 0.0)) + offset_s
                shifted_words = []
                for word in shifted.get("words", []):
                    shifted_word = dict(word)
                    if "start" in shifted_word:
                        shifted_word["start"] = float(shifted_word["start"]) + offset_s
                    if "end" in shifted_word:
                        shifted_word["end"] = float(shifted_word["end"]) + offset_s
                    shifted_words.append(shifted_word)
                shifted["words"] = shifted_words
                aligned_segments.append(shifted)
                text = str(shifted.get("text", "")).strip()
                if text:
                    chunk_texts.append(text)
            chunk["text"] = " ".join(chunk_texts).strip()
            chunk["chunk_audio"] = chunk_audio

        final_result = {"language": language or "en", "segments": aligned_segments}

        if diarizer is not None:
            diar_audio = whisperx.load_audio(audio_path)
            t_stage = time.time()
            diarize_segments = diarizer(diar_audio)
            stage_times["diarize"] = round(time.time() - t_stage, 3)

            t_stage = time.time()
            final_result = whisperx.assign_word_speakers(diarize_segments, final_result)
            stage_times["assign_speakers"] = round(time.time() - t_stage, 3)

        row = build_whisperx_prediction_row(sample, final_result, stage_times)
        row["num_chunks"] = len(chunks)
        row["chunks"] = chunks
        predictions.append(row)

    infer_time = time.time() - started
    print(f"WhisperX 鎺ㄧ悊瀹屾垚锛岃€楁椂 {infer_time:.1f}s锛屽钩鍧?{infer_time / len(audio_paths) * 1000:.0f} ms/鏉?)
    return predictions, infer_time


def summarize_predictions(predictions: list[dict], references: list[str], infer_time: float, model: str, engine: str) -> dict:
    hypotheses = [prediction.get("hypothesis", "") for prediction in predictions]
    wer = compute_wer(references, hypotheses)
    cer = compute_cer(references, hypotheses)
    return {
        "engine": engine,
        "model": model,
        "num_samples": len(predictions),
        "WER": round(wer, 4),
        "CER": round(cer, 4),
        "infer_time_sec": round(infer_time, 1),
        "ms_per_sample": round(infer_time / len(predictions) * 1000, 1) if predictions else 0.0,
    }


def save_outputs(predictions: list[dict], summary: dict, outdir: str) -> tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)
    out_json = os.path.join(outdir, "asr_predictions.json")
    summary_json = os.path.join(outdir, "c2_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out_json, summary_json


def print_examples(samples: list[dict], predictions: list[dict], limit: int = 5) -> None:
    print("\n---- 璇嗗埆鏍蜂緥 ----")
    for sample, prediction in list(zip(samples, predictions))[:limit]:
        print(f"[{sample['id']}]")
        print(f"  鍙傝€? {sample.get('text', '')}")
        print(f"  璇嗗埆: {prediction.get('hypothesis', '')}")


def run(args) -> dict:
    samples, data_root = load_dataset_rows(args.dataset, start=args.start, n=args.n)
    audio_paths = resolve_audio_paths(samples, data_root)
    references = [sample.get("text", "") for sample in samples]

    print(f"鏁版嵁闆? {os.path.abspath(args.dataset)}")
    print(f"鍏?{len(samples)} 鏉″緟澶勭悊鏍锋湰")

    if args.engine == "transformers":
        predictions, infer_time = run_transformers_engine(samples, audio_paths, args)
    else:
        predictions, infer_time = run_whisperx_engine(samples, audio_paths, args)

    summary = summarize_predictions(predictions, references, infer_time, args.model, args.engine)
    print(f"\n==== 璇勬祴缁撴灉锛坽len(samples)} 鏉★級====")
    print(f"WER (璇嶉敊璇巼): {summary['WER'] * 100:.2f}%")
    print(f"CER (瀛楃閿欒鐜?: {summary['CER'] * 100:.2f}%")
    print_examples(samples, predictions)
    out_json, summary_json = save_outputs(predictions, summary, args.outdir)
    print(f"\n宸蹭繚瀛? {out_json} (渚汣3浣跨敤) 鍜?{summary_json}")
    return summary


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
