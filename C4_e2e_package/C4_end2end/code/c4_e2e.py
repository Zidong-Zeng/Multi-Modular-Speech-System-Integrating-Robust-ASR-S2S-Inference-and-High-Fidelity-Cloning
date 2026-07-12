# -*- coding: utf-8 -*-
"""
C4: end-to-end speech-to-text translation.

链 接 操 作
1. 环境：
   conda activate speech_tcx

2. 标准全量数据集输入（推荐给总系统主流程）：
   python C4_end2end/code/c4_e2e.py \
     --dataset /root/siton-tmp/assignment_C/common_data/dataset.json \
     --model /root/siton-tmp/assignment_C/model/seamless-m4t-v2-large \
     --outdir /root/siton-tmp/assignment_C/C4_end2end/outputs_core \
     --batch_size 50

3. 和 C1 模块衔接：
   C1 预处理短音频目录：
     /root/siton-tmp/assignment_C/C1_audio_processing/sample_data
   C1 标准 16k 音频：
     /root/siton-tmp/assignment_C/C1_audio_processing/sample_data/augmented/clean
   C1 8k / 44.1k / speed / volume / noise 输入也可以直接用 --input_dir 指向对应目录。
   如果 C1 的 sample_manifest.json 已经写好，可用：
     python C4_end2end/code/c4_e2e.py --dataset C1_audio_processing/sample_data/sample_manifest.json
   如果 manifest 为空或暂未生成，可直接扫描目录：
     python C4_end2end/code/c4_e2e.py \
       --input_dir C1_audio_processing/sample_data/augmented/clean \
       --outdir C4_end2end/outputs_c1_clean

4. 输出给 C5 模块：
   本脚本输出：
     <outdir>/c4_results.json
     <outdir>/c4_summary.json
   C5 读取 c4_results.json 中每条样本的：
     id, audio, e2e_translation_zh, emotion, speaker
   其中 e2e_translation_zh 是 C4 中文翻译文本，建议 C5 用 id 命名输出语音。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_offline(model_path: str, offline: bool) -> None:
    if offline or Path(model_path).expanduser().exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def first_present(mapping: dict, keys: list[str], default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return default


def normalize_manifest_item(item: dict, index: int) -> dict:
    audio = first_present(
        item,
        ["audio", "audio_path", "path", "wav", "file", "file_path", "processed_audio"],
    )
    if not audio:
        raise ValueError(f"sample {index} has no audio/audio_path/path field: {item}")

    sample_id = first_present(item, ["id", "uid", "sample_id"], Path(audio).stem)
    return {
        "id": sample_id,
        "audio": audio,
        "reference_en": first_present(item, ["text", "reference_en", "reference", "transcript"]),
        "emotion": first_present(item, ["emotion", "label", "emotion_label"]),
        "speaker": first_present(item, ["speaker", "speaker_id"]),
    }


def infer_metadata_from_cremad_name(path: Path) -> tuple[str, str]:
    emotion_map = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    }
    parts = path.stem.split("_")
    speaker = parts[0] if parts and parts[0].isdigit() else ""
    emotion = emotion_map.get(parts[2], "") if len(parts) >= 3 else ""
    return speaker, emotion


def extract_sample_list(obj: Any) -> list[dict]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["samples", "items", "data", "audios", "files"]:
            value = obj.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("dataset/manifest must be a list or a dict containing samples/items/data")


def load_samples(dataset_path: Path | None, input_dir: Path | None) -> tuple[list[dict], Path]:
    if input_dir is not None:
        scan_dir = input_dir if input_dir.is_absolute() else PROJECT_ROOT / input_dir
        audio_files = sorted(
            path for path in scan_dir.rglob("*") if path.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
        )
        samples = []
        for path in audio_files:
            speaker, emotion = infer_metadata_from_cremad_name(path)
            try:
                audio_value = str(path.relative_to(PROJECT_ROOT))
            except ValueError:
                audio_value = str(path)
            samples.append(
                {
                    "id": path.stem,
                    "audio": audio_value,
                    "reference_en": "",
                    "emotion": emotion,
                    "speaker": speaker,
                }
            )
        return samples, PROJECT_ROOT

    if dataset_path is None:
        raise ValueError("Either --dataset or --input_dir is required")

    obj = load_json(dataset_path)
    raw_samples = extract_sample_list(obj) if obj else []
    samples = [normalize_manifest_item(item, idx) for idx, item in enumerate(raw_samples)]
    return samples, dataset_path.parent


def resolve_audio_path(audio: str, audio_root: Path) -> Path:
    raw = Path(audio).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                audio_root / raw,
                PROJECT_ROOT / raw,
                PROJECT_ROOT / "common_data" / raw,
                PROJECT_ROOT / "C1_audio_processing" / "sample_data" / raw,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Cannot find audio file: {audio}; tried: {[str(p) for p in candidates]}")


def infer_chunk(samples: list[dict], audio_root: Path, processor, model, device: str, tgt_lang: str) -> list[str]:
    import librosa
    import torch

    audios = []
    for sample in samples:
        audio_path = resolve_audio_path(sample["audio"], audio_root)
        wav, _ = librosa.load(str(audio_path), sr=16000)
        audios.append(wav)

    inputs = processor(
        audios=audios,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    ).to(device)

    if device != "cpu":
        inputs = inputs.to(torch.float16)

    with torch.no_grad():
        generated = model.generate(**inputs, tgt_lang=tgt_lang)

    preds = processor.batch_decode(generated, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    del audios, inputs, generated
    if device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="C4 end-to-end speech-to-text translation")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "common_data" / "dataset.json"))
    parser.add_argument("--input_dir", default="", help="Optional C1 audio directory. If set, scan audio files directly.")
    parser.add_argument("--audio_root", default="", help="Root for relative audio paths. Default: dataset parent.")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "model" / "seamless-m4t-v2-large"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "C4_end2end" / "outputs_core"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=0, help="0 means run to the end.")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--tgt_lang", default="cmn", help="Mandarin Chinese target language code for SeamlessM4T.")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0")

    model_path = str(Path(args.model).expanduser())
    configure_offline(model_path, args.offline)

    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

    dataset_path = Path(args.dataset).expanduser() if args.dataset else None
    input_dir = Path(args.input_dir).expanduser() if args.input_dir else None
    samples, inferred_audio_root = load_samples(dataset_path, input_dir)
    audio_root = Path(args.audio_root).expanduser() if args.audio_root else inferred_audio_root

    if not samples:
        raise ValueError(
            "No input samples found. Check --dataset, or use --input_dir to scan C1 audio folders directly."
        )

    end = len(samples) if args.count <= 0 else min(len(samples), args.start + args.count)
    run_samples = samples[args.start:end]
    if not run_samples:
        raise ValueError(f"Empty run range: start={args.start}, end={end}, total={len(samples)}")

    outdir = Path(args.outdir).expanduser()
    results_path = outdir / "c4_results.json"
    summary_path = outdir / "c4_summary.json"
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[C4] dataset={dataset_path.resolve() if dataset_path else ''}")
    print(f"[C4] input_dir={input_dir.resolve() if input_dir else ''}")
    print(f"[C4] audio_root={audio_root.resolve()}")
    print(f"[C4] total_samples={len(samples)} run_range=[{args.start}, {end}) run_samples={len(run_samples)}")
    print(f"[C4] model={model_path}")
    print(f"[C4] outdir={outdir.resolve()}")
    print(f"[C4] device={device} batch_size={args.batch_size} tgt_lang={args.tgt_lang}")

    load_t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_path)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device).eval()
    load_time = time.time() - load_t0
    print(f"[C4] model loaded in {load_time:.1f}s")

    results = []
    total_infer_time = 0.0
    wall_t0 = time.time()

    for offset in range(0, len(run_samples), args.batch_size):
        chunk = run_samples[offset : offset + args.batch_size]
        chunk_start = args.start + offset
        chunk_end = chunk_start + len(chunk)
        print(f"[C4] batch [{chunk_start}, {chunk_end}) size={len(chunk)}")

        chunk_t0 = time.time()
        translations = infer_chunk(chunk, audio_root, processor, model, device, args.tgt_lang)
        chunk_time = time.time() - chunk_t0
        total_infer_time += chunk_time

        for sample, translation in zip(chunk, translations):
            audio_abs = resolve_audio_path(sample["audio"], audio_root)
            results.append(
                {
                    "id": sample["id"],
                    "audio": sample["audio"],
                    "audio_abs": str(audio_abs),
                    "reference_en": sample.get("reference_en", ""),
                    "emotion": sample.get("emotion", ""),
                    "speaker": sample.get("speaker", ""),
                    "e2e_translation_zh": translation,
                }
            )

        summary = {
            "module": "C4",
            "task": "end_to_end_speech_to_text_translation",
            "model": model_path,
            "dataset": str(dataset_path.resolve()) if dataset_path else "",
            "input_dir": str(input_dir.resolve()) if input_dir else "",
            "audio_root": str(audio_root.resolve()),
            "num_samples": len(results),
            "dataset_total_samples": len(samples),
            "processed_range": [args.start, args.start + len(results)],
            "batch_size": args.batch_size,
            "tgt_lang": args.tgt_lang,
            "model_load_time_sec": round(load_time, 3),
            "e2e_time_sec": round(total_infer_time, 3),
            "wall_time_sec": round(time.time() - wall_t0, 3),
            "avg_time_sec_per_sample": round(total_infer_time / len(results), 6) if results else None,
            "completed": len(results) == len(run_samples),
            "c5_text_field": "e2e_translation_zh",
        }
        save_json(results, results_path)
        save_json(summary, summary_path)
        print(f"[C4] batch done in {chunk_time:.2f}s; saved {len(results)} rows")

    print(f"[C4] done. results={results_path.resolve()}")
    print(f"[C4] done. summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
