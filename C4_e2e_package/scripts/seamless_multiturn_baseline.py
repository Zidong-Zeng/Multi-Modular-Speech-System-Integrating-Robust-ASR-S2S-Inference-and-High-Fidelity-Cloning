# -*- coding: utf-8 -*-
"""
Single-turn SeamlessM4T baseline for the audio-only multi-turn dataset.

This script intentionally does not keep dialogue history. Each turn is translated
from the current audio only, so it is a good contrast against Qwen2.5-Omni's
history mode.
"""
import argparse
import gc
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import librosa
import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText


HERE = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_path(path, base_dir):
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(base_dir, path)),
        os.path.abspath(os.path.join(os.path.dirname(base_dir), path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def translate_audio(audio_path, processor, model, device, tgt_lang):
    wav, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(
        audios=[wav],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    ).to(device)
    if device != "cpu":
        inputs = inputs.to(torch.float16)

    with torch.no_grad():
        generated = model.generate(**inputs, tgt_lang=tgt_lang)

    pred = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    del wav, inputs, generated
    if device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(HERE, "..", "common_data", "multiturn_dialogues.json"))
    parser.add_argument("--model", default=os.path.join(HERE, "..", "model", "seamless-m4t-v2-large"))
    parser.add_argument("--outdir", default=os.path.join(HERE, "..", "C4_end2end", "outputs_seamless_multiturn"))
    parser.add_argument("--start_dialogue", type=int, default=0)
    parser.add_argument("--count_dialogues", type=int, default=0, help="0 means run all dialogues.")
    parser.add_argument("--tgt_lang", default="cmn")
    parser.add_argument("--allow_missing_audio", action="store_true")
    args = parser.parse_args()

    full_dataset = load_json(args.dataset)
    end_dialogue = len(full_dataset) if args.count_dialogues <= 0 else min(
        len(full_dataset), args.start_dialogue + args.count_dialogues
    )
    dataset = full_dataset[args.start_dialogue:end_dialogue]
    dataset_dir = os.path.dirname(os.path.abspath(args.dataset))

    os.makedirs(args.outdir, exist_ok=True)
    results_path = os.path.join(args.outdir, "seamless_multiturn_results.json")
    summary_path = os.path.join(args.outdir, "seamless_multiturn_summary.json")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] dataset={os.path.abspath(args.dataset)} dialogues={len(dataset)} range=[{args.start_dialogue}, {end_dialogue}) total={len(full_dataset)}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] outdir={args.outdir}")
    print(f"[INFO] device={device} tgt_lang={args.tgt_lang}")

    load_t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device).eval()
    print(f"[INFO] model loaded in {time.time() - load_t0:.1f}s")

    rows = []
    total_infer_time = 0.0
    wall_t0 = time.time()

    for dialogue in dataset:
        dialogue_id = dialogue.get("dialogue_id", "dialogue")
        for turn in dialogue.get("turns", []):
            turn_id = turn.get("turn_id", len(rows) + 1)
            audio = resolve_path(turn["audio"], dataset_dir)
            if not os.path.exists(audio):
                message = f"Missing audio for {dialogue_id} turn {turn_id}: {audio}"
                if not args.allow_missing_audio:
                    raise FileNotFoundError(message)
                print(f"[WARN] {message}")
                rows.append({
                    "dialogue_id": dialogue_id,
                    "mode": "seamless_single_turn",
                    "turn_id": turn_id,
                    "audio": audio,
                    "spoken_content": turn.get("spoken_content", ""),
                    "expected": turn.get("expected", ""),
                    "response": "",
                    "missing_audio": True,
                    "infer_time_sec": 0.0,
                })
                continue

            t0 = time.time()
            response = translate_audio(audio, processor, model, device, args.tgt_lang)
            elapsed = time.time() - t0
            total_infer_time += elapsed
            row = {
                "dialogue_id": dialogue_id,
                "mode": "seamless_single_turn",
                "turn_id": turn_id,
                "audio": audio,
                "spoken_content": turn.get("spoken_content", ""),
                "expected": turn.get("expected", ""),
                "response": response,
                "missing_audio": False,
                "infer_time_sec": round(elapsed, 4),
            }
            rows.append(row)
            print(f"[INFO] seamless {dialogue_id} turn={turn_id} time={elapsed:.2f}s response={response}")
            save_json(rows, results_path)

    summary = {
        "model": args.model,
        "dataset": os.path.abspath(args.dataset),
        "num_dialogues": len(dataset),
        "dataset_total_dialogues": len(full_dataset),
        "processed_dialogue_range": [args.start_dialogue, end_dialogue],
        "num_result_rows": len(rows),
        "mode": "seamless_single_turn",
        "tgt_lang": args.tgt_lang,
        "seamless_multiturn_time_sec": round(total_infer_time, 4),
        "seamless_multiturn_wall_time_sec": round(time.time() - wall_t0, 4),
        "history_supported": False,
        "description": "Each audio turn is translated independently; no dialogue history is used.",
    }
    save_json(summary, summary_path)
    print(f"[DONE] saved results: {results_path}")
    print(f"[DONE] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
