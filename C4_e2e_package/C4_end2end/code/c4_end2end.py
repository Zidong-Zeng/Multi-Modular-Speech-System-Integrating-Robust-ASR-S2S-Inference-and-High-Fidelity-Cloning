# -*- coding: utf-8 -*-
"""
C4 end-to-end speech translation with chunked batching.

The script loads SeamlessM4T once, runs the selected dataset range in chunks
(`--batch_size`, default 50), and keeps one cumulative result file:

  - c4_results.json
  - c4_summary.json
"""
import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _maybe_offline():
    argv = sys.argv
    off = "--offline" in argv
    if "--model" in argv:
        k = argv.index("--model") + 1
        if k < len(argv) and os.path.isdir(argv[k]):
            off = True
    if off:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


_maybe_offline()

import librosa
import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

HERE = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_c3_map(c3_path):
    c3_map = {}
    if os.path.exists(c3_path):
        for r in load_json(c3_path):
            c3_map[r["id"]] = r.get("translation_zh", "")
    return c3_map


def get_c3_total_time():
    c3_sum_path = os.path.join(HERE, "..", "..", "C3_cascade", "outputs", "c3_summary.json")
    if not os.path.exists(c3_sum_path):
        return None
    try:
        return load_json(c3_sum_path).get("total_time_sec")
    except Exception:
        return None


def absolute_audio_path(data_root, sample):
    audio = sample["audio"]
    return audio if os.path.isabs(audio) else os.path.join(data_root, audio)


def infer_chunk(chunk, data_root, processor, model, device, tgt_lang):
    audios = []
    for sample in chunk:
        audio_path = absolute_audio_path(data_root, sample)
        wav, _ = librosa.load(audio_path, sr=16000)
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
    preds = [p.strip() for p in preds]

    del audios, inputs, generated
    if device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(HERE, "..", "..", "common_data", "dataset.json"))
    parser.add_argument("--model", default="facebook/seamless-m4t-v2-large")
    parser.add_argument("--c3", default=os.path.join(HERE, "..", "..", "C3_cascade", "outputs", "cascade_results.json"))
    parser.add_argument("--start", type=int, default=0, help="Start dataset index. Default: 0")
    parser.add_argument("--count", type=int, default=0, help="Number of samples. Default 0 means run to dataset end.")
    parser.add_argument("--batch_size", type=int, default=50, help="Chunk size. Default: 50")
    parser.add_argument("--tgt_lang", default="cmn", help="Target language. cmn = Mandarin Chinese.")
    parser.add_argument("--offline", action="store_true", help="Use local/cache model only.")
    parser.add_argument("--outdir", default=os.path.join(HERE, "..", "outputs"))
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0")

    os.makedirs(args.outdir, exist_ok=True)
    results_path = os.path.join(args.outdir, "c4_results.json")
    summary_path = os.path.join(args.outdir, "c4_summary.json")

    dataset = load_json(args.dataset)
    data_root = os.path.dirname(os.path.abspath(args.dataset))
    end = len(dataset) if args.count <= 0 else min(len(dataset), args.start + args.count)
    samples = dataset[args.start:end]

    print(f"Dataset: {os.path.abspath(args.dataset)}")
    print(f"Total samples: {len(dataset)}; this run: {len(samples)}; range=[{args.start}, {end})")
    print(f"Chunk size: {args.batch_size}")

    c3_map = build_c3_map(args.c3)
    c3_total = get_c3_total_time()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"Loading end-to-end model: {args.model}")
    print("=" * 60)
    load_t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device).eval()
    print(f"Model load time: {time.time() - load_t0:.1f}s", end="")
    if device != "cpu":
        print(f"; GPU allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print()

    results = []
    total_e2e_time = 0.0
    total_wall_t0 = time.time()

    print(f"\nStarting chunked inference for {len(samples)} samples...")
    for chunk_offset in range(0, len(samples), args.batch_size):
        chunk = samples[chunk_offset:chunk_offset + args.batch_size]
        global_start = args.start + chunk_offset
        global_end = global_start + len(chunk)

        print("-" * 60)
        print(f"Batch range: [{global_start}, {global_end}); batch size={len(chunk)}")

        chunk_t0 = time.time()
        preds = infer_chunk(chunk, data_root, processor, model, device, args.tgt_lang)
        chunk_time = time.time() - chunk_t0
        total_e2e_time += chunk_time

        for sample, translation in zip(chunk, preds):
            results.append({
                "id": sample["id"],
                "audio": sample["audio"],
                "reference_en": sample.get("text", ""),
                "emotion": sample.get("emotion", ""),
                "speaker": sample.get("speaker", ""),
                "e2e_translation_zh": translation,
                "c3_cascade_translation_zh": c3_map.get(sample["id"], ""),
            })

        summary = {
            "model": args.model,
            "num_samples": len(results),
            "dataset_total_samples": len(dataset),
            "start": args.start,
            "count_requested": args.count,
            "processed_range": [args.start, args.start + len(results)],
            "batch_size": args.batch_size,
            "tgt_lang": args.tgt_lang,
            "e2e_time_sec": round(total_e2e_time, 1),
            "e2e_wall_time_sec": round(time.time() - total_wall_t0, 1),
            "avg_time_sec_per_sample": round(total_e2e_time / len(results), 4) if results else None,
            "c3_cascade_time_sec": c3_total,
            "completed": len(results) == len(samples),
        }

        save_json(results, results_path)
        save_json(summary, summary_path)

        print(f"Batch done in {chunk_time:.1f}s; accumulated {len(results)}/{len(samples)} samples")
        print(f"Updated: {results_path}")
        print(f"Updated: {summary_path}")

    print("\nPreview: C4 end-to-end vs C3 cascade")
    for result in results[:10]:
        print(f"[{result['id']}] reference_en: {result['reference_en']}")
        print(f"  C4: {result['e2e_translation_zh']}")
        print(f"  C3: {result['c3_cascade_translation_zh']}")

    print(f"\nTotal C4 e2e time for {len(results)} samples: {total_e2e_time:.1f}s")
    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
