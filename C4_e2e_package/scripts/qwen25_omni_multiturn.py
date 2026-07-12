# -*- coding: utf-8 -*-
"""
Run audio-only multi-turn dialogue experiments with Qwen2.5-Omni.

The task instruction is expected to be inside the user's audio. This script
does not add a textual user prompt such as "translate this audio"; each user
turn is passed as audio only.

Typical outputs:
  C4_end2end/outputs_qwen25_multiturn/qwen25_omni_multiturn_results.json
  C4_end2end/outputs_qwen25_multiturn/qwen25_omni_multiturn_summary.json
"""
import argparse
import gc
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import soundfile as sf
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
except Exception as exc:
    print("[ERROR] Qwen2.5-Omni dependencies are not ready:", repr(exc))
    print("Install the official preview transformers and qwen-omni-utils first:")
    print("  pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview")
    print("  pip install qwen-omni-utils -U")
    raise


HERE = os.path.dirname(os.path.abspath(__file__))


TEMPLATE_DATASET = [
    {
        "dialogue_id": "dlg_001_weather_umbrella",
        "description": "Audio-only context-dependent translation/dialogue test.",
        "turns": [
            {
                "turn_id": 1,
                "audio": "common_data/multiturn_audio/dlg_001/turn1.wav",
                "spoken_content": (
                    "please translate this English voice into Chinese: "
                    "it's raining outside, don't forget to take your umbrella."
                ),
                "expected": "外面正在下雨，别忘了带伞。"
            },
            {
                "turn_id": 2,
                "audio": "common_data/multiturn_audio/dlg_001/turn2.wav",
                "spoken_content": "what's the weather like now?",
                "expected": "现在外面在下雨。"
            },
            {
                "turn_id": 3,
                "audio": "common_data/multiturn_audio/dlg_001/turn3.wav",
                "spoken_content": "How should I reply to this sentence?",
                "expected": "可以回复：好的，谢谢提醒，我会带伞的。"
            }
        ]
    }
]


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


def clean_response(text):
    text = text.strip()
    marker = "\nassistant\n"
    if marker in text:
        text = text.rsplit(marker, 1)[-1].strip()
    elif "assistant\n" in text:
        text = text.rsplit("assistant\n", 1)[-1].strip()
    return text.strip()


def normalize_audio_outputs(audio_outputs):
    if audio_outputs is None:
        return []
    if isinstance(audio_outputs, torch.Tensor):
        if audio_outputs.ndim == 1:
            return [audio_outputs.detach().cpu().float().numpy()]
        return [item.detach().cpu().float().numpy() for item in audio_outputs]
    if isinstance(audio_outputs, (list, tuple)):
        normalized = []
        for item in audio_outputs:
            if isinstance(item, torch.Tensor):
                normalized.append(item.detach().cpu().float().numpy())
            else:
                normalized.append(item)
        return normalized
    return [audio_outputs]


def save_audio_output(audio_item, path, sample_rate):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio_item, sample_rate)


def make_audio_user_turn(audio_path):
    return {
        "role": "user",
        "content": [{"type": "audio", "audio": audio_path}],
    }


def make_assistant_turn(text):
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    }


def generate_one(processor, model, messages, max_new_tokens, return_audio, speaker):
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            use_audio_in_video=False,
            return_audio=return_audio,
            max_new_tokens=max_new_tokens,
            speaker=speaker,
        )

    audio_outputs = []
    if return_audio:
        generated, audio_outputs = generated
        audio_outputs = normalize_audio_outputs(audio_outputs)

    decoded = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    response = clean_response(decoded[0] if decoded else "")

    del text, audios, images, videos, inputs, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return response, audio_outputs


def run_dialogue(
    dialogue,
    mode,
    dataset_dir,
    outdir,
    processor,
    model,
    max_new_tokens,
    return_audio,
    speaker,
    audio_sample_rate,
    strict_missing,
):
    dialogue_id = dialogue.get("dialogue_id", "dialogue")
    history = []
    rows = []
    infer_time = 0.0

    for turn in dialogue.get("turns", []):
        turn_id = turn.get("turn_id", len(rows) + 1)
        audio = resolve_path(turn["audio"], dataset_dir)
        if not os.path.exists(audio):
            message = f"Missing audio for {dialogue_id} turn {turn_id}: {audio}"
            if strict_missing:
                raise FileNotFoundError(message)
            print(f"[WARN] {message}")
            rows.append({
                "dialogue_id": dialogue_id,
                "mode": mode,
                "turn_id": turn_id,
                "audio": audio,
                "expected": turn.get("expected", ""),
                "response": "",
                "output_audio": "",
                "missing_audio": True,
            })
            continue

        user_turn = make_audio_user_turn(audio)
        messages = history + [user_turn] if mode == "history" else [user_turn]

        t0 = time.time()
        response, audio_outputs = generate_one(
            processor=processor,
            model=model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            return_audio=return_audio,
            speaker=speaker,
        )
        elapsed = time.time() - t0
        infer_time += elapsed

        output_audio = ""
        if return_audio and audio_outputs:
            audio_dir = os.path.join(outdir, "audio", mode, dialogue_id)
            output_audio = os.path.join(audio_dir, f"turn{turn_id}.wav")
            save_audio_output(audio_outputs[0], output_audio, audio_sample_rate)

        row = {
            "dialogue_id": dialogue_id,
            "mode": mode,
            "turn_id": turn_id,
            "audio": audio,
            "spoken_content": turn.get("spoken_content", ""),
            "expected": turn.get("expected", ""),
            "response": response,
            "output_audio": output_audio,
            "missing_audio": False,
            "infer_time_sec": round(elapsed, 4),
        }
        rows.append(row)
        print(f"[INFO] {mode} {dialogue_id} turn={turn_id} time={elapsed:.1f}s response={response}")

        if mode == "history":
            history.append(user_turn)
            history.append(make_assistant_turn(response))

    return rows, infer_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(HERE, "..", "common_data", "multiturn_dialogues.json"))
    parser.add_argument("--model", default=os.path.join(HERE, "..", "model", "Qwen2.5-Omni-7B"))
    parser.add_argument("--outdir", default=os.path.join(HERE, "..", "C4_end2end", "outputs_qwen25_multiturn"))
    parser.add_argument("--mode", default="both", choices=["history", "single", "both"])
    parser.add_argument("--start_dialogue", type=int, default=0)
    parser.add_argument("--count_dialogues", type=int, default=0, help="0 means run all dialogues.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--return_audio", action="store_true")
    parser.add_argument("--speaker", default="Chelsie", choices=["Chelsie", "Ethan"])
    parser.add_argument("--audio_sample_rate", type=int, default=24000)
    parser.add_argument("--keep_talker", action="store_true")
    parser.add_argument("--allow_missing_audio", action="store_true", help="Write placeholder rows if audio files are not uploaded yet.")
    parser.add_argument("--init_template", action="store_true", help="Create a template multiturn_dialogues.json and exit.")
    args = parser.parse_args()

    if args.init_template:
        save_json(TEMPLATE_DATASET, args.dataset)
        print(f"[DONE] wrote template dataset: {args.dataset}")
        return

    full_dataset = load_json(args.dataset)
    end_dialogue = len(full_dataset) if args.count_dialogues <= 0 else min(
        len(full_dataset),
        args.start_dialogue + args.count_dialogues,
    )
    dataset = full_dataset[args.start_dialogue:end_dialogue]
    dataset_dir = os.path.dirname(os.path.abspath(args.dataset))
    os.makedirs(args.outdir, exist_ok=True)
    results_path = os.path.join(args.outdir, "qwen25_omni_multiturn_results.json")
    summary_path = os.path.join(args.outdir, "qwen25_omni_multiturn_summary.json")

    print(
        f"[INFO] dataset={os.path.abspath(args.dataset)} "
        f"dialogues={len(dataset)} range=[{args.start_dialogue}, {end_dialogue}) "
        f"total={len(full_dataset)}"
    )
    print(f"[INFO] model={args.model}")
    print(f"[INFO] outdir={args.outdir}")
    print(f"[INFO] mode={args.mode} return_audio={args.return_audio} speaker={args.speaker}")

    t_load = time.time()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    if not args.keep_talker and not args.return_audio:
        model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model)
    print(f"[INFO] model loaded in {time.time() - t_load:.1f}s")

    modes = ["history", "single"] if args.mode == "both" else [args.mode]
    all_rows = []
    total_infer_time = 0.0
    wall_t0 = time.time()

    for dialogue in dataset:
        for mode in modes:
            rows, infer_time = run_dialogue(
                dialogue=dialogue,
                mode=mode,
                dataset_dir=dataset_dir,
                outdir=args.outdir,
                processor=processor,
                model=model,
                max_new_tokens=args.max_new_tokens,
                return_audio=args.return_audio,
                speaker=args.speaker,
                audio_sample_rate=args.audio_sample_rate,
                strict_missing=not args.allow_missing_audio,
            )
            all_rows.extend(rows)
            total_infer_time += infer_time
            save_json(all_rows, results_path)

    summary = {
        "model": args.model,
        "dataset": os.path.abspath(args.dataset),
        "num_dialogues": len(dataset),
        "dataset_total_dialogues": len(full_dataset),
        "processed_dialogue_range": [args.start_dialogue, end_dialogue],
        "num_result_rows": len(all_rows),
        "mode": args.mode,
        "max_new_tokens": args.max_new_tokens,
        "return_audio": args.return_audio,
        "speaker": args.speaker if args.return_audio else None,
        "audio_sample_rate": args.audio_sample_rate if args.return_audio else None,
        "qwen25_omni_multiturn_time_sec": round(total_infer_time, 4),
        "qwen25_omni_multiturn_wall_time_sec": round(time.time() - wall_t0, 4),
        "audio_only_user_turns": True,
        "text_user_prompt_added": False,
        "history_mode_description": "history keeps previous user audio turns and assistant text responses.",
        "single_mode_description": "single runs each user audio turn independently without dialogue history.",
    }
    save_json(summary, summary_path)
    print(f"[DONE] saved results: {results_path}")
    print(f"[DONE] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
