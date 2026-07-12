# -*- coding: utf-8 -*-
"""
Run Qwen2.5-Omni-7B on the shared C dataset.

This script keeps outputs separated from Kimi/SeamlessM4T:
  C4_end2end/outputs_qwen25_omni/qwen25_omni_results.json
  C4_end2end/outputs_qwen25_omni/qwen25_omni_summary.json

It uses text-only generation (`return_audio=False`) by default. Add
`--return_audio` to save Qwen's Chinese speech output as wav files.
"""
import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import joblib
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    from prosody_utils import extract_prosody_features, style_prompt_for_emotion, vectorize_features
except Exception as exc:
    print("[ERROR] Qwen2.5-Omni dependencies are not ready:", repr(exc))
    print("Install the official preview transformers and qwen-omni-utils first:")
    print("  pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview")
    print("  pip install qwen-omni-utils -U")
    raise

HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

TRANSLATION_PROMPT = (
    "You are working as a strict speech translation system and professional emotional voice actor. "
    "Listen to the English speech, preserve the speaker's emotion, rhythm, intensity, and tone, "
    "and translate its meaning into Simplified Chinese. Output only the Chinese "
    "translation. Do not repeat the English transcript. Do not add explanations, "
    "greetings, apologies, suggestions, or follow-up offers."
)

GENERIC_STYLE_PROMPT = (
    "Preserve the source speaker's emotional tone, speaking pace, intensity, and prosody. "
    "Do not flatten the delivery into a neutral narration."
)

UNWANTED_TAILS = [
    "如果还有类似的翻译问题，或者其他事，都可以跟我说哦。",
    "如果还有类似的翻译问题，或者其他事，都可以跟我说。",
    "如果还有其他问题，请随时告诉我。",
    "如果你还有其他问题，可以继续问我。",
    "如果需要其他帮助，请告诉我。",
]

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
EMOTION2VEC_LABEL_MAP = {
    "angry": "angry",
    "anger": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "other",
    "surprised": "other",
    "other": "other",
    "unknown": "unknown",
    "<unk>": "unknown",
}

EMOTION2VEC_STYLE_PROMPTS = {
    "angry": "The source emotion is angry. Use a tense, sharp, firm voice with stronger intensity.",
    "disgust": "The source emotion is disgust. Use a displeased, rejecting voice with clear aversion.",
    "fear": "The source emotion is fear. Use a nervous, cautious voice with slight tension.",
    "happy": "The source emotion is happy. Use a bright, lively, smiling voice.",
    "neutral": "The source emotion is neutral. Use a calm, plain, steady voice.",
    "sad": "The source emotion is sad. Use a low, slow, weak voice with reduced energy.",
}

REFERENCE_AUDIO_PROMPT = (
    "The first audio is a Chinese emotional reference. Use it only as a speaking style reference. "
    "Do not translate, imitate the words, or repeat the content of the first audio. "
    "The second audio is the English speech to translate. "
    "Translate the second audio into Simplified Chinese and speak the Chinese translation with the same emotional tone, rhythm, and intensity as the first audio."
)

DISGUST_NO_REFERENCE_PROMPT = (
    "The source emotion is disgust, but no Chinese disgust reference audio is available. "
    "Use text-only style control: speak with a displeased, rejecting voice with clear aversion."
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def audio_path(data_root, sample):
    audio = sample["audio"]
    return audio if os.path.isabs(audio) else os.path.join(data_root, audio)


def make_conversation(sample, data_root, style_prompt="", reference_audio=""):
    style_text = ""
    if style_prompt:
        style_text = f" Style cue: {style_prompt}"
    content = []
    if reference_audio:
        content.append({"type": "audio", "audio": reference_audio})
    content.append({"type": "audio", "audio": audio_path(data_root, sample)})
    if reference_audio:
        instruction = (
            f"{TRANSLATION_PROMPT} "
            f"{REFERENCE_AUDIO_PROMPT} "
            "Output only the Chinese translation itself."
            f"{style_text}"
        )
    else:
        instruction = (
            f"{TRANSLATION_PROMPT} "
            "Translate this speech into Simplified Chinese. "
            "Only output the translation itself, and keep the original emotion."
            f"{style_text}"
        )
    content.append({"type": "text", "text": instruction})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": content,
        },
    ]


def make_emotion_conversation(sample, data_root):
    labels = ", ".join(EMOTION_LABELS)
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a strict speech emotion classifier. "
                        "Listen to the speech audio and classify the speaker's emotion. "
                        f"Choose exactly one label from: {labels}. "
                        "Output only the label, with no explanation."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path(data_root, sample)},
                {"type": "text", "text": f"Classify the emotion. Labels: {labels}."},
            ],
        },
    ]


def clean_translation(text):
    text = text.strip()
    marker = "\nassistant\n"
    if marker in text:
        text = text.rsplit(marker, 1)[-1].strip()
    elif "assistant\n" in text:
        text = text.rsplit("assistant\n", 1)[-1].strip()

    for tail in UNWANTED_TAILS:
        if text.endswith(tail):
            text = text[: -len(tail)].rstrip()

    # If the model still produces multiple paragraphs, keep the translation-like
    # first paragraph instead of the conversational follow-up.
    parts = [part.strip() for part in text.splitlines() if part.strip()]
    if parts:
        text = parts[0]

    return text


def clean_predicted_emotion(text):
    text = text.strip().lower()
    marker = "\nassistant\n"
    if marker in text:
        text = text.rsplit(marker, 1)[-1].strip().lower()
    elif "assistant\n" in text:
        text = text.rsplit("assistant\n", 1)[-1].strip().lower()

    for label in EMOTION_LABELS:
        if label in text:
            return label
    return text.split()[0].strip(".,;:!()[]{}\"'") if text.split() else ""


def normalize_ser_label(label):
    label = str(label or "").strip().lower()
    if "/" in label:
        label = label.rsplit("/", 1)[-1].strip()
    return EMOTION2VEC_LABEL_MAP.get(label, label)


def pick_emotion2vec_prediction(raw):
    first = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(first, dict):
        return "", "", None, raw

    labels = first.get("labels") or first.get("label")
    scores = first.get("scores") or first.get("score")
    if isinstance(labels, list) and labels:
        if isinstance(scores, list) and len(scores) == len(labels):
            best_idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
            raw_label = str(labels[best_idx])
            confidence = float(scores[best_idx])
        else:
            raw_label = str(labels[0])
            confidence = None
        return normalize_ser_label(raw_label), raw_label, confidence, raw

    if isinstance(labels, str):
        confidence = float(scores) if isinstance(scores, (int, float)) else None
        return normalize_ser_label(labels), labels, confidence, raw

    if isinstance(scores, dict) and scores:
        raw_label, confidence = max(scores.items(), key=lambda kv: float(kv[1]))
        return normalize_ser_label(raw_label), str(raw_label), float(confidence), raw

    return "", "", None, raw


def load_emotion2vec_model(path):
    from funasr import AutoModel

    return AutoModel(model=path, disable_update=True)


def resolve_reference_audio(reference_dir, emotion):
    if not reference_dir or not emotion:
        return "", ""
    if emotion == "disgust":
        return "", "no_disgust_reference"
    path = os.path.join(reference_dir, f"{emotion}.wav")
    if os.path.exists(path):
        return path, "reference_audio"
    return "", "missing_reference_audio"


def predict_emotion2vec_batch(batch, data_root, ser_model, output_dir, confidence_threshold, reference_dir=""):
    rows = []
    os.makedirs(output_dir, exist_ok=True)
    for sample in batch:
        raw = ser_model.generate(
            input=audio_path(data_root, sample),
            output_dir=output_dir,
            granularity="utterance",
            extract_embedding=False,
        )
        emotion, raw_label, confidence, raw_output = pick_emotion2vec_prediction(raw)
        style_prompt = EMOTION2VEC_STYLE_PROMPTS.get(emotion, GENERIC_STYLE_PROMPT)
        style_source = "emotion2vec"
        reference_audio, reference_status = resolve_reference_audio(reference_dir, emotion)
        if confidence is not None and confidence < confidence_threshold:
            style_prompt = GENERIC_STYLE_PROMPT
            style_source = "emotion2vec_fallback"
            reference_audio = ""
            reference_status = "confidence_fallback"
        elif reference_status == "reference_audio":
            style_prompt = (
                f"{EMOTION2VEC_STYLE_PROMPTS.get(emotion, GENERIC_STYLE_PROMPT)} "
                "Follow the reference audio for prosody and emotional delivery."
            )
            style_source = "emotion2vec_reference_audio"
        elif reference_status == "no_disgust_reference":
            style_prompt = DISGUST_NO_REFERENCE_PROMPT
            style_source = "emotion2vec_no_reference_disgust"
        elif reference_dir and reference_status == "missing_reference_audio":
            style_source = "emotion2vec_missing_reference_audio"
        rows.append({
            "emotion2vec_emotion": emotion,
            "emotion2vec_raw_label": raw_label,
            "emotion2vec_confidence": confidence,
            "emotion2vec_confidence_threshold": confidence_threshold,
            "emotion2vec_raw_output": raw_output,
            "reference_audio": reference_audio,
            "reference_audio_status": reference_status,
            "style_prompt": style_prompt,
            "style_source": style_source,
        })
    return rows


def predict_emotion2vec_audio(audio_file, ser_model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    raw = ser_model.generate(
        input=audio_file,
        output_dir=output_dir,
        granularity="utterance",
        extract_embedding=False,
    )
    emotion, raw_label, confidence, raw_output = pick_emotion2vec_prediction(raw)
    return {
        "emotion": emotion,
        "raw_label": raw_label,
        "confidence": confidence,
        "raw_output": raw_output,
    }


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


def load_style_controller(path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid controller bundle: {path}")
    return bundle


def predict_style_batch(batch, data_root, controller_bundle, confidence_threshold):
    model = controller_bundle["model"]
    feature_names = controller_bundle.get("feature_names")
    style_prompts = controller_bundle.get("style_prompts", {})
    rows = []
    for sample in batch:
        features = extract_prosody_features(audio_path(data_root, sample))
        X = np.asarray([vectorize_features(features, feature_names)], dtype=np.float32)
        emotion = str(model.predict(X)[0])
        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            confidence = float(np.max(probs))
        style_prompt = style_prompts.get(emotion, style_prompt_for_emotion(emotion))
        if confidence is not None and confidence < confidence_threshold:
            style_prompt = GENERIC_STYLE_PROMPT
        rows.append({
            "controller_emotion": emotion,
            "controller_confidence": confidence,
            "style_prompt": style_prompt,
            "style_confidence_threshold": confidence_threshold,
            "prosody_features": features,
        })
    return rows


def infer_batch(
    batch,
    data_root,
    processor,
    model,
    max_new_tokens,
    return_audio,
    speaker,
    style_rows=None,
    do_sample=False,
    temperature=0.8,
    top_p=0.9,
):
    style_rows = style_rows or [{} for _ in batch]
    conversations = [
        make_conversation(
            sample,
            data_root,
            style_rows[idx].get("style_prompt", ""),
            style_rows[idx].get("reference_audio", ""),
        )
        for idx, sample in enumerate(batch)
    ]
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
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
        generate_kwargs = {
            "use_audio_in_video": False,
            "return_audio": return_audio,
            "max_new_tokens": max_new_tokens,
            "speaker": speaker,
        }
        if do_sample:
            generate_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })
        generated = model.generate(**inputs, **generate_kwargs)

    audio_outputs = []
    if return_audio:
        generated, audio_outputs = generated
        audio_outputs = normalize_audio_outputs(audio_outputs)

    outputs = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    outputs = [clean_translation(item) for item in outputs]

    del conversations, text, audios, images, videos, inputs, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return outputs, audio_outputs


def score_candidate(target_emotion, predicted_emotion, confidence):
    confidence = float(confidence or 0.0)
    if target_emotion in EMOTION_LABELS and predicted_emotion == target_emotion:
        return 3.0 + confidence
    if target_emotion in EMOTION_LABELS and target_emotion != "neutral":
        if predicted_emotion == "neutral":
            return -1.0 + 0.1 * confidence
        if predicted_emotion in EMOTION_LABELS:
            return 1.0 + 0.5 * confidence
        return -0.5 + 0.1 * confidence
    if predicted_emotion in EMOTION_LABELS:
        return confidence
    return -0.5 + 0.1 * confidence


def infer_and_rerank_sample(
    sample,
    data_root,
    processor,
    model,
    ser_model,
    max_new_tokens,
    speaker,
    style_row,
    audio_sample_rate,
    candidate_count,
    candidate_audio_dir,
    candidate_ser_dir,
    do_sample,
    temperature,
    top_p,
):
    sample_id = sample.get("id", "sample")
    target_emotion = style_row.get("emotion2vec_emotion", "")
    candidates = []

    for cand_idx in range(candidate_count):
        translations, audio_outputs = infer_batch(
            batch=[sample],
            data_root=data_root,
            processor=processor,
            model=model,
            max_new_tokens=max_new_tokens,
            return_audio=True,
            speaker=speaker,
            style_rows=[style_row],
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        if not audio_outputs:
            raise RuntimeError(f"No audio output for candidate {cand_idx} of {sample_id}")

        cand_audio = os.path.join(candidate_audio_dir, f"qwen25_omni_{sample_id}_cand{cand_idx + 1}.wav")
        save_audio_output(audio_outputs[0], cand_audio, audio_sample_rate)
        ser = predict_emotion2vec_audio(cand_audio, ser_model, candidate_ser_dir)
        cand_score = score_candidate(target_emotion, ser["emotion"], ser["confidence"])
        candidates.append({
            "candidate_index": cand_idx + 1,
            "translation": translations[0] if translations else "",
            "audio": cand_audio,
            "ser_emotion": ser["emotion"],
            "ser_raw_label": ser["raw_label"],
            "ser_confidence": ser["confidence"],
            "rerank_score": cand_score,
        })

    best = max(candidates, key=lambda item: item["rerank_score"])
    return best["translation"], best["audio"], {
        "target_emotion": target_emotion,
        "selected_candidate_index": best["candidate_index"],
        "selected_candidate_ser_emotion": best["ser_emotion"],
        "selected_candidate_ser_confidence": best["ser_confidence"],
        "candidate_count": candidate_count,
        "candidates": candidates,
    }


def infer_emotion_batch(batch, data_root, processor, model, max_new_tokens):
    conversations = [make_emotion_conversation(sample, data_root) for sample in batch]
    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
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
            return_audio=False,
            max_new_tokens=max_new_tokens,
        )

    outputs = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    outputs = [clean_predicted_emotion(item) for item in outputs]

    del conversations, text, audios, images, videos, inputs, generated
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(HERE, "..", "..", "common_data", "dataset.json"))
    parser.add_argument("--model", default=os.path.join(HERE, "..", "..", "model", "Qwen2.5-Omni-7B"))
    parser.add_argument("--outdir", default=os.path.join(HERE, "..", "outputs_qwen25_omni"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--return_audio", action="store_true", help="Generate and save Chinese speech output.")
    parser.add_argument("--speaker", default="Chelsie", choices=["Chelsie", "Ethan"], help="Qwen output voice.")
    parser.add_argument("--audio_sample_rate", type=int, default=24000, help="Output wav sample rate.")
    parser.add_argument("--keep_talker", action="store_true", help="Do not disable talker. Uses more VRAM.")
    parser.add_argument("--predict_emotion", action="store_true", help="Ask Qwen to predict one emotion label from the input audio.")
    parser.add_argument("--emotion_max_new_tokens", type=int, default=16, help="Max tokens for emotion classification.")
    parser.add_argument("--style_controller", default="", help="Path to trained lightweight emotion/prosody controller.")
    parser.add_argument("--style_conf_threshold", type=float, default=0.6, help="Use generic style prompt if controller confidence is below this value.")
    parser.add_argument("--save_prosody_features", action="store_true", help="Save extracted prosody features in each result when using controller.")
    parser.add_argument("--emotion2vec_style", action="store_true", help="Use emotion2vec SER on source audio to create an emotion style cue for Qwen generation.")
    parser.add_argument("--emotion2vec_model", default=os.path.join(HERE, "..", "..", "model", "emotion2vec_plus_large"), help="Local emotion2vec model directory.")
    parser.add_argument("--emotion2vec_tmp_dir", default="", help="Temporary output dir for emotion2vec inference.")
    parser.add_argument("--emotion2vec_conf_threshold", type=float, default=0.75, help="Use generic style prompt if emotion2vec confidence is below this value.")
    parser.add_argument("--reference_audio_style", action="store_true", help="Use a Chinese emotional reference audio bank to guide Qwen speech style.")
    parser.add_argument("--reference_audio_dir", default=os.path.join(HERE, "..", "..", "common_data", "emotion_refs", "casia_liuchanhg"), help="Directory containing angry/fear/happy/neutral/sad wav reference files.")
    parser.add_argument("--candidate_count", type=int, default=1, help="Generate N audio candidates and select the best one with emotion2vec SER. Requires --return_audio and --emotion2vec_style.")
    parser.add_argument("--candidate_do_sample", action="store_true", default=True, help="Use sampling when generating reranking candidates.")
    parser.add_argument("--candidate_temperature", type=float, default=0.9)
    parser.add_argument("--candidate_top_p", type=float, default=0.95)
    args = parser.parse_args()
    if args.emotion2vec_style and args.style_controller:
        raise SystemExit("Use either --emotion2vec_style or --style_controller, not both.")
    if args.candidate_count < 1:
        raise SystemExit("--candidate_count must be >= 1")
    if args.candidate_count > 1:
        if not args.return_audio or not args.emotion2vec_style:
            raise SystemExit("--candidate_count > 1 requires --return_audio and --emotion2vec_style")
        if args.batch_size != 1:
            raise SystemExit("--candidate_count > 1 requires --batch_size 1")
    if args.reference_audio_style and not args.emotion2vec_style:
        raise SystemExit("--reference_audio_style requires --emotion2vec_style")

    os.makedirs(args.outdir, exist_ok=True)
    audio_outdir = os.path.join(args.outdir, "audio")
    candidate_audio_dir = os.path.join(args.outdir, "audio_candidates")
    results_path = os.path.join(args.outdir, "qwen25_omni_results.json")
    summary_path = os.path.join(args.outdir, "qwen25_omni_summary.json")

    dataset = load_json(args.dataset)
    data_root = os.path.dirname(os.path.abspath(args.dataset))
    end = len(dataset) if args.count <= 0 else min(len(dataset), args.start + args.count)
    samples = dataset[args.start:end]

    print(f"[INFO] dataset={os.path.abspath(args.dataset)} total={len(dataset)} range=[{args.start}, {end}) n={len(samples)}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] outdir={args.outdir}")
    print(f"[INFO] batch_size={args.batch_size} max_new_tokens={args.max_new_tokens}")
    print(f"[INFO] return_audio={args.return_audio} speaker={args.speaker}")
    print(f"[INFO] predict_emotion={args.predict_emotion}")
    print(f"[INFO] style_controller={args.style_controller or None}")
    print(f"[INFO] style_conf_threshold={args.style_conf_threshold}")
    print(f"[INFO] emotion2vec_style={args.emotion2vec_style}")
    print(f"[INFO] emotion2vec_model={args.emotion2vec_model if args.emotion2vec_style else None}")
    print(f"[INFO] emotion2vec_conf_threshold={args.emotion2vec_conf_threshold}")
    print(f"[INFO] reference_audio_style={args.reference_audio_style}")
    print(f"[INFO] reference_audio_dir={args.reference_audio_dir if args.reference_audio_style else None}")
    print(f"[INFO] candidate_count={args.candidate_count}")
    print(f"[INFO] candidate_do_sample={args.candidate_do_sample} temperature={args.candidate_temperature} top_p={args.candidate_top_p}")
    if args.return_audio and args.batch_size != 1:
        print("[WARN] return_audio=True is safest with --batch_size 1. If OOM occurs, rerun with --batch_size 1.")

    style_controller = load_style_controller(args.style_controller) if args.style_controller else None
    emotion2vec_model = None
    emotion2vec_tmp_dir = args.emotion2vec_tmp_dir or os.path.join(args.outdir, "emotion2vec_tmp")
    t_emotion2vec_load = 0.0
    if args.emotion2vec_style:
        t_ser_load = time.time()
        emotion2vec_model = load_emotion2vec_model(args.emotion2vec_model)
        t_emotion2vec_load = time.time() - t_ser_load
        print(f"[INFO] emotion2vec loaded in {t_emotion2vec_load:.1f}s")

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

    results = []
    infer_time = 0.0
    controller_time_total = 0.0
    emotion2vec_time_total = 0.0
    rerank_time_total = 0.0
    wall_t0 = time.time()

    for offset in range(0, len(samples), args.batch_size):
        batch = samples[offset:offset + args.batch_size]
        global_start = args.start + offset
        global_end = global_start + len(batch)
        print(f"[INFO] batch [{global_start}, {global_end})")

        t0 = time.time()
        style_rows = [{} for _ in batch]
        controller_time = 0.0
        emotion2vec_time = 0.0
        if style_controller is not None:
            t_style = time.time()
            style_rows = predict_style_batch(batch, data_root, style_controller, args.style_conf_threshold)
            controller_time = time.time() - t_style
            controller_time_total += controller_time
        elif emotion2vec_model is not None:
            t_style = time.time()
            style_rows = predict_emotion2vec_batch(
                batch,
                data_root,
                emotion2vec_model,
                emotion2vec_tmp_dir,
                args.emotion2vec_conf_threshold,
                args.reference_audio_dir if args.reference_audio_style else "",
            )
            emotion2vec_time = time.time() - t_style
            emotion2vec_time_total += emotion2vec_time
        audio_outputs = []
        rerank_rows = [{} for _ in batch]
        rerank_time = 0.0
        if args.candidate_count > 1:
            translations = []
            final_audio_paths = []
            t_rerank = time.time()
            for idx, sample in enumerate(batch):
                translation, selected_audio, rerank_info = infer_and_rerank_sample(
                    sample=sample,
                    data_root=data_root,
                    processor=processor,
                    model=model,
                    ser_model=emotion2vec_model,
                    max_new_tokens=args.max_new_tokens,
                    speaker=args.speaker,
                    style_row=style_rows[idx],
                    audio_sample_rate=args.audio_sample_rate,
                    candidate_count=args.candidate_count,
                    candidate_audio_dir=candidate_audio_dir,
                    candidate_ser_dir=os.path.join(emotion2vec_tmp_dir, "candidate_ser"),
                    do_sample=args.candidate_do_sample,
                    temperature=args.candidate_temperature,
                    top_p=args.candidate_top_p,
                )
                final_audio = os.path.join(audio_outdir, f"qwen25_omni_{sample.get('id', global_start + idx)}.wav")
                os.makedirs(os.path.dirname(final_audio), exist_ok=True)
                data, sr = sf.read(selected_audio)
                sf.write(final_audio, data, sr)
                translations.append(translation)
                final_audio_paths.append(final_audio)
                rerank_rows[idx] = rerank_info
            rerank_time = time.time() - t_rerank
            rerank_time_total += rerank_time
        else:
            translations, audio_outputs = infer_batch(
                batch=batch,
                data_root=data_root,
                processor=processor,
                model=model,
                max_new_tokens=args.max_new_tokens,
                return_audio=args.return_audio,
                speaker=args.speaker,
                style_rows=style_rows,
            )
            final_audio_paths = [""] * len(batch)
        predicted_emotions = [""] * len(batch)
        if args.predict_emotion:
            predicted_emotions = infer_emotion_batch(
                batch=batch,
                data_root=data_root,
                processor=processor,
                model=model,
                max_new_tokens=args.emotion_max_new_tokens,
            )
        batch_time = time.time() - t0
        infer_time += batch_time

        for idx, (sample, translation) in enumerate(zip(batch, translations)):
            output_audio = ""
            if args.candidate_count > 1:
                output_audio = final_audio_paths[idx]
            elif args.return_audio and idx < len(audio_outputs):
                output_audio = os.path.join(audio_outdir, f"qwen25_omni_{sample.get('id', global_start + idx)}.wav")
                save_audio_output(audio_outputs[idx], output_audio, args.audio_sample_rate)
            results.append({
                "id": sample.get("id", ""),
                "audio": sample.get("audio", ""),
                "reference_en": sample.get("text", ""),
                "emotion": sample.get("emotion", ""),
                "qwen25_omni_predicted_emotion": predicted_emotions[idx] if idx < len(predicted_emotions) else "",
                "controller_emotion": style_rows[idx].get("controller_emotion", "") if idx < len(style_rows) else "",
                "controller_confidence": style_rows[idx].get("controller_confidence", None) if idx < len(style_rows) else None,
                "emotion2vec_emotion": style_rows[idx].get("emotion2vec_emotion", "") if idx < len(style_rows) else "",
                "emotion2vec_raw_label": style_rows[idx].get("emotion2vec_raw_label", "") if idx < len(style_rows) else "",
                "emotion2vec_confidence": style_rows[idx].get("emotion2vec_confidence", None) if idx < len(style_rows) else None,
                "emotion2vec_confidence_threshold": style_rows[idx].get("emotion2vec_confidence_threshold", None) if idx < len(style_rows) else None,
                "reference_audio": style_rows[idx].get("reference_audio", "") if idx < len(style_rows) else "",
                "reference_audio_status": style_rows[idx].get("reference_audio_status", "") if idx < len(style_rows) else "",
                "style_prompt": style_rows[idx].get("style_prompt", "") if idx < len(style_rows) else "",
                "style_source": style_rows[idx].get("style_source", "controller" if args.style_controller else "none") if idx < len(style_rows) else "none",
                "style_confidence_threshold": style_rows[idx].get("style_confidence_threshold", None) if idx < len(style_rows) else None,
                "prosody_features": style_rows[idx].get("prosody_features", {}) if args.save_prosody_features and idx < len(style_rows) else {},
                "rerank": rerank_rows[idx] if idx < len(rerank_rows) else {},
                "speaker": sample.get("speaker", ""),
                "qwen25_omni_translation_zh": translation,
                "qwen25_omni_audio": output_audio,
            })

        summary = {
            "model": args.model,
            "num_samples": len(results),
            "dataset_total_samples": len(dataset),
            "processed_range": [args.start, args.start + len(results)],
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "return_audio": args.return_audio,
            "predict_emotion": args.predict_emotion,
            "emotion_labels": EMOTION_LABELS if args.predict_emotion else None,
            "style_controller": args.style_controller or None,
            "style_conf_threshold": args.style_conf_threshold,
            "emotion2vec_style": args.emotion2vec_style,
            "emotion2vec_model": args.emotion2vec_model if args.emotion2vec_style else None,
            "emotion2vec_conf_threshold": args.emotion2vec_conf_threshold,
            "emotion2vec_tmp_dir": emotion2vec_tmp_dir if args.emotion2vec_style else None,
            "emotion2vec_load_time_sec": round(t_emotion2vec_load, 4),
            "reference_audio_style": args.reference_audio_style,
            "reference_audio_dir": args.reference_audio_dir if args.reference_audio_style else None,
            "disgust_reference_policy": "no mapping; text-only disgust cue" if args.reference_audio_style else None,
            "candidate_count": args.candidate_count,
            "candidate_do_sample": args.candidate_do_sample,
            "candidate_temperature": args.candidate_temperature,
            "candidate_top_p": args.candidate_top_p,
            "candidate_audio_dir": candidate_audio_dir if args.candidate_count > 1 else None,
            "save_prosody_features": args.save_prosody_features,
            "last_batch_controller_time_sec": round(controller_time, 4),
            "controller_time_sec": round(controller_time_total, 4),
            "avg_controller_time_sec_per_sample": round(controller_time_total / len(results), 6) if results else None,
            "last_batch_emotion2vec_time_sec": round(emotion2vec_time, 4),
            "emotion2vec_time_sec": round(emotion2vec_time_total, 4),
            "avg_emotion2vec_time_sec_per_sample": round(emotion2vec_time_total / len(results), 6) if results else None,
            "last_batch_rerank_time_sec": round(rerank_time, 4),
            "rerank_time_sec": round(rerank_time_total, 4),
            "avg_rerank_time_sec_per_sample": round(rerank_time_total / len(results), 6) if results else None,
            "speaker": args.speaker if args.return_audio else None,
            "audio_sample_rate": args.audio_sample_rate if args.return_audio else None,
            "audio_outdir": audio_outdir if args.return_audio else None,
            "talker_disabled": (not args.keep_talker and not args.return_audio),
            "qwen25_omni_time_sec": round(infer_time, 1),
            "qwen25_omni_wall_time_sec": round(time.time() - wall_t0, 1),
            "completed": len(results) == len(samples),
        }
        save_json(results, results_path)
        save_json(summary, summary_path)
        print(f"[INFO] done batch in {batch_time:.1f}s, accumulated {len(results)}/{len(samples)}")
        print(f"[INFO] updated {results_path}")

    print(f"[DONE] saved results: {results_path}")
    print(f"[DONE] saved summary: {summary_path}")


if __name__ == "__main__":
    main()
