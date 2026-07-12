# -*- coding: utf-8 -*-
"""Evaluate emotion preservation for generated S2S audio.

Two automatic signals are supported:
  1) SER classification metrics, using either a sklearn bundle or emotion2vec.
  2) Prosody distance between source audio and generated audio.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from prosody_utils import EMOTION_LABELS, FEATURE_NAMES, extract_prosody_features, resolve_audio_path, vectorize_features


HERE = os.path.dirname(os.path.abspath(__file__))
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
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_path(path, *roots):
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    for root in roots:
        candidate = os.path.join(root, path)
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(os.path.join(roots[0], path)) if roots else os.path.abspath(path)


def load_sklearn_ser(path):
    bundle = joblib.load(path)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    feature_names = bundle.get("feature_names", FEATURE_NAMES) if isinstance(bundle, dict) else FEATURE_NAMES
    return model, feature_names


def predict_with_sklearn(model, feature_names, audio_file, sample_rate):
    features = extract_prosody_features(audio_file, target_sr=sample_rate)
    X = np.asarray([vectorize_features(features, feature_names)], dtype=np.float32)
    pred = str(model.predict(X)[0])
    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = float(np.max(probs))
    return pred, confidence, features


def load_emotion2vec(model_path):
    from funasr import AutoModel

    return AutoModel(model=model_path, disable_update=True)


def normalize_ser_label(label):
    label = str(label or "").strip().lower()
    if "/" in label:
        label = label.rsplit("/", 1)[-1].strip()
    if label in {"<unk>", "unk"}:
        label = "unknown"
    return EMOTION2VEC_LABEL_MAP.get(label, label)


def pick_emotion2vec_prediction(raw):
    """Return raw_label, mapped_label, confidence from FunASR/emotion2vec output."""
    first = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(first, dict):
        return "", "", None

    labels = first.get("labels") or first.get("label")
    scores = first.get("scores") or first.get("score")

    if isinstance(labels, list) and labels:
        if isinstance(scores, list) and len(scores) == len(labels):
            best_idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
            raw_label = labels[best_idx]
            confidence = float(scores[best_idx])
        else:
            raw_label = labels[0]
            confidence = None
        return str(raw_label), normalize_ser_label(raw_label), confidence

    if isinstance(labels, str):
        confidence = None
        if isinstance(scores, (int, float)):
            confidence = float(scores)
        return labels, normalize_ser_label(labels), confidence

    # Some versions use an "emotion" field or directly store score dictionaries.
    for key in ("emotion", "pred", "prediction"):
        if key in first:
            raw_label = first[key]
            return str(raw_label), normalize_ser_label(raw_label), None

    if isinstance(scores, dict) and scores:
        raw_label, confidence = max(scores.items(), key=lambda kv: float(kv[1]))
        return str(raw_label), normalize_ser_label(raw_label), float(confidence)

    return "", "", None


def predict_with_emotion2vec(model, audio_file, output_dir):
    raw = model.generate(
        input=audio_file,
        output_dir=output_dir,
        granularity="utterance",
        extract_embedding=False,
    )
    raw_label, mapped_label, confidence = pick_emotion2vec_prediction(raw)
    return mapped_label, confidence, raw_label, raw


def relative_abs_diff(a, b):
    a = float(a or 0.0)
    b = float(b or 0.0)
    return abs(a - b) / (abs(a) + 1e-6)


def prosody_distance(src_features, out_features):
    keys = ["pitch_mean", "pitch_range", "rms_mean", "rms_std", "pause_ratio", "onset_rate", "duration"]
    parts = {key: relative_abs_diff(src_features.get(key, 0.0), out_features.get(key, 0.0)) for key in keys}
    return float(np.mean(list(parts.values()))), parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(HERE, "..", "common_data", "dataset.json"))
    parser.add_argument("--results", required=True, help="qwen25_omni_results.json with generated audio paths.")
    parser.add_argument("--out", default="", help="Output evaluation JSON. Defaults to result dir/emotion_eval.json.")
    parser.add_argument("--ser_backend", choices=["none", "sklearn", "emotion2vec"], default="none")
    parser.add_argument("--sklearn_ser", default="", help="Path to a separate sklearn SER model bundle.")
    parser.add_argument("--ser_model", default="", help="SER model path/name. For emotion2vec, pass local model/emotion2vec_plus_large.")
    parser.add_argument("--ser_tmp_dir", default="", help="Temporary output dir for SER backend outputs.")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()
    if args.sklearn_ser and args.ser_backend == "none":
        args.ser_backend = "sklearn"
    if args.ser_backend == "sklearn" and not args.sklearn_ser:
        raise SystemExit("--ser_backend sklearn requires --sklearn_ser")
    if args.ser_backend == "emotion2vec" and not args.ser_model:
        raise SystemExit("--ser_backend emotion2vec requires --ser_model")

    dataset = load_json(args.dataset)
    dataset_by_id = {item.get("id"): item for item in dataset}
    data_root = os.path.dirname(os.path.abspath(args.dataset))
    result_root = os.path.dirname(os.path.abspath(args.results))
    results = load_json(args.results)

    ser_model = None
    ser_features = FEATURE_NAMES
    ser_tmp_dir = args.ser_tmp_dir or os.path.join(result_root, "ser_eval_tmp")
    if args.ser_backend == "sklearn":
        ser_model, ser_features = load_sklearn_ser(args.sklearn_ser)
    elif args.ser_backend == "emotion2vec":
        ser_model = load_emotion2vec(args.ser_model)

    rows = []
    labels = []
    ser_preds = []
    distances = []
    skipped = []

    for item in results:
        sid = item.get("id", "")
        sample = dataset_by_id.get(sid, item)
        label = sample.get("emotion") or item.get("emotion", "")
        output_audio = item.get("qwen25_omni_audio") or item.get("audio_output") or ""
        output_audio = resolve_path(output_audio, result_root, os.getcwd())
        if not output_audio or not os.path.exists(output_audio):
            skipped.append({"id": sid, "reason": "missing output audio", "path": output_audio})
            continue

        src_audio = resolve_audio_path(data_root, sample)
        src_features = extract_prosody_features(src_audio, target_sr=args.sample_rate)
        out_features = extract_prosody_features(output_audio, target_sr=args.sample_rate)
        dist, dist_parts = prosody_distance(src_features, out_features)
        distances.append(dist)

        ser_pred = ""
        ser_conf = None
        ser_raw_label = ""
        ser_raw_output = None
        if ser_model is not None:
            if args.ser_backend == "sklearn":
                ser_pred, ser_conf, _ = predict_with_sklearn(ser_model, ser_features, output_audio, args.sample_rate)
                ser_raw_label = ser_pred
            elif args.ser_backend == "emotion2vec":
                ser_pred, ser_conf, ser_raw_label, ser_raw_output = predict_with_emotion2vec(
                    ser_model,
                    output_audio,
                    ser_tmp_dir,
                )
            if label in EMOTION_LABELS:
                labels.append(label)
                ser_preds.append(ser_pred if ser_pred in EMOTION_LABELS else "__other__")

        rows.append({
            "id": sid,
            "label_emotion": label,
            "generated_audio": output_audio,
            "ser_predicted_emotion": ser_pred,
            "ser_raw_label": ser_raw_label,
            "ser_confidence": ser_conf,
            "ser_backend": args.ser_backend,
            "ser_raw_output": ser_raw_output,
            "prosody_distance": round(dist, 6),
            "prosody_distance_parts": {k: round(v, 6) for k, v in dist_parts.items()},
        })

    summary = {
        "results": os.path.abspath(args.results),
        "num_results": len(results),
        "num_evaluated_audio": len(rows),
        "num_skipped": len(skipped),
        "label_counts": dict(Counter(labels)),
        "ser_backend": args.ser_backend,
        "ser_model": os.path.abspath(args.sklearn_ser) if args.ser_backend == "sklearn" else args.ser_model or None,
        "ser_prediction_counts": dict(Counter(ser_preds)),
        "prosody_distance_mean": round(float(np.mean(distances)), 6) if distances else None,
        "prosody_distance_median": round(float(np.median(distances)), 6) if distances else None,
        "skipped": skipped[:20],
    }

    if labels and ser_preds:
        summary.update({
            "emotion_accuracy": round(float(accuracy_score(labels, ser_preds)), 4),
            "emotion_macro_f1": round(float(f1_score(labels, ser_preds, labels=EMOTION_LABELS, average="macro", zero_division=0)), 4),
            "emotion_uar": round(float(balanced_accuracy_score(labels, ser_preds)), 4),
            "emotion_labels": EMOTION_LABELS,
            "emotion_confusion_matrix": confusion_matrix(labels, ser_preds, labels=EMOTION_LABELS).tolist(),
        })

    out = args.out or os.path.join(result_root, "emotion_preservation_eval.json")
    save_json({"summary": summary, "items": rows}, out)
    print(f"[DONE] saved {out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
