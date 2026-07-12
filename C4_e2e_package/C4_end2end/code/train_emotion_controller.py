# -*- coding: utf-8 -*-
"""Train a lightweight emotion/prosody controller for Qwen S2S style prompting."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from prosody_utils import EMOTION_LABELS, FEATURE_NAMES, STYLE_PROMPTS, vectorize_features


HERE = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_model(kind: str, seed: int):
    if kind == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=400, early_stopping=True, random_state=seed)),
        ])
    if kind == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ])
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown model kind: {kind}")


def split_indices(y, groups, test_size: float, seed: int):
    groups = np.asarray(groups)
    n_classes = len(set(y.tolist()))
    min_test_size = min(0.5, max(test_size, (n_classes + 1) / max(len(y), 1)))
    test_size = min_test_size
    if len(set(groups)) > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(splitter.split(np.zeros_like(y), y, groups))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.zeros_like(y), y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=os.path.join(HERE, "..", "C4_end2end", "emotion_controller", "prosody_features.json"))
    parser.add_argument("--outdir", default=os.path.join(HERE, "..", "C4_end2end", "emotion_controller"))
    parser.add_argument("--model_kind", choices=["rf", "mlp", "logreg"], default="rf")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_json(args.features)
    items = data["items"] if isinstance(data, dict) and "items" in data else data
    usable = [item for item in items if item.get("emotion") in EMOTION_LABELS and item.get("features")]
    if len(usable) < 10:
        raise SystemExit(f"not enough usable samples: {len(usable)}")

    X = np.asarray([vectorize_features(item["features"], FEATURE_NAMES) for item in usable], dtype=np.float32)
    y = np.asarray([item["emotion"] for item in usable])
    groups = [item.get("speaker") or item.get("id", "").split("_")[0] or str(i) for i, item in enumerate(usable)]

    train_idx, test_idx = split_indices(y, groups, args.test_size, args.seed)
    model = make_model(args.model_kind, args.seed)

    t0 = time.time()
    model.fit(X[train_idx], y[train_idx])
    train_sec = time.time() - t0

    pred = model.predict(X[test_idx])
    labels = EMOTION_LABELS
    metrics = {
        "features": os.path.abspath(args.features),
        "model_kind": args.model_kind,
        "num_samples": int(len(usable)),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "test_size": args.test_size,
        "seed": args.seed,
        "train_sec": round(train_sec, 2),
        "label_counts": dict(Counter(y.tolist())),
        "accuracy": round(float(accuracy_score(y[test_idx], pred)), 4),
        "macro_f1": round(float(f1_score(y[test_idx], pred, average="macro", labels=labels, zero_division=0)), 4),
        "uar": round(float(balanced_accuracy_score(y[test_idx], pred)), 4),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y[test_idx], pred, labels=labels).tolist(),
        "classification_report": classification_report(y[test_idx], pred, labels=labels, zero_division=0, output_dict=True),
    }

    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, "emotion_controller.pkl")
    config_path = os.path.join(args.outdir, "emotion_controller_config.json")
    metrics_path = os.path.join(args.outdir, "emotion_controller_metrics.json")
    joblib.dump({
        "model": model,
        "feature_names": FEATURE_NAMES,
        "labels": labels,
        "style_prompts": STYLE_PROMPTS,
        "model_kind": args.model_kind,
    }, model_path)
    save_json({
        "model_path": model_path,
        "feature_names": FEATURE_NAMES,
        "labels": labels,
        "style_prompts": STYLE_PROMPTS,
    }, config_path)
    save_json(metrics, metrics_path)

    print(f"[DONE] saved model: {model_path}")
    print(f"[DONE] saved metrics: {metrics_path}")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "macro_f1", "uar", "num_train", "num_test"]}, indent=2))


if __name__ == "__main__":
    main()
