# -*- coding: utf-8 -*-
"""Analyze VAD outputs from different thresholds and draw report figures."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_threshold_outputs(pattern: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        threshold = float(data.get("params", {}).get("threshold"))
        items = data.get("items", [])
        for item in items:
            summary = item.get("summary", {})
            audio_ms = int(summary.get("audio_ms", 0) or 0)
            speech_ms = int(summary.get("speech_ms", 0) or 0)
            ratio = speech_ms / audio_ms if audio_ms else 0.0
            rows.append(
                {
                    "threshold": threshold,
                    "id": item.get("id"),
                    "emotion": item.get("metadata", {}).get("emotion"),
                    "num_segments": int(summary.get("num_segments", 0) or 0),
                    "audio_ms": audio_ms,
                    "speech_ms": speech_ms,
                    "speech_ratio": ratio,
                }
            )
        summary_rows.append(summarize_threshold(threshold, items))
    if not rows:
        raise ValueError(f"No VAD dataset outputs matched pattern: {pattern}")
    return pd.DataFrame(rows), pd.DataFrame(summary_rows).sort_values("threshold")


def summarize_threshold(threshold: float, items: list[dict]) -> dict:
    ratios = []
    segment_counts = []
    for item in items:
        summary = item.get("summary", {})
        audio_ms = int(summary.get("audio_ms", 0) or 0)
        speech_ms = int(summary.get("speech_ms", 0) or 0)
        ratios.append(speech_ms / audio_ms if audio_ms else 0.0)
        segment_counts.append(int(summary.get("num_segments", 0) or 0))
    count = len(items)
    total_audio_ms = sum(int(item.get("summary", {}).get("audio_ms", 0) or 0) for item in items)
    total_speech_ms = sum(int(item.get("summary", {}).get("speech_ms", 0) or 0) for item in items)
    return {
        "threshold": threshold,
        "num_items": count,
        "total_audio_h": total_audio_ms / 3600000,
        "total_speech_h": total_speech_ms / 3600000,
        "mean_speech_ratio": sum(ratios) / count if count else 0.0,
        "median_speech_ratio": pd.Series(ratios).median() if ratios else 0.0,
        "zero_segment_count": sum(1 for c in segment_counts if c == 0),
        "low_coverage_count": sum(1 for r in ratios if r < 0.2),
        "full_coverage_count": sum(1 for r in ratios if r >= 0.98),
        "multi_segment_count": sum(1 for c in segment_counts if c >= 2),
    }


def write_summary_csv(summary_df: pd.DataFrame, outdir: str) -> str:
    path = os.path.join(outdir, "vad_threshold_summary.csv")
    summary_df.to_csv(path, index=False)
    return path


def plot_threshold_trends(summary_df: pd.DataFrame, outdir: str) -> str:
    fig, ax1 = plt.subplots(figsize=(10, 5.6))
    x = summary_df["threshold"].astype(str)
    ax1.plot(x, summary_df["mean_speech_ratio"], marker="o", linewidth=2.2, label="Mean speech/audio ratio")
    ax1.plot(x, summary_df["median_speech_ratio"], marker="s", linewidth=2.2, label="Median speech/audio ratio")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Speech / audio ratio")
    ax1.set_xlabel("VAD threshold")
    ax1.grid(True, axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, summary_df["low_coverage_count"], marker="^", color="#d97706", linewidth=2, label="Low coverage (<0.2)")
    ax2.plot(x, summary_df["zero_segment_count"], marker="x", color="#dc2626", linewidth=2, label="Zero segment")
    ax2.set_ylabel("Risk sample count")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    ax1.set_title("VAD threshold sensitivity on CREMA-D")
    fig.tight_layout()
    path = os.path.join(outdir, "01_threshold_trends.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_segment_distribution(df: pd.DataFrame, outdir: str) -> str:
    counts = (
        df.assign(segment_bucket=df["num_segments"].clip(upper=4).astype(int).astype(str))
        .replace({"segment_bucket": {"4": "4+"}})
        .groupby(["threshold", "segment_bucket"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    for col in ["0", "1", "2", "3", "4+"]:
        if col not in counts:
            counts[col] = 0
    counts = counts[["0", "1", "2", "3", "4+"]]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    counts.plot(kind="bar", stacked=True, ax=ax, color=["#dc2626", "#2563eb", "#14b8a6", "#f59e0b", "#7c3aed"])
    ax.set_title("Detected segment count distribution by threshold")
    ax.set_xlabel("VAD threshold")
    ax.set_ylabel("Number of utterances")
    ax.legend(title="Segments", ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    path = os.path.join(outdir, "02_segment_count_distribution.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_ratio_boxplot(df: pd.DataFrame, outdir: str) -> str:
    thresholds = sorted(df["threshold"].unique())
    data = [df[df["threshold"] == t]["speech_ratio"].values for t in thresholds]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    box = ax.boxplot(data, tick_labels=[str(t) for t in thresholds], showfliers=False, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#dbeafe")
    ax.axhspan(0.2, 0.98, color="#dcfce7", alpha=0.25, label="Reasonable operating band")
    ax.set_title("Speech coverage distribution by threshold")
    ax.set_xlabel("VAD threshold")
    ax.set_ylabel("Speech / audio ratio")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    path = os.path.join(outdir, "03_speech_ratio_boxplot.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_ratio_histograms(df: pd.DataFrame, outdir: str) -> str:
    thresholds = sorted(df["threshold"].unique())
    ncols = min(2, len(thresholds))
    nrows = (len(thresholds) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.6 * nrows), sharex=True, sharey=True)
    axes = list(np.asarray(axes).ravel())
    bins = [i / 20 for i in range(21)]
    for ax, threshold in zip(axes, thresholds):
        subset = df[df["threshold"] == threshold]
        ax.hist(subset["speech_ratio"], bins=bins, color="#2563eb", alpha=0.78, edgecolor="white")
        ax.set_title(f"threshold={threshold}")
        ax.grid(True, axis="y", alpha=0.2)
    for ax in axes[len(thresholds) :]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel("Speech / audio ratio")
    for ax in axes[::ncols]:
        ax.set_ylabel("Utterance count")
    fig.suptitle("Speech coverage histogram under different thresholds", y=0.98)
    fig.tight_layout()
    path = os.path.join(outdir, "04_speech_ratio_histograms.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_markdown_report(summary_df: pd.DataFrame, figures: list[str], outdir: str) -> str:
    best = choose_recommended_threshold(summary_df)
    path = os.path.join(outdir, "vad_threshold_analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# VAD Threshold Analysis\n\n")
        f.write("## Summary Table\n\n")
        f.write(dataframe_to_markdown(summary_df))
        f.write("\n\n")
        f.write(f"Recommended threshold: **{best}**\n\n")
        f.write("Reason: it balances low miss risk and moderate silence trimming on CREMA-D. ")
        f.write("Formal VAD selection still requires frame-level labels and AUC/DCF.\n\n")
        f.write("## Figures\n\n")
        for figure in figures:
            f.write(f"![{Path(figure).stem}]({Path(figure).name})\n\n")
    return path


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Write a small GitHub-style table without optional pandas dependencies."""
    columns = list(df.columns)
    formatted_rows = [[format_markdown_value(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [
        max(len(str(column)), *(len(row[index]) for row in formatted_rows))
        for index, column in enumerate(columns)
    ]
    header = "| " + " | ".join(str(column).ljust(widths[index]) for index, column in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in formatted_rows
    ]
    return "\n".join([header, divider, *body])


def format_markdown_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def choose_recommended_threshold(summary_df: pd.DataFrame) -> float:
    candidates = summary_df.copy()
    candidates["score"] = (
        (candidates["mean_speech_ratio"] - 0.85).abs()
        + candidates["zero_segment_count"] / candidates["num_items"]
        + candidates["low_coverage_count"] / candidates["num_items"]
        + 0.2 * candidates["full_coverage_count"] / candidates["num_items"]
    )
    return float(candidates.sort_values("score").iloc[0]["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw threshold analysis figures from VAD dataset JSON outputs")
    parser.add_argument("--pattern", default=os.path.join("..", "outputs", "vad_stage1_sample_t*.json"))
    parser.add_argument("--outdir", default=os.path.join("..", "outputs", "vad_threshold_analysis"))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df, summary_df = load_threshold_outputs(args.pattern)
    csv_path = write_summary_csv(summary_df, args.outdir)
    figures = [
        plot_threshold_trends(summary_df, args.outdir),
        plot_segment_distribution(df, args.outdir),
        plot_ratio_boxplot(df, args.outdir),
        plot_ratio_histograms(df, args.outdir),
    ]
    report_path = write_markdown_report(summary_df, figures, args.outdir)
    print(f"Wrote summary: {csv_path}")
    print(f"Wrote report: {report_path}")
    for figure in figures:
        print(f"Wrote figure: {figure}")


if __name__ == "__main__":
    main()
