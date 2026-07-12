# -*- coding: utf-8 -*-
import csv
import json
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT / "C4_end2end" / "outputs_compare_qwen_seamless"

SEAMLESS_RESULTS = PROJECT / "C4_end2end" / "outputs" / "c4_results.json"
SEAMLESS_SUMMARY = PROJECT / "C4_end2end" / "outputs" / "c4_summary.json"
QWEN_RESULTS = PROJECT / "C4_end2end" / "outputs_qwen25_omni_s2t" / "qwen25_omni_results.json"
QWEN_SUMMARY = PROJECT / "C4_end2end" / "outputs_qwen25_omni_s2t" / "qwen25_omni_summary.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_cell(text, limit=72):
    text = str(text or "").replace("\n", " ").replace("|", "\\|").strip()
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= limit else text[: limit - 1] + "..."


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    seamless = load_json(SEAMLESS_RESULTS)
    qwen = load_json(QWEN_RESULTS)
    seamless_summary = load_json(SEAMLESS_SUMMARY)
    qwen_summary = load_json(QWEN_SUMMARY)

    seamless_by_id = {item.get("id"): item for item in seamless}
    rows = []
    for idx, q_item in enumerate(qwen):
        sample_id = q_item.get("id")
        s_item = seamless_by_id.get(sample_id, {})
        rows.append({
            "index": idx,
            "id": sample_id,
            "emotion": q_item.get("emotion") or s_item.get("emotion", ""),
            "speaker": q_item.get("speaker") or s_item.get("speaker", ""),
            "reference_en": q_item.get("reference_en") or s_item.get("reference_en", ""),
            "seamlessm4t_zh": s_item.get("e2e_translation_zh", ""),
            "qwen25_omni_zh": q_item.get("qwen25_omni_translation_zh", ""),
            "c3_cascade_zh": s_item.get("c3_cascade_translation_zh", ""),
            "audio": q_item.get("audio") or s_item.get("audio", ""),
        })

    csv_path = OUTDIR / "qwen_vs_seamless_full_aligned.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "index", "id", "emotion", "speaker", "reference_en",
            "seamlessm4t_zh", "qwen25_omni_zh", "c3_cascade_zh", "audio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    selected = []
    seen_by_emotion = {}
    for row in rows:
        emotion = row["emotion"] or "unknown"
        if seen_by_emotion.get(emotion, 0) < 2:
            selected.append(row)
            seen_by_emotion[emotion] = seen_by_emotion.get(emotion, 0) + 1
        if len(selected) >= 18:
            break
    if len(selected) < 18:
        used = {row["id"] for row in selected}
        for row in rows:
            if row["id"] not in used:
                selected.append(row)
                used.add(row["id"])
            if len(selected) >= 18:
                break

    examples_path = OUTDIR / "qwen_vs_seamless_examples.md"
    with open(examples_path, "w", encoding="utf-8") as f:
        f.write("# Qwen2.5-Omni vs SeamlessM4T S2T 对比样例\n\n")
        f.write(f"- SeamlessM4T: {seamless_summary.get('num_samples')} 条，completed={seamless_summary.get('completed')}\n")
        f.write(f"- Qwen2.5-Omni 当前结果: {qwen_summary.get('num_samples')} 条，completed={qwen_summary.get('completed')}\n")
        f.write(f"- 当前表格对齐样本数: {len(rows)} 条\n\n")
        f.write("| id | emotion | reference_en | SeamlessM4T 中文 | Qwen2.5-Omni 中文 | C3 级联中文 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in selected:
            f.write(
                "| {id} | {emotion} | {reference_en} | {seamlessm4t_zh} | {qwen25_omni_zh} | {c3_cascade_zh} |\n".format(
                    id=safe_cell(row["id"], 28),
                    emotion=safe_cell(row["emotion"], 10),
                    reference_en=safe_cell(row["reference_en"], 55),
                    seamlessm4t_zh=safe_cell(row["seamlessm4t_zh"], 55),
                    qwen25_omni_zh=safe_cell(row["qwen25_omni_zh"], 55),
                    c3_cascade_zh=safe_cell(row["c3_cascade_zh"], 55),
                )
            )

    qwen_time = qwen_summary.get("qwen25_omni_time_sec")
    qwen_n = qwen_summary.get("num_samples") or len(qwen)
    seamless_time = seamless_summary.get("e2e_time_sec")
    seamless_n = seamless_summary.get("num_samples") or len(seamless)

    summary_path = OUTDIR / "qwen_vs_seamless_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Qwen2.5-Omni vs SeamlessM4T S2T 汇总\n\n")
        f.write("| Model | Samples | Completed | Total inference time (s) | Avg time / sample (s) | Output |\n")
        f.write("|---|---:|---|---:|---:|---|\n")
        seamless_avg = seamless_time / seamless_n if seamless_time and seamless_n else 0.0
        qwen_avg = qwen_time / qwen_n if qwen_time and qwen_n else 0.0
        f.write(f"| SeamlessM4T-v2-large | {seamless_n} | {seamless_summary.get('completed')} | {seamless_time} | {seamless_avg:.4f} | speech -> Chinese text |\n")
        f.write(f"| Qwen2.5-Omni-7B | {qwen_n} | {qwen_summary.get('completed')} | {qwen_time} | {qwen_avg:.4f} | speech -> Chinese text |\n")
        f.write("\n")
        f.write(f"当前 Qwen 结果覆盖 processed_range={qwen_summary.get('processed_range')}，对齐样本数为 {len(rows)}。\n")
        if not qwen_summary.get("completed"):
            f.write(f"如果要完整数据集对比，需要从 index {qwen_n} 继续跑到 7442。\n")

    print(f"aligned_rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"examples_md={examples_path}")
    print(f"summary_md={summary_path}")


if __name__ == "__main__":
    main()
