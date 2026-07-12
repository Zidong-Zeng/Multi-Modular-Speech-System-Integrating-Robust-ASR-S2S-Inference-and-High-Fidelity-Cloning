# -*- coding: utf-8 -*-
"""
C5_TTS 实验结果可视化脚本
读取 outputs/ 下各模块的JSON汇总文件，生成统计表格和图表
"""
import os, sys, json, time
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
# 自动查找 outputs 目录，按优先级：C5_TTS > CCC > 当前目录
_candidates = [
    os.path.join(HERE, "..", "..", "CCC", "outputs"),       # C5_TTS/code → CCC/outputs (真实数据目录)
    os.path.join(HERE, "..", "..", "C5_TTS", "outputs"),    # CCC/code → C5_TTS/outputs
    os.path.join(HERE, "..", "C5_TTS", "outputs"),           # C5_TTS/code → C5_TTS/outputs
    os.path.join(HERE, "..", "outputs"),                      # {any}/code → {any}/outputs
]
_OUTDIR = None
for c in _candidates:
    if os.path.isdir(c):
        _OUTDIR = os.path.abspath(c)
        break
if _OUTDIR is None:
    _OUTDIR = os.path.join(HERE, "..", "outputs")
OUTPUTS = _OUTDIR
REPORT_DIR = os.path.join(OUTPUTS, "report")
os.makedirs(REPORT_DIR, exist_ok=True)
print(f"[INFO] Data source: {OUTPUTS}")
print(f"[INFO] Report dir:  {REPORT_DIR}")

def load_json(*parts):
    path = os.path.join(OUTPUTS, *parts)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# ============================================================
# 1. 多语种合成统计 (进阶①)
# ============================================================
def chart_multilingual():
    data = load_json("multilingual", "multilingual_summary.json")
    if not data:
        return

    results = data["results"]
    langs = [r["lang"] for r in results]
    durations = [r["duration_sec"] for r in results]
    synth_times = [r["synth_sec"] for r in results]
    rtfs = [r["rtf"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 图1: 音频时长 vs 合成耗时 (双柱)
    x = np.arange(len(langs))
    w = 0.35
    axes[0].bar(x - w/2, durations, w, label='Audio Duration(s)', color='#4CAF50')
    axes[0].bar(x + w/2, synth_times, w, label='Synth Time(s)', color='#2196F3')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(langs, fontsize=12)
    axes[0].set_ylabel('Seconds')
    axes[0].set_title('Multi-Language: Duration vs Synth Time')
    axes[0].legend()

    # 图2: RTF
    colors = ['#FF6B6B' if r > 1.5 else '#4ECDC4' for r in rtfs]
    axes[1].bar(langs, rtfs, color=colors)
    axes[1].axhline(y=1.0, color='red', linestyle='--', label='RTF=1.0 (real-time)')
    axes[1].set_ylabel('RTF')
    axes[1].set_title('Real-Time Factor (<1 = faster than real-time)')
    axes[1].legend()

    # 图3: 文本长度归一化的语速
    text_lens = [len(r["text"]) for r in results]
    char_per_sec = [l/d for l, d in zip(text_lens, durations)]
    axes[2].bar(langs, char_per_sec, color='#9C27B0')
    axes[2].set_ylabel('Chars/Second')
    axes[2].set_title('Speaking Rate (chars/sec)')

    fig.suptitle('Multi-Language TTS Evaluation (CosyVoice2-0.5B)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "chart_multilingual.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] chart_multilingual.png")

    # 表格
    print("\n  Table: Multi-Language Results")
    print(f"  {'Lang':<6} {'Duration(s)':<14} {'Synth(s)':<12} {'RTF':<8} {'Chars/s':<10}")
    print("  " + "-" * 55)
    for i, r in enumerate(results):
        print(f"  {r['lang']:<6} {r['duration_sec']:<14.2f} {r['synth_sec']:<12.2f} {r['rtf']:<8.3f} {char_per_sec[i]:<10.1f}")


# ============================================================
# 2. 语速调节统计 (进阶②)
# ============================================================
def chart_speed():
    data = load_json("speed_timbre", "speed_results.json")
    if not data:
        return

    speeds = [r["speed_value"] for r in data]
    durations = [r["duration_sec"] for r in data]
    labels = [r["speed_name"] for r in data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: 语速 vs 时长
    axes[0].plot(speeds, durations, 'o-', color='#2196F3', linewidth=2, markersize=10)
    for s, d, l in zip(speeds, durations, labels):
        axes[0].annotate(f'{l}\n{d:.1f}s', (s, d), textcoords="offset points",
                         xytext=(0, 15), ha='center', fontsize=9)
    axes[0].set_xlabel('Speed Factor')
    axes[0].set_ylabel('Duration (s)')
    axes[0].set_title('Speed vs Duration')
    axes[0].grid(alpha=0.3)

    # 图2: 理论 vs 实际时长
    normal_dur = durations[list(speeds).index(1.0)]
    expected = [normal_dur / s for s in speeds]
    actual = durations
    x = np.arange(len(speeds))
    w = 0.35
    axes[1].bar(x - w/2, expected, w, label='Expected (1/speed)', color='#90CAF9')
    axes[1].bar(x + w/2, actual, w, label='Actual', color='#1565C0')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Duration (s)')
    axes[1].set_title('Expected vs Actual Duration')
    axes[1].legend()

    fig.suptitle('Speed Control Evaluation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "chart_speed.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] chart_speed.png")

    print("\n  Table: Speed Control")
    print(f"  {'Speed':<12} {'Factor':<8} {'Duration(s)':<14} {'Synth(s)':<12}")
    print("  " + "-" * 50)
    for r in data:
        print(f"  {r['speed_name']:<12} {r['speed_value']:<8.2f} {r['duration_sec']:<14.2f} {r['synth_sec']:<12.2f}")


# ============================================================
# 3. 情感控制声学特征对比 (进阶③)
# ============================================================
def chart_emotion():
    data = load_json("emotion", "emotion_results.json")
    if not data:
        return

    emotions = [r["emotion"] for r in data]
    f0_means = [r["acoustic"]["f0_mean"] for r in data]
    f0_stds = [r["acoustic"]["f0_std"] for r in data]
    rms_energies = [r["acoustic"]["rms_energy"] for r in data]
    durations = [r["acoustic"]["duration_sec"] for r in data]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 图1: F0均值 (情绪越高 F0越高)
    colors_f0 = ['#E91E63' if e in ('angry', 'excited') else
                 '#4CAF50' if e in ('happy', 'gentle') else
                 '#2196F3' if e == 'calm' else
                 '#FF9800' if e == 'sad' else '#9E9E9E' for e in emotions]
    bars = axes[0, 0].bar(range(len(emotions)), f0_means, color=colors_f0)
    axes[0, 0].set_xticks(range(len(emotions)))
    axes[0, 0].set_xticklabels(emotions, fontsize=11)
    axes[0, 0].set_ylabel('F0 Mean (Hz)')
    axes[0, 0].set_title('Pitch (F0 Mean) by Emotion')
    for bar, val in zip(bars, f0_means):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                        f'{val:.0f}', ha='center', fontsize=8)

    # 图2: RMS能量
    bars2 = axes[0, 1].bar(range(len(emotions)), rms_energies, color=colors_f0)
    axes[0, 1].set_xticks(range(len(emotions)))
    axes[0, 1].set_xticklabels(emotions, fontsize=11)
    axes[0, 1].set_ylabel('RMS Energy')
    axes[0, 1].set_title('Energy (RMS) by Emotion')
    for bar, val in zip(bars2, rms_energies):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{val:.3f}', ha='center', fontsize=7)

    # 图3: 时长
    bars3 = axes[1, 0].bar(range(len(emotions)), durations, color=colors_f0)
    axes[1, 0].set_xticks(range(len(emotions)))
    axes[1, 0].set_xticklabels(emotions, fontsize=11)
    axes[1, 0].set_ylabel('Duration (s)')
    axes[1, 0].set_title('Duration by Emotion')
    for bar, val in zip(bars3, durations):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.1f}s', ha='center', fontsize=8)

    # 图4: F0 Mean vs F0 Std 散点 (情绪区分)
    for i, e in enumerate(emotions):
        axes[1, 1].scatter(f0_means[i], f0_stds[i], s=200, color=colors_f0[i],
                          edgecolors='black', linewidth=0.5, zorder=5)
        axes[1, 1].annotate(e, (f0_means[i], f0_stds[i]),
                           textcoords="offset points", xytext=(8, 5), fontsize=9)
    axes[1, 1].set_xlabel('F0 Mean (Hz)')
    axes[1, 1].set_ylabel('F0 Std (Hz)')
    axes[1, 1].set_title('Pitch Mean vs Variation')
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle('Emotion Control — Acoustic Feature Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "chart_emotion.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] chart_emotion.png")

    print("\n  Table: Emotion Acoustic Features")
    print(f"  {'Emotion':<10} {'F0_Mean(Hz)':<14} {'F0_Std':<10} {'RMS':<10} {'Dur(s)':<10}")
    print("  " + "-" * 60)
    for r in data:
        a = r["acoustic"]
        print(f"  {r['emotion']:<10} {a['f0_mean']:<14.1f} {a['f0_std']:<10.1f} "
              f"{a['rms_energy']:<10.4f} {a['duration_sec']:<10.1f}")


# ============================================================
# 4. 情感强度梯度 (进阶③)
# ============================================================
def chart_intensity():
    data = load_json("emotion", "intensity_results.json")
    if not data:
        return

    # 分离 sad 和 happy 强度
    sad_items = [r for r in data if r["emotion"] == "sad"]
    happy_items = [r for r in data if r["emotion"] == "happy"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, items, title, color_map in [
        (axes[0], sad_items, "Sadness Intensity Gradient", plt.cm.Blues),
        (axes[1], happy_items, "Happiness Intensity Gradient", plt.cm.Oranges),
    ]:
        levels = [r["intensity_label"].split("_")[-1] for r in items]
        labels = [r["description"] for r in items]
        f0s = [r["acoustic"]["f0_mean"] for r in items]
        rms_vals = [r["acoustic"]["rms_energy"] for r in items]
        durs = [r["acoustic"]["duration_sec"] for r in items]

        x = np.arange(len(levels))
        w = 0.25

        # 双轴
        ax2 = ax.twinx()
        b1 = ax.bar(x - w, f0s, w, color=[color_map(0.4 + 0.2*i) for i in range(len(levels))],
                     label='F0 Mean (Hz)')
        b2 = ax2.bar(x, rms_vals, w, color=[color_map(0.6 + 0.15*i) for i in range(len(levels))],
                      alpha=0.7, label='RMS Energy')

        ax.set_xticks(x)
        ax.set_xticklabels([f"Lv.{lv}" for lv in levels], fontsize=11)
        ax.set_ylabel('F0 Mean (Hz)', color='blue')
        ax2.set_ylabel('RMS Energy', color='orange')
        ax.set_title(title)

        # 标注时长
        for i, (lv, dur) in enumerate(zip(levels, durs)):
            ax.annotate(f'{dur:.1f}s', (i, f0s[i]), textcoords="offset points",
                        xytext=(0, -18), ha='center', fontsize=8, color='gray')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    fig.suptitle('Emotion Intensity Gradient Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "chart_intensity.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] chart_intensity.png")

    print("\n  Table: Emotion Intensity Gradient")
    print(f"  {'Label':<10} {'Description':<25} {'F0(Hz)':<10} {'RMS':<10} {'Dur(s)':<10}")
    print("  " + "-" * 70)
    for r in data:
        a = r["acoustic"]
        print(f"  {r['intensity_label']:<10} {r['description']:<25} {a['f0_mean']:<10.1f} "
              f"{a['rms_energy']:<10.4f} {a['duration_sec']:<10.1f}")


# ============================================================
# 5. 说话人克隆时长影响 (进阶④)
# ============================================================
def chart_speaker_duration():
    data = load_json("speaker_clone", "duration_results.json")
    if not data:
        return

    results = data["results"]
    durations = [r["duration_sec"] for r in results]
    similarities = [r["similarity"] for r in results]
    synth_times = [r["synth_sec"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = '#2196F3'
    ax1.plot(durations, similarities, 'o-', color=color, linewidth=2, markersize=12)
    ax1.set_xlabel('Reference Audio Duration (seconds)', fontsize=12)
    ax1.set_ylabel('Similarity Score', color=color, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color)
    for d, s in zip(durations, similarities):
        ax1.annotate(f'{s:.4f}', (d, s), textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=10, fontweight='bold', color=color)

    ax2 = ax1.twinx()
    ax2.bar(durations, synth_times, width=0.3, alpha=0.3, color='#FF9800')
    ax2.set_ylabel('Synth Time (s)', color='#FF9800')
    ax2.tick_params(axis='y', labelcolor='#FF9800')

    ax1.set_title('Speaker Clone: Reference Duration vs Similarity', fontsize=13, fontweight='bold')
    ax1.set_xticks(durations)
    ax1.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "chart_speaker_clone.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] chart_speaker_clone.png")

    print("\n  Table: Reference Duration Impact")
    print(f"  {'Ref Dur(s)':<14} {'Similarity':<12} {'Audio(s)':<12} {'Synth(s)':<12}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['duration_sec']:<14} {r['similarity']:<12.4f} {r['audio_sec']:<12.2f} {r['synth_sec']:<12.2f}")


# ============================================================
# 6. 综合仪表盘 (Summary Dashboard)
# ============================================================
def chart_summary_dashboard():
    """生成综合概览图"""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('C5_TTS — CosyVoice2-0.5B 全量实验结果概览', fontsize=16, fontweight='bold')

    # ---- 子图1: 多语种RTF ----
    ax1 = fig.add_subplot(2, 3, 1)
    ml_data = load_json("multilingual", "multilingual_summary.json")
    if ml_data:
        langs = [r["lang"] for r in ml_data["results"]]
        rtfs = [r["rtf"] for r in ml_data["results"]]
        ax1.bar(langs, rtfs, color=['#FF6B6B' if r > 1.5 else '#4ECDC4' for r in rtfs])
        ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
        ax1.set_title('Multilingual RTF')
        ax1.set_ylabel('RTF')

    # ---- 子图2: 语速曲线 ----
    ax2 = fig.add_subplot(2, 3, 2)
    sp_data = load_json("speed_timbre", "speed_results.json")
    if sp_data:
        s_vals = [r["speed_value"] for r in sp_data]
        s_durs = [r["duration_sec"] for r in sp_data]
        ax2.plot(s_vals, s_durs, 'o-', color='#2196F3', linewidth=2, markersize=8)
        ax2.set_title('Speed Control Curve')
        ax2.set_xlabel('Speed Factor')
        ax2.set_ylabel('Duration (s)')
        ax2.grid(alpha=0.3)

    # ---- 子图3: 情感F0分布 ----
    ax3 = fig.add_subplot(2, 3, 3)
    em_data = load_json("emotion", "emotion_results.json")
    if em_data:
        em_labels = [r["emotion"] for r in em_data]
        em_f0 = [r["acoustic"]["f0_mean"] for r in em_data]
        colors = ['#E91E63', '#4CAF50', '#FF9800', '#2196F3', '#9C27B0', '#FF5722', '#607D8B']
        ax3.barh(range(len(em_labels)), em_f0, color=colors[:len(em_labels)])
        ax3.set_yticks(range(len(em_labels)))
        ax3.set_yticklabels(em_labels)
        ax3.set_title('F0 Mean by Emotion')
        ax3.set_xlabel('Hz')

    # ---- 子图4: 说话人克隆相似度 ----
    ax4 = fig.add_subplot(2, 3, 4)
    sc_data = load_json("speaker_clone", "duration_results.json")
    if sc_data:
        sc_durs = [r["duration_sec"] for r in sc_data["results"]]
        sc_sims = [r["similarity"] for r in sc_data["results"]]
        ax4.bar([str(d) + 's' for d in sc_durs], sc_sims, color='#4CAF50')
        ax4.set_title('Clone Similarity by Ref Duration')
        ax4.set_ylabel('Similarity')
        for i, (d, s) in enumerate(zip(sc_durs, sc_sims)):
            ax4.text(i, s + 0.01, f'{s:.3f}', ha='center', fontsize=9, fontweight='bold')

    # ---- 子图5: 多说话人音色对比 (cross_gender_results.json) ----
    ax5 = fig.add_subplot(2, 3, 5)
    cg_data = load_json("speaker_clone", "cross_gender_results.json")
    if cg_data:
        spk_names = [r["speaker"] for r in cg_data[:8]]
        spk_durs = [r["audio_sec"] for r in cg_data[:8]]
        colors_spk = plt.cm.tab10(np.linspace(0, 1, len(spk_names)))
        ax5.barh(range(len(spk_names)), spk_durs, color=colors_spk)
        ax5.set_yticks(range(len(spk_names)))
        ax5.set_yticklabels(spk_names, fontsize=7)
        ax5.set_title('Multi-Speaker Clone Duration')
        ax5.set_xlabel('Audio Duration (s)')
    else:
        ax5.text(0.5, 0.5, 'No cross_gender_results.json', ha='center', va='center',
                transform=ax5.transAxes, fontsize=10, color='gray')

    # ---- 子图6: 实验统计 (自动从已有JSON生成) ----
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = ["=== C5_TTS Experiment Summary ===", f"Model: CosyVoice2-0.5B", ""]

    if ml_data:
        n_lang = len(ml_data['results'])
        avg_rtf = np.mean([r["rtf"] for r in ml_data["results"]])
        summary_text.append(f"[Multilingual] {n_lang} languages")
        summary_text.append(f"  Avg RTF = {avg_rtf:.3f}")

    if em_data:
        n_emo = len(em_data)
        f0s = [r["acoustic"]["f0_mean"] for r in em_data]
        rms_vals = [r["acoustic"]["rms_energy"] for r in em_data]
        durs = [r["acoustic"]["duration_sec"] for r in em_data]
        summary_text.append(f"[Emotion] {n_emo} types")
        summary_text.append(f"  F0 range: {min(f0s):.0f} ~ {max(f0s):.0f} Hz")
        summary_text.append(f"  RMS range: {min(rms_vals):.4f} ~ {max(rms_vals):.4f}")
        summary_text.append(f"  Dur range: {min(durs):.1f} ~ {max(durs):.1f} s")

    if sp_data:
        n_sp = len(sp_data)
        sp_range = f"{sp_data[-1]['speed_value']}x ~ {sp_data[0]['speed_value']}x"
        summary_text.append(f"[Speed] {n_sp} levels ({sp_range})")

    if sc_data:
        n_sc = len(sc_data['results'])
        sims = [r["similarity"] for r in sc_data["results"]]
        summary_text.append(f"[Speaker Clone] {n_sc} duration tests")
        summary_text.append(f"  Similarity: {min(sims):.4f} ~ {max(sims):.4f}")

    if cg_data:
        summary_text.append(f"[Multi-Speaker] {len(cg_data)} speakers tested")

    summary_text.append("")
    summary_text.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")

    for i, line in enumerate(summary_text):
        ax6.text(0.05, 0.97 - i * 0.06, line, transform=ax6.transAxes,
                fontsize=9, fontfamily='monospace', verticalalignment='top')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(REPORT_DIR, "dashboard_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] dashboard_summary.png")


# ============================================================
# 7. 批量合成统计 (基础要求)
# ============================================================
def chart_batch():
    data = load_json("c5_summary.json")
    if not data or not data.get("s2s_items"):
        return

    items = data["s2s_items"]
    if len(items) == 0:
        return

    ids = [it["id"] for it in items]
    audio_secs = [it["audio_sec"] for it in items]
    synth_secs = [it["synth_sec"] for it in items]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ids))
    w = 0.35
    ax.bar(x - w/2, audio_secs, w, label='Audio Duration (s)', color='#4CAF50')
    ax.bar(x + w/2, synth_secs, w, label='Synth Time (s)', color='#FF9800', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha='right')
    ax.set_ylabel('Seconds')
    ax.set_title(f'Batch S2S Synthesis ({data.get("s2s_count", len(items))} items, '
                 f'total {data.get("s2s_total_sec", 0):.1f}s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "chart_batch_s2s.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] chart_batch_s2s.png")

    print(f"\n  Table: Batch S2S Synthesis ({len(items)} items)")
    print(f"  {'ID':<10} {'Audio(s)':<12} {'Synth(s)':<12} {'RTF':<8}")
    print("  " + "-" * 45)
    for it in items:
        rtf = it["synth_sec"] / max(it["audio_sec"], 0.01)
        print(f"  {it['id']:<10} {it['audio_sec']:<12.2f} {it['synth_sec']:<12.2f} {rtf:<8.2f}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("C5_TTS Experiment Results Visualization")
    print(f"Output: {REPORT_DIR}")
    print("=" * 60)

    chart_multilingual()
    chart_speed()
    chart_emotion()
    chart_intensity()
    chart_speaker_duration()
    chart_batch()
    chart_summary_dashboard()

    print(f"\n[OK] All charts saved to: {os.path.abspath(REPORT_DIR)}")
    print("Files generated:")
    for f in sorted(os.listdir(REPORT_DIR)):
        print(f"  - {f}")