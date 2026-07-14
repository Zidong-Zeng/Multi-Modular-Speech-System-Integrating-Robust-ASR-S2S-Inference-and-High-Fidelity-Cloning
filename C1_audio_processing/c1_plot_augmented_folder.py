# -*- coding: utf-8 -*-
"""
C1 增强音频可视化（文件夹直读版）

直接扫描指定目录下的 speed/ volume/ noise/ vad/ 等子文件夹，
为每个 .wav 生成四合一频谱总览图（波形 + 线性谱 + Mel谱 + MFCC）。

用法（直接对标 c1_audio_processing.py）：
    python c1_plot_augmented_folder.py --input /path/to/augmented --outdir /path/to/plots
"""

import os
import glob
import argparse
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False


def draw_overview(y, sr, title, save_path):
    """绘制四合一总览图并保存"""
    # 计算频谱
    S_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=1024, hop_length=256)),
        ref=np.max
    )
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80
    )
    fbank_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_fft=1024, hop_length=256, n_mfcc=13
    )

    fig, axes = plt.subplots(4, 1, figsize=(11, 13))

    # 1. 波形
    t = np.arange(len(y)) / sr
    axes[0].plot(t, y, linewidth=0.6)
    axes[0].set_title(f"{title}  –  Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, t[-1])

    # 2. 线性频谱
    im1 = librosa.display.specshow(
        S_db, sr=sr, hop_length=256, x_axis="time", y_axis="hz",
        ax=axes[1], cmap="magma"
    )
    axes[1].set_title("Linear Spectrogram (STFT, dB)")
    fig.colorbar(im1, ax=axes[1], format="%+2.0f dB")

    # 3. Mel 频谱
    im2 = librosa.display.specshow(
        fbank_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel",
        ax=axes[2], cmap="magma"
    )
    axes[2].set_title("Mel-Spectrogram / FBank (80 mel)")
    fig.colorbar(im2, ax=axes[2], format="%+2.0f dB")

    # 4. MFCC
    im3 = librosa.display.specshow(
        mfcc, sr=sr, hop_length=256, x_axis="time",
        ax=axes[3], cmap="viridis"
    )
    axes[3].set_title("MFCC (13 coeffs)")
    fig.colorbar(im3, ax=axes[3])

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", 
                    default="/root/siton-tmp/assignment_C/C1_audio_processing/outputs/augmented",
                    help="增强音频的根目录（包含 speed/ noise/ volume/ vad/ 等子文件夹）")
    ap.add_argument("--outdir", 
                    default="/root/siton-tmp/assignment_C/C1_audio_processing/outputs/plots_augmented",
                    help="绘图输出目录")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 【核心改动】直接扫描子文件夹，而不是读 JSON
    # 常用的增强子目录名称
    subdirs = ["speed", "volume", "noise", "vad", "clean"]
    all_files = []
    
    for sub in subdirs:
        target_dir = os.path.join(args.input, sub)
        if os.path.exists(target_dir):
            files = glob.glob(os.path.join(target_dir, "*.wav"))
            # 为每个文件附带所属子目录信息，便于做标题
            for f in files:
                all_files.append((f, sub))
    
    # 如果上面没找到，递归搜索所有 .wav（保险起见）
    if not all_files:
        print(f"在 {args.input} 的子文件夹中未找到 .wav，尝试递归搜索...")
        all_files = [(f, "unknown") for f in glob.glob(os.path.join(args.input, "**", "*.wav"), recursive=True)]

    if not all_files:
        print(f"未找到任何音频文件。请检查目录: {args.input}")
        return

    print(f"共发现 {len(all_files)} 个增强音频，开始绘图...")

    success = 0
    for idx, (audio_path, aug_type) in enumerate(all_files, 1):
        base = os.path.splitext(os.path.basename(audio_path))[0]
        # 输出图片名字，直接沿用文件名
        out_png = os.path.join(args.outdir, f"{base}.png")
        
        # 如果图片已存在，可以跳过（可选），这里不跳过，直接覆盖
        try:
            # 加载音频（增强音频已是 16k 单声道）
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # 构造标题：显示文件名 + 类型
            title = f"{base}  [{aug_type}]"
            
            draw_overview(y, sr, title, out_png)
            print(f"[{idx}/{len(all_files)}] 已保存: {out_png}")
            success += 1
        except Exception as e:
            print(f"[{idx}/{len(all_files)}] 失败: {audio_path} | 错误: {e}")

    print(f"✅ 绘图完成！成功绘制 {success} 张，保存在: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()