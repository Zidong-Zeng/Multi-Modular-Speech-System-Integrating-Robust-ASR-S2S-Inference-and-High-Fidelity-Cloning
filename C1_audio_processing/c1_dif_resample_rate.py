# -*- coding: utf-8 -*-
"""
C1 语音数据处理与音频增强 —— 进阶任务：多采样率分析演示脚本

修改内容：
- 遍历 [8000, 16000, 44100] 三种采样率进行处理
- 为不同采样率的结果生成独立的子文件夹
"""
import os
import glob
import argparse
import subprocess

import numpy as np
import librosa
import librosa.display
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))


def banner(title):
    line = "=" * 60
    print(f"\n{line}\n{title}\n{line}")


def probe_with_ffprobe(path):
    cmd = ["ffprobe", "-v", "error",
           "-show_entries", "stream=codec_name,sample_rate,channels:format=duration,bit_rate",
           "-of", "default=noprint_wrappers=1", path]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f"(ffprobe 不可用: {e})"


def process_one(path, outdir, target_sr, log):
    """处理单个音频文件，返回处理后的单声道波形与采样率（供下游使用）。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    # 为文件名加入采样率后缀，避免子文件夹内的文件相互覆盖
    sr_suffix = f"_{target_sr}Hz"
    
    banner(f"==== 处理音频: {os.path.basename(path)} [目标采样率: {target_sr} Hz] ====")

    # 【1】统计采样率、时长、通道数
    log(f"-- ffprobe 原始元信息 ({stem}) --")
    log(probe_with_ffprobe(path))
    y_orig, sr_orig = librosa.load(path, sr=None, mono=False)
    n_channels = 1 if y_orig.ndim == 1 else y_orig.shape[0]
    n_samples = y_orig.shape[-1]
    log(f"原始采样率 {sr_orig} Hz | 声道 {n_channels} | 采样点 {n_samples} | 时长 {n_samples/sr_orig:.3f}s")

    # 【2】立体声→单声道 + 重采样 + 静音裁剪 + 峰值归一化
    y_mono = librosa.to_mono(y_orig) if n_channels > 1 else y_orig
    y_rs = librosa.resample(y_mono, orig_sr=sr_orig, target_sr=target_sr)
    y_trim, _ = librosa.effects.trim(y_rs, top_db=30)
    y_norm = y_trim / (np.max(np.abs(y_trim)) + 1e-9)
    
    log(f"处理摘要: {n_channels}ch->1ch, {sr_orig}->{target_sr}Hz, 裁剪后大小 {y_trim.shape[-1]} 采样点, 峰值归一化到 1.0")
    
    out_wav = os.path.join(outdir, f"{stem}{sr_suffix}_processed.wav")
    sf.write(out_wav, y_norm, target_sr)

    # 【3】格式转换 wav -> mp3 / flac
    for ext in ("mp3", "flac"):
        tgt = os.path.join(outdir, f"{stem}{sr_suffix}_processed.{ext}")
        subprocess.run(["ffmpeg", "-y", "-i", out_wav, tgt], capture_output=True, text=True)
    log(f"格式转换: 已生成 {stem}{sr_suffix}_processed.mp3 / .flac")

    # 【4&5】波形 / 线性频谱 / Mel(FBank) / MFCC
    y, sr = y_norm, target_sr
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    fbank_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=1024, hop_length=256, n_mfcc=13)
    log(f"特征形状: 线性频谱 {S_db.shape} | Mel/FBank {fbank_db.shape} | MFCC {mfcc.shape}")

    fig, axes = plt.subplots(4, 1, figsize=(11, 13))
    t = np.arange(len(y)) / sr
    axes[0].plot(t, y, linewidth=0.6); axes[0].set_title(f"{stem}  1. Waveform ({sr} Hz)")
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Amplitude"); axes[0].set_xlim(0, t[-1])
    
    im1 = librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=axes[1], cmap="magma")
    axes[1].set_title("2. Linear Spectrogram (STFT, dB)"); fig.colorbar(im1, ax=axes[1], format="%+2.0f dB")
    
    im2 = librosa.display.specshow(fbank_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", ax=axes[2], cmap="magma")
    axes[2].set_title("3. Mel-Spectrogram / FBank (80 mel)"); fig.colorbar(im2, ax=axes[2], format="%+2.0f dB")
    
    im3 = librosa.display.specshow(mfcc, sr=sr, hop_length=256, x_axis="time", ax=axes[3], cmap="viridis")
    axes[3].set_title("4. MFCC (13 coeffs)"); fig.colorbar(im3, ax=axes[3])
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{stem}{sr_suffix}_overview.png"), dpi=130)
    plt.close(fig)
    log(f"已保存总览图: {stem}{sr_suffix}_overview.png\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/root/siton-tmp/assignment_C/common_data/dataset/cremad/AudioWAV",
                    help="音频文件或目录")
    ap.add_argument("--outdir", default="/root/siton-tmp/assignment_C/C1_audio_processing/outputs")
    args = ap.parse_args()

    # 【修改点】定义待处理的采样率列表（差异较大，便于对比分析）
    # 如果当前默认的 16000 结果已存在，可以将其移除，例如只保留 [8000, 44100]
    sample_rates = [8000, 44100] 

    # 获取所有音频文件
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.wav")))
    else:
        files = [args.input]
    
    if not files:
        print(f"未找到音频文件: {args.input}"); return

    print(f"数据路径: {os.path.abspath(args.input)}")
    print(f"共发现 {len(files)} 个音频。")
    
    # 遍历每一个采样率进行批量处理
    for sr in sample_rates:
        print(f"\n>>> 开始统一处理为 {sr} Hz 采样率的数据...")
        
        # 【修改点】为不同的采样率建立独立的输出子目录
        sr_outdir = os.path.join(args.outdir, f"sr_{sr}")
        os.makedirs(sr_outdir, exist_ok=True)
        
        report_lines = []
        log = lambda m="": (print(m), report_lines.append(str(m)))
        log(f"输出路径: {sr_outdir}")

        for f in files:
            # 调用处理函数，传递特定的采样率参数
            process_one(f, sr_outdir, sr, log)

        # 保存不同采样率的处理报告
        with open(os.path.join(sr_outdir, "C1_report.txt"), "w", encoding="utf-8") as fp:
            fp.write("\n".join(report_lines))
        print(f"--- 完成 {sr} Hz 采样率处理，报告已保存至: {sr_outdir} ---")

    banner("所有采样率处理完成")
    print(f"全部结果保存在: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()