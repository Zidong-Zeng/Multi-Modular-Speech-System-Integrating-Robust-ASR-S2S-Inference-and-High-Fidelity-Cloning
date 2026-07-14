# -*- coding: utf-8 -*-
"""
C1 语音数据处理进阶任务 —— 数据增强与流水线生成脚本

功能：
1. 音频基础处理（重采样16k, 单声道, 峰值归一化）
2. VAD 静音切割：切出中间的短句片段
3. 速度扰动：0.9倍速 和 1.1倍速
4. 音量扰动：0.5倍音量 和 1.5倍音量
5. 噪声增强：从指定目录读取噪声，按 SNR=5dB, 10dB, 15dB 混合生成
6. 构建可复用流水线：自动生成增强后的音频列表与元数据 JSON 文件

增强输出按类别分目录组织：
  outdir/
    ├── clean/          # 原始干净语音（重采样+归一化）
    ├── vad/            # VAD切割片段
    ├── speed/          # 速度扰动
    ├── volume/         # 音量扰动
    └── noise/          # 加噪增强
    └── augmented_manifest.json   # 元数据清单（含 audio 和 audio_path 绝对路径）
"""
import os
import glob
import argparse
import json
import numpy as np
import librosa
import soundfile as sf

def ensure_wav_length(audio, target_len):
    """循环或截断音频以匹配指定长度"""
    if len(audio) >= target_len:
        return audio[:target_len]
    repeats = int(np.ceil(target_len / len(audio)))
    return np.tile(audio, repeats)[:target_len]

def add_noise(speech, noise, target_snr):
    """
    按目标信噪比(dB)混入噪声
    """
    speech_rms = np.sqrt(np.mean(speech**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    scale = speech_rms / (noise_rms * (10**(target_snr/20)) + 1e-9)
    noisy_speech = speech + (noise * scale)
    return noisy_speech / (np.max(np.abs(noisy_speech)) + 1e-9)

def process_one_audio(path, outdir, target_sr, noise_data=None,
                      enable_vad=True, enable_speed=True,
                      enable_volume=True, enable_noise=True):
    """
    处理单个音频文件，生成各类增强并保存在对应的子目录中。
    返回该文件生成的所有增强样本的元信息列表。
    """
    stem = os.path.splitext(os.path.basename(path))[0]

    # 1. 加载音频
    y_orig, sr_orig = librosa.load(path, sr=None, mono=False)

    # --- 基础处理：单声道 + 16k 重采样 + 归一化 ---
    y_mono = librosa.to_mono(y_orig) if y_orig.ndim > 1 else y_orig
    y_rs = librosa.resample(y_mono, orig_sr=sr_orig, target_sr=target_sr)
    y_clean = y_rs / (np.max(np.abs(y_rs)) + 1e-9)

    # 保存干净语音到 clean/ 子目录（不加入清单，可根据需要自行添加）
    clean_subdir = os.path.join(outdir, "clean")
    os.makedirs(clean_subdir, exist_ok=True)
    clean_path = os.path.join(clean_subdir, f"{stem}_clean.wav")
    sf.write(clean_path, y_clean, target_sr)

    meta_list = []

    # 2. VAD 静音裁剪
    if enable_vad:
        vad_subdir = os.path.join(outdir, "vad")
        os.makedirs(vad_subdir, exist_ok=True)
        intervals = librosa.effects.split(y_clean, top_db=25, frame_length=2048, hop_length=512)
        for i, (start, end) in enumerate(intervals):
            segment = y_clean[start:end]
            if len(segment) > target_sr * 0.5:  # 过滤 <0.5s
                vad_path = os.path.join(vad_subdir, f"{stem}_vad_seg{i}.wav")
                sf.write(vad_path, segment, target_sr)
                # ------------------ 改动点 ------------------
                abs_vad_path = os.path.abspath(vad_path)
                meta_list.append({
                    "id": f"{stem}_vad_seg{i}",
                    "audio": abs_vad_path,        # 新增字段1
                    "audio_path": abs_vad_path,   # 新增字段2（内容相同）
                    "aug_type": "vad",
                    "params": f"seg_{i}"
                })
                # -----------------------------------------

    # 3. 速度扰动
    if enable_speed:
        speed_subdir = os.path.join(outdir, "speed")
        os.makedirs(speed_subdir, exist_ok=True)
        for rate in [0.9, 1.1]:
            y_speed = librosa.effects.time_stretch(y_clean, rate=rate)
            spd_path = os.path.join(speed_subdir, f"{stem}_speed_{rate}.wav")
            sf.write(spd_path, y_speed, target_sr)
            abs_spd_path = os.path.abspath(spd_path)
            meta_list.append({
                "id": f"{stem}_speed_{rate}",
                "audio": abs_spd_path,
                "audio_path": abs_spd_path,
                "aug_type": "speed",
                "params": f"rate_{rate}"
            })

    # 4. 音量扰动
    if enable_volume:
        volume_subdir = os.path.join(outdir, "volume")
        os.makedirs(volume_subdir, exist_ok=True)
        for gain in [0.5, 1.5]:
            y_vol = np.clip(y_clean * gain, -1.0, 1.0)
            vol_path = os.path.join(volume_subdir, f"{stem}_volume_{gain}.wav")
            sf.write(vol_path, y_vol, target_sr)
            abs_vol_path = os.path.abspath(vol_path)
            meta_list.append({
                "id": f"{stem}_volume_{gain}",
                "audio": abs_vol_path,
                "audio_path": abs_vol_path,
                "aug_type": "volume",
                "params": f"gain_{gain}"
            })

    # 5. 噪声增强
    if enable_noise and noise_data is not None:
        noise_subdir = os.path.join(outdir, "noise")
        os.makedirs(noise_subdir, exist_ok=True)
        noise = ensure_wav_length(noise_data, len(y_clean))
        for snr in [5, 10, 15]:
            y_noisy = add_noise(y_clean, noise, target_snr=snr)
            noise_path = os.path.join(noise_subdir, f"{stem}_noise_snr{snr}.wav")
            sf.write(noise_path, y_noisy, target_sr)
            abs_noise_path = os.path.abspath(noise_path)
            meta_list.append({
                "id": f"{stem}_noise_snr{snr}",
                "audio": abs_noise_path,
                "audio_path": abs_noise_path,
                "aug_type": "noise",
                "params": f"snr_{snr}"
            })

    return meta_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/root/siton-tmp/assignment_C/common_data/dataset/cremad/AudioWAV",
                    help="音频文件或目录")
    ap.add_argument("--outdir", default="/root/siton-tmp/assignment_C/C1_audio_processing/outputs/augmented",
                    help="增强输出的根目录（内部自动按类别分文件夹）")
    ap.add_argument("--target_sr", type=int, default=16000,
                    help="目标采样率（建议16k）")
    ap.add_argument("--noise_file", default=None,
                    help="【可选】噪声WAV文件路径，不提供则跳过噪声增强")
    ap.add_argument("--no-vad", action="store_true", help="禁用VAD切割")
    ap.add_argument("--no-speed", action="store_true", help="禁用速度扰动")
    ap.add_argument("--no-volume", action="store_true", help="禁用音量扰动")
    ap.add_argument("--no-noise", action="store_true", help="禁用噪声增强")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 获取输入文件列表
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.wav")))
    else:
        files = [args.input]
    if not files:
        print(f"未找到音频文件: {args.input}")
        return

    # 加载噪声
    noise_data = None
    if not args.no_noise:
        if args.noise_file and os.path.exists(args.noise_file):
            print(f"正在读取背景噪声样本: {args.noise_file} ...")
            noise_data, _ = librosa.load(args.noise_file, sr=args.target_sr, mono=True)
        else:
            print("警告: 未提供 --noise_file 或文件不存在，噪声增强步骤将被跳过。")
            args.no_noise = True

    print(f"共发现 {len(files)} 个音频，开始处理数据增强...")
    all_meta = []

    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        print(f"处理: {stem}")
        try:
            metas = process_one_audio(
                f, args.outdir, args.target_sr,
                noise_data=noise_data,
                enable_vad=not args.no_vad,
                enable_speed=not args.no_speed,
                enable_volume=not args.no_volume,
                enable_noise=not args.no_noise
            )
            all_meta.extend(metas)
        except Exception as e:
            print(f"处理 {stem} 时发生错误: {e}")

    # 生成元数据清单（放在输出根目录）
    manifest_path = os.path.join(args.outdir, "augmented_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=4)

    print(f"数据处理完成！共生成 {len(all_meta)} 条增强音频数据。")
    print(f"增强数据保存位置: {os.path.abspath(args.outdir)}")
    print(f"流水线元数据清单: {manifest_path}")


if __name__ == "__main__":
    main()