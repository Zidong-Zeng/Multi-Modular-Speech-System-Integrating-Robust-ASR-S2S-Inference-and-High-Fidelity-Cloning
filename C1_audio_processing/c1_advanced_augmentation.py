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
"""
import os
import glob
import argparse
import json
import subprocess
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
    # 保证信噪比成立
    scale = speech_rms / (noise_rms * (10**(target_snr/20)) + 1e-9)
    noisy_speech = speech + (noise * scale)
    # 峰值归一化，避免爆音
    return noisy_speech / (np.max(np.abs(noisy_speech)) + 1e-9)

def process_one_audio(path, outdir, target_sr, noise_data=None, enable_vad=True, enable_speed=True, enable_volume=True, enable_noise=True):
    stem = os.path.splitext(os.path.basename(path))[0]
    y_orig, sr_orig = librosa.load(path, sr=None, mono=False)
    
    # --- 基础处理：单声道 + 16k 重采样 + 归一化 ---
    y_mono = librosa.to_mono(y_orig) if y_orig.ndim > 1 else y_orig
    y_rs = librosa.resample(y_mono, orig_sr=sr_orig, target_sr=target_sr)
    y_clean = y_rs / (np.max(np.abs(y_rs)) + 1e-9) # 16k, 归一化后的干净语音
    
    clean_path = os.path.join(outdir, f"{stem}_clean.wav")
    sf.write(clean_path, y_clean, target_sr)

    meta_list = [] # 用于记录生成的增强数据元信息

    # 1. VAD 静音裁剪 (将长句切成短片段)
    if enable_vad:
        intervals = librosa.effects.split(y_clean, top_db=25, frame_length=2048, hop_length=512)
        for i, (start, end) in enumerate(intervals):
            segment = y_clean[start:end]
            # VAD 切出来的片断通常比较短，要求音素完整，长度过短则丢弃
            if len(segment) > target_sr * 0.5: # 过滤小于0.5秒的片段
                vad_path = os.path.join(outdir, f"{stem}_vad_seg{i}.wav")
                sf.write(vad_path, segment, target_sr)
                meta_list.append({"id": f"{stem}_vad_seg{i}", "path": vad_path, "aug_type": "vad", "params": f"seg_{i}"})

    # 2. 速度扰动
    if enable_speed:
        for rate in [0.9, 1.1]:
            y_speed = librosa.effects.time_stretch(y_clean, rate=rate)
            spd_path = os.path.join(outdir, f"{stem}_speed_{rate}.wav")
            sf.write(spd_path, y_speed, target_sr)
            meta_list.append({"id": f"{stem}_speed_{rate}", "path": spd_path, "aug_type": "speed", "params": f"rate_{rate}"})

    # 3. 音量扰动 (注意边界，防止失真)
    if enable_volume:
        for gain in [0.5, 1.5]:
            y_vol = np.clip(y_clean * gain, -1.0, 1.0)
            vol_path = os.path.join(outdir, f"{stem}_volume_{gain}.wav")
            sf.write(vol_path, y_vol, target_sr)
            meta_list.append({"id": f"{stem}_volume_{gain}", "path": vol_path, "aug_type": "volume", "params": f"gain_{gain}"})

    # 4. 噪声增强
    if enable_noise and noise_data is not None:
        # 需要找到和 y_clean 一样长的噪声片段
        noise = ensure_wav_length(noise_data, len(y_clean))
        for snr in [5, 10, 15]: # 信噪比越低，噪声越大
            y_noisy = add_noise(y_clean, noise, target_snr=snr)
            noise_path = os.path.join(outdir, f"{stem}_noise_snr{snr}.wav")
            sf.write(noise_path, y_noisy, target_sr)
            meta_list.append({"id": f"{stem}_noise_snr{snr}", "path": noise_path, "aug_type": "noise", "params": f"snr_{snr}"})

    return meta_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/root/siton-tmp/assignment_C/common_data/dataset/cremad/AudioWAV",
                    help="音频文件或目录")
    ap.add_argument("--outdir", default="/root/siton-tmp/assignment_C/C1_audio_processing/outputs/augmented",
                    help="增强输出的目录")
    ap.add_argument("--target_sr", type=int, default=16000,
                    help="目标采样率（建议使用16k，适配ASR模型）")
    # 依赖：你需要准备一段纯噪声（如白噪音或背景噪音）的 wav 文件
    ap.add_argument("--noise_file", default=None, 
                    help="【可选】指定一段噪声WAV文件用于噪声增强，不传则不执行噪声增强")
    # 功能开关
    ap.add_argument("--no-vad", action="store_true", help="禁用VAD切割")
    ap.add_argument("--no-speed", action="store_true", help="禁用速度扰动")
    ap.add_argument("--no-volume", action="store_true", help="禁用音量扰动")
    ap.add_argument("--no-noise", action="store_true", help="禁用噪声增强")
    
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 获取文件列表
    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.wav")))
    else:
        files = [args.input]
    if not files:
        print(f"未找到音频文件: {args.input}"); return

    # 加载噪声文件
    noise_data = None
    if not args.no_noise:
        if args.noise_file and os.path.exists(args.noise_file):
            print(f"正在读取背景噪声样本: {args.noise_file} ...")
            noise_data, _ = librosa.load(args.noise_file, sr=args.target_sr, mono=True)
        else:
            print("⚠️ 警告: 未提供 --noise_file 或文件不存在，噪声增强步骤将被跳过。")
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

    # 生成可复用的流水线元数据文件
    manifest_path = os.path.join(args.outdir, "augmented_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 数据处理完成！共生成 {len(all_meta)} 条增强音频数据。")
    print(f"📁 增强数据保存位置: {os.path.abspath(args.outdir)}")
    print(f"📋 流水线元数据清单: {manifest_path}")

if __name__ == "__main__":
    main()