# -*- coding: utf-8 -*-
"""
进阶③ 情感控制能力验证
通过 CosyVoice2 的 inference_instruct2 接口，用自然语言描述情感
支持7种情感 + 强度梯度 + 声学特征分析 (F0/语速/能量)
"""
import os, sys, json, time
import torch
import torchaudio
import numpy as np

# 7种情感指令预设
EMOTION_INSTRUCTIONS = {
    "happy":   "用开心愉悦的语气朗读，充满快乐和活力，语调上扬，声音明亮",
    "sad":     "用悲伤低沉的语气朗读，语速缓慢，声音沉重压抑，充满哀愁",
    "angry":   "用严肃愤怒的语气朗读，语气坚定有力，语速偏快，声音高亢",
    "gentle":  "用温柔亲切的语气朗读，像朋友聊天一样娓娓道来，声音柔和",
    "calm":    "用平静中性的语气朗读，客观冷静地播报，声音平稳",
    "excited": "用激动兴奋的语气朗读，充满热情，语速快，声音洪亮",
    "tired":   "用慵懒疲惫的语气朗读，语速慢，声音松弛无力，有气无力",
}

# 情感强度梯度 (以sad为例)
EMOTION_INTENSITY = {
    "sad": [
        ("sad_1", "略带一丝忧伤，语气稍微低沉"),
        ("sad_2", "有些难过，声音明显沉重"),
        ("sad_3", "非常悲伤，语速缓慢，充满哀愁"),
        ("sad_4", "极度悲痛，几乎要哭泣，声音颤抖"),
    ],
    "happy": [
        ("happy_1", "微微开心，语气轻快"),
        ("happy_2", "比较高兴，笑容洋溢"),
        ("happy_3", "非常开心，兴高采烈"),
        ("happy_4", "极度兴奋，欢呼雀跃"),
    ],
}


class EmotionController:
    """情感控制器 - 基于inference_instruct2"""

    def __init__(self, cosyvoice_model):
        self.model = cosyvoice_model
        self.sr = cosyvoice_model.sample_rate

    def synthesize_emotion(self, text, emotion, prompt_wav, intensity=None):
        """合成指定情感的语音

        Args:
            text: 目标文本
            emotion: 情感名称 (happy/sad/angry/gentle/calm/excited/tired)
            prompt_wav: 参考音频路径
            intensity: 情感强度标签 (如 sad_2)，None则使用默认强度
        """
        if intensity and emotion in EMOTION_INTENSITY:
            # 查找对应强度
            instruct = None
            for name, desc in EMOTION_INTENSITY.get(emotion, []):
                if name == intensity:
                    instruct = desc
                    break
            if instruct is None:
                instruct = EMOTION_INSTRUCTIONS.get(emotion, EMOTION_INSTRUCTIONS["calm"])
        else:
            instruct = EMOTION_INSTRUCTIONS.get(emotion, EMOTION_INSTRUCTIONS["calm"])

        full_instruct = instruct + "<|endofprompt|>"
        gen = self.model.inference_instruct2(
            text, full_instruct, prompt_wav, stream=False
        )
        chunks = [j["tts_speech"] for j in gen]
        audio = torch.cat(chunks, dim=1)
        return audio


def analyze_acoustic_features(audio_np, sr):
    """分析声学特征: F0, 语速(时长), 能量(RMS)"""
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()

    features = {
        "duration_sec": len(audio_np) / sr,
        "rms_energy": float(np.sqrt(np.mean(audio_np ** 2))),
        "peak_amplitude": float(np.max(np.abs(audio_np))),
    }

    # F0分析 (可选，需要pyworld)
    try:
        import pyworld as pw
        audio_f64 = audio_np.astype(np.float64)
        f0, t = pw.harvest(audio_f64, sr)
        f0_valid = f0[f0 > 0]
        if len(f0_valid) > 0:
            features["f0_mean"] = float(np.mean(f0_valid))
            features["f0_std"] = float(np.std(f0_valid))
            features["f0_min"] = float(np.min(f0_valid))
            features["f0_max"] = float(np.max(f0_valid))
            features["f0_range"] = features["f0_max"] - features["f0_min"]
        else:
            features["f0_mean"] = 0
            features["f0_std"] = 0
    except ImportError:
        features["f0_mean"] = -1

    return features


def run_emotion_test(cosyvoice_model, prompt_wav, 
                      output_dir="./outputs/emotion", test_text=None, repeat=2):
    """一键测试7种情感"""
    os.makedirs(output_dir, exist_ok=True)
    if test_text is None:
        test_text = "今天的项目终于完成了，我们可以好好休息一下了。这段时间大家都辛苦了。"

    ec = EmotionController(cosyvoice_model)
    results = []

    print(f"\n{'='*60}")
    print(f"情感控制测试: {len(EMOTION_INSTRUCTIONS)}种情感 × {repeat}")
    print(f"{'='*60}")

    for emotion in EMOTION_INSTRUCTIONS:
        for r in range(repeat):
            t_start = time.time()
            audio = ec.synthesize_emotion(test_text, emotion, prompt_wav)
            synth_time = time.time() - t_start
            duration = audio.shape[1] / cosyvoice_model.sample_rate
            path = os.path.join(output_dir, f"emotion_{emotion}_{r+1}.wav")
            torchaudio.save(path, audio, cosyvoice_model.sample_rate)

            # 声学特征分析
            audio_np = audio.squeeze().cpu().numpy()
            acoustic = analyze_acoustic_features(audio_np, cosyvoice_model.sample_rate)

            result = {
                "emotion": emotion,
                "repeat": r + 1,
                "duration_sec": round(duration, 2),
                "synth_sec": round(synth_time, 2),
                "wav": os.path.basename(path),
                "acoustic": acoustic,
            }
            results.append(result)
            f0_str = f"F0={acoustic.get('f0_mean', -1):.1f}" if acoustic.get('f0_mean', -1) > 0 else ""
            print(f"[{emotion}] ({r+1}/{repeat}) -> {duration:.2f}s / {synth_time:.2f}s {f0_str}")

    # 保存汇总
    with open(os.path.join(output_dir, "emotion_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 按情感汇总统计
    from collections import defaultdict
    stats = defaultdict(list)
    for r in results:
        stats[r["emotion"]].append(r)

    print(f"\n{'='*60}")
    print("情感声学特征对比")
    print(f"{'='*60}")
    print(f"{'情感':<10} {'平均时长(s)':<14} {'平均F0(Hz)':<14} {'平均RMS':<12}")
    print("-" * 50)
    for emotion, items in sorted(stats.items()):
        avg_dur = np.mean([x["duration_sec"] for x in items])
        avg_f0 = np.mean([x["acoustic"].get("f0_mean", 0) for x in items if x["acoustic"].get("f0_mean", -1) > 0])
        avg_rms = np.mean([x["acoustic"]["rms_energy"] for x in items])
        f0_display = f"{avg_f0:.1f}" if avg_f0 > 0 else "N/A"
        print(f"{emotion:<10} {avg_dur:<14.2f} {f0_display:<14} {avg_rms:<12.4f}")

    print(f"\n已保存结果到: {output_dir}")
    return results


def run_intensity_test(cosyvoice_model, prompt_wav,
                        output_dir="./outputs/emotion", test_text=None):
    """情感强度梯度测试"""
    os.makedirs(output_dir, exist_ok=True)
    if test_text is None:
        test_text = "这件事实在是出乎意料，我不知道该说什么好。"

    ec = EmotionController(cosyvoice_model)
    results = []

    print(f"\n{'='*60}")
    print("情感强度梯度测试")
    print(f"{'='*60}")

    for emotion, intensities in EMOTION_INTENSITY.items():
        for name, desc in intensities:
            t_start = time.time()
            audio = ec.synthesize_emotion(test_text, emotion, prompt_wav, intensity=name)
            synth_time = time.time() - t_start
            duration = audio.shape[1] / cosyvoice_model.sample_rate
            path = os.path.join(output_dir, f"intensity_{name}.wav")
            torchaudio.save(path, audio, cosyvoice_model.sample_rate)

            acoustic = analyze_acoustic_features(
                audio.squeeze().cpu().numpy(), cosyvoice_model.sample_rate
            )
            results.append({
                "intensity_label": name,
                "emotion": emotion,
                "description": desc,
                "duration_sec": round(duration, 2),
                "synth_sec": round(synth_time, 2),
                "wav": os.path.basename(path),
                "acoustic": acoustic,
            })
            print(f"[{name}] -> {duration:.2f}s / {synth_time:.2f}s F0={acoustic.get('f0_mean', -1):.1f}")

    with open(os.path.join(output_dir, "intensity_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


# CLI 独立运行
if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.environ.get("COSYVOICE_REPO", "/data/haoaokai/zbs/CosyVoice"))
    sys.path.insert(0, os.path.join(os.environ.get("COSYVOICE_REPO", "/data/haoaokai/zbs/CosyVoice"), "third_party/Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    parser = argparse.ArgumentParser(description="情感控制测试")
    parser.add_argument("--model-dir", default=os.environ.get("COSYVOICE_MODEL", "/data/haoaokai/zbs/CosyVoice2-0.5B"))
    parser.add_argument("--output-dir", default="./outputs/emotion")
    parser.add_argument("--test", choices=["emotion", "intensity", "all"], default="all")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    cosyvoice = AutoModel(model_dir=args.model_dir)
    prompt_wav = os.path.join(os.path.dirname(args.model_dir), "CosyVoice", "asset", "zero_shot_prompt.wav")

    if args.test in ("emotion", "all"):
        run_emotion_test(cosyvoice, prompt_wav, args.output_dir, repeat=args.repeat)
    if args.test in ("intensity", "all"):
        run_intensity_test(cosyvoice, prompt_wav, args.output_dir)