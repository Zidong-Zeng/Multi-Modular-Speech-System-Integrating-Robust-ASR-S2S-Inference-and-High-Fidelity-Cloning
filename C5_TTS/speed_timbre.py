# -*- coding: utf-8 -*-
"""
进阶② 语速与音色控制
方案A: Instruction指令控制 (推荐，inference_instruct2原生支持)
方案B: 后处理time-stretch (librosa，备选)
支持 speed 参数原生调速 + 多音色参考音频库
"""
import os, sys, json, time
import torch
import torchaudio
import numpy as np

# 语速档位预设 (instruction方式)
SPEED_INSTRUCTIONS = {
    "very_slow": "用非常慢的语速朗读，每个字都拖长",
    "slow": "语速较慢，从容不迫地朗读",
    "normal": "用正常的语速朗读",
    "fast": "语速较快，清晰快速地朗读",
    "very_fast": "用非常快的语速，快速说完",
}

# 语速数值预设 (原生speed参数)
SPEED_VALUES = {
    "very_slow": 0.7,
    "slow": 0.85,
    "normal": 1.0,
    "fast": 1.2,
    "very_fast": 1.5,
}


class SpeedController:
    """语速控制器 - 原生 speed 参数 + 后处理time-stretch"""

    def __init__(self, cosyvoice_model):
        self.model = cosyvoice_model
        self.sr = cosyvoice_model.sample_rate

    def synthesize_with_speed(self, text, prompt_text, prompt_wav, speed=1.0):
        """使用原生speed参数合成（推荐）"""
        gen = self.model.inference_zero_shot(
            text, prompt_text, prompt_wav, stream=False, speed=speed
        )
        chunks = [j["tts_speech"] for j in gen]
        audio = torch.cat(chunks, dim=1)
        return audio

    def synthesize_with_instruction(self, text, instruct, prompt_wav):
        """使用instruct2接口控制语速"""
        gen = self.model.inference_instruct2(
            text, instruct + "<|endofprompt|>", prompt_wav, stream=False
        )
        chunks = [j["tts_speech"] for j in gen]
        audio = torch.cat(chunks, dim=1)
        return audio

    def time_stretch(self, audio, rate):
        """后处理变速 (备选，使用PyTorch原生方法)"""
        if rate == 1.0:
            return audio
        try:
            import librosa
            audio_np = audio.squeeze().cpu().numpy() if torch.is_tensor(audio) else audio.squeeze()
            stretched = librosa.effects.time_stretch(y=audio_np, rate=rate)
            return torch.from_numpy(stretched).unsqueeze(0)
        except ImportError:
            # fallback: torchaudio speed
            waveform, sr = audio, self.sr
            if torch.is_tensor(audio):
                waveform = audio
            effects = [["speed", str(rate)]]
            stretched, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sr, effects
            )
            return stretched


class TimbreController:
    """音色控制器 - 通过不同参考音频实现多音色"""

    def __init__(self, cosyvoice_model, voice_bank_dir="./voice_bank"):
        self.model = cosyvoice_model
        self.sr = cosyvoice_model.sample_rate
        self.voice_bank_dir = voice_bank_dir
        self.voices = {}  # voice_name -> {"wav": path, "text": str, "spk_id": str}
        os.makedirs(voice_bank_dir, exist_ok=True)
        self._load_voice_bank()

    def _load_voice_bank(self):
        """从voice_bank目录加载已注册音色"""
        config_path = os.path.join(self.voice_bank_dir, "voice_bank.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.voices = json.load(f)
            # 注册到CosyVoice模型中
            for name, info in self.voices.items():
                if info.get("wav") and os.path.exists(info["wav"]):
                    try:
                        self.model.add_zero_shot_spk(
                            info["text"], info["wav"], name
                        )
                    except Exception as e:
                        print(f"[WARNING] 注册音色 '{name}' 失败: {e}")

    def register_voice(self, name, ref_wav, ref_text=""):
        """注册新音色"""
        self.voices[name] = {"wav": ref_wav, "text": ref_text, "spk_id": name}
        self.model.add_zero_shot_spk(ref_text, ref_wav, name)
        self.model.save_spkinfo()
        # 保存到JSON
        config_path = os.path.join(self.voice_bank_dir, "voice_bank.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.voices, f, ensure_ascii=False, indent=2)
        print(f"[音色] 已注册: {name}")
        return name

    def synthesize_with_voice(self, text, voice_name, prompt_text="", prompt_wav=""):
        """使用已注册的音色合成"""
        if voice_name in self.voices:
            gen = self.model.inference_zero_shot(
                text, "", "", zero_shot_spk_id=voice_name, stream=False
            )
        else:
            # fallback: 直接用参考音频
            info = self.voices.get(voice_name, {})
            ref_wav = info.get("wav", prompt_wav)
            ref_text = info.get("text", prompt_text)
            if ref_text:
                gen = self.model.inference_zero_shot(
                    text, ref_text, ref_wav, stream=False
                )
            else:
                # ref_text 为空时走 cross_lingual，避免空 prompt_text 导致乱码
                gen = self.model.inference_cross_lingual(
                    text, ref_wav, stream=False
                )
        chunks = [j["tts_speech"] for j in gen]
        audio = torch.cat(chunks, dim=1)
        return audio

    def list_voices(self):
        """列出所有已注册音色"""
        return list(self.voices.keys())

    def synthesize_with_ref(self, text, ref_wav, ref_text=""):
        """直接使用参考音频路径合成（不注册）

        重要：当 ref_text 为空（参考音频内容未知）时，自动走 cross_lingual 模式，
        避免 prompt_text 与实际音频不匹配导致的输出乱码。
        """
        if ref_text:
            # zero_shot 模式：需要 prompt_text 与音频内容严格一致
            gen = self.model.inference_zero_shot(
                text, ref_text, ref_wav, stream=False
            )
        else:
            # cross_lingual 模式：不需要 prompt_text，从音频提取音色
            gen = self.model.inference_cross_lingual(
                text, ref_wav, stream=False
            )
        chunks = [j["tts_speech"] for j in gen]
        audio = torch.cat(chunks, dim=1)
        return audio


def analyze_f0(audio_np, sr):
    """分析基频F0分布，量化音色差异"""
    try:
        import pyworld as pw
        audio_f64 = audio_np.astype(np.float64)
        f0, t = pw.harvest(audio_f64, sr)
        f0_valid = f0[f0 > 0]
        if len(f0_valid) == 0:
            return {"f0_mean": 0, "f0_std": 0, "f0_min": 0, "f0_max": 0}
        return {
            "f0_mean": float(np.mean(f0_valid)),
            "f0_std": float(np.std(f0_valid)),
            "f0_min": float(np.min(f0_valid)),
            "f0_max": float(np.max(f0_valid)),
        }
    except ImportError:
        return {"f0_mean": -1, "f0_std": -1, "f0_min": -1, "f0_max": -1}


def run_speed_test(cosyvoice_model, prompt_wav, prompt_text,
                    output_dir="./outputs/speed_timbre", test_text=None):
    """一键语速测试：5档语速对比"""
    os.makedirs(output_dir, exist_ok=True)
    if test_text is None:
        test_text = "今天天气真不错，我们一起去公园散步吧。人工智能正在深刻改变着我们的日常生活。"

    sc = SpeedController(cosyvoice_model)
    results = []
    print(f"\n{'='*60}")
    print("语速测试 (原生speed参数): 5档")
    print(f"{'='*60}")

    for speed_name, speed_val in SPEED_VALUES.items():
        t_start = time.time()
        audio = sc.synthesize_with_speed(test_text, prompt_text, prompt_wav, speed=speed_val)
        synth_time = time.time() - t_start
        duration = audio.shape[1] / cosyvoice_model.sample_rate
        path = os.path.join(output_dir, f"speed_{speed_name}.wav")
        torchaudio.save(path, audio, cosyvoice_model.sample_rate)

        result = {
            "speed_name": speed_name,
            "speed_value": speed_val,
            "duration_sec": round(duration, 2),
            "synth_sec": round(synth_time, 2),
            "wav": os.path.basename(path),
        }
        results.append(result)
        print(f"[{speed_name}] speed={speed_val} -> {duration:.2f}s音频 / {synth_time:.2f}s合成")

    # 保存汇总
    with open(os.path.join(output_dir, "speed_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def run_timbre_test(cosyvoice_model, prompt_wav, prompt_text,
                     ref_audio_dir=None, output_dir="./outputs/speed_timbre", test_text=None):
    """一键音色测试：使用不同参考音频合成"""
    os.makedirs(output_dir, exist_ok=True)
    if test_text is None:
        test_text = "你好，欢迎使用智能语音合成系统。今天我们将一起探讨人工智能的发展趋势。"

    tc = TimbreController(cosyvoice_model)
    
    # 默认使用内置的zero_shot_prompt作为基准
    results = []
    
    # 1. 基准音色 (默认prompt_wav)
    t_start = time.time()
    audio = tc.synthesize_with_ref(test_text, prompt_wav, prompt_text)
    synth_time = time.time() - t_start
    duration = audio.shape[1] / cosyvoice_model.sample_rate
    path = os.path.join(output_dir, "timbre_default.wav")
    torchaudio.save(path, audio, cosyvoice_model.sample_rate)
    results.append({
        "voice": "default",
        "ref_wav": prompt_wav,
        "duration_sec": round(duration, 2),
        "synth_sec": round(synth_time, 2),
        "wav": os.path.basename(path),
    })
    print(f"[default] -> {duration:.2f}s音频 / {synth_time:.2f}s合成")

    # 2. 如果有参考音频目录，逐个测试
    if ref_audio_dir and os.path.isdir(ref_audio_dir):
        audio_files = [f for f in os.listdir(ref_audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        for af in audio_files[:5]:  # 最多测试5个
            voice_name = os.path.splitext(af)[0]
            ref_path = os.path.join(ref_audio_dir, af)
            try:
                t_start = time.time()
                audio = tc.synthesize_with_ref(test_text, ref_path, "")
                synth_time = time.time() - t_start
                duration = audio.shape[1] / cosyvoice_model.sample_rate
                path = os.path.join(output_dir, f"timbre_{voice_name}.wav")
                torchaudio.save(path, audio, cosyvoice_model.sample_rate)
                results.append({
                    "voice": voice_name,
                    "ref_wav": ref_path,
                    "duration_sec": round(duration, 2),
                    "synth_sec": round(synth_time, 2),
                    "wav": os.path.basename(path),
                    # F0分析
                    "f0": analyze_f0(audio.squeeze().cpu().numpy(), cosyvoice_model.sample_rate),
                })
                print(f"[{voice_name}] -> {duration:.2f}s音频 / {synth_time:.2f}s合成")
            except Exception as e:
                print(f"[WARNING] 音色 '{voice_name}' 合成失败: {e}")

    with open(os.path.join(output_dir, "timbre_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


# CLI 独立运行
if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.environ.get("COSYVOICE_REPO", "/data/haoaokai/zbs/CosyVoice"))
    sys.path.insert(0, os.path.join(os.environ.get("COSYVOICE_REPO", "/data/haoaokai/zbs/CosyVoice"), "third_party/Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    parser = argparse.ArgumentParser(description="语速与音色控制测试")
    parser.add_argument("--model-dir", default=os.environ.get("COSYVOICE_MODEL", "/data/haoaokai/zbs/CosyVoice2-0.5B"))
    parser.add_argument("--ref-audio-dir", default=None, help="参考音频目录")
    parser.add_argument("--output-dir", default="./outputs/speed_timbre")
    parser.add_argument("--test", choices=["speed", "timbre", "all"], default="all")
    args = parser.parse_args()

    cosyvoice = AutoModel(model_dir=args.model_dir)
    prompt_wav = os.path.join(os.path.dirname(args.model_dir), "CosyVoice", "asset", "zero_shot_prompt.wav")
    prompt_text = "希望你以后能够做的比我还好呦。"

    if args.test in ("speed", "all"):
        run_speed_test(cosyvoice, prompt_wav, prompt_text, args.output_dir)
    if args.test in ("timbre", "all"):
        run_timbre_test(cosyvoice, prompt_wav, prompt_text, args.ref_audio_dir, args.output_dir)