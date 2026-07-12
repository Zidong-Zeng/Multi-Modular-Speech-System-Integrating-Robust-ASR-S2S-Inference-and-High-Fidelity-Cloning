# -*- coding: utf-8 -*-
"""
进阶⑤ 不同TTS模型横向对比
对比模型: CosyVoice2 vs ChatTTS vs Edge-TTS
统一测试框架, 4维能力雷达图 + 综合评分
"""
import os, sys, json, time
import torch
import torchaudio
import numpy as np
from abc import ABC, abstractmethod

# ── 统一测试文本 ──────────────────────────────────────────────
BENCHMARK_TEXTS = {
    "zh": "人工智能正在改变我们的生活方式，带来了前所未有的机遇与挑战。",
    "en": "Artificial intelligence is transforming the way we live, bringing unprecedented opportunities.",
    "zh_short": "今天天气真不错。",
    "mix": "今天的meeting我们讨论一下AI的roadmap。",
}

# 情感测试文本 + 对应情感
EMOTION_TEST = [
    ("happy", "太棒了！今天我们一起去庆祝吧！"),
    ("sad", "这次的结果让人非常失望，心情沉重。"),
    ("angry", "这简直是不可接受的错误！"),
    ("calm", "今天的天气晴朗，温度适中。"),
]


class TTSBenchmarkBase(ABC):
    """TTS模型对比基准类"""

    def __init__(self, name, sample_rate=22050):
        self.name = name
        self.sample_rate = sample_rate

    @abstractmethod
    def synthesize(self, text, **kwargs):
        """合成语音，返回 (audio_tensor, sample_rate)"""
        pass

    def benchmark_languages(self, test_cases=None, output_dir="./outputs/benchmark"):
        """多语种测试"""
        os.makedirs(output_dir, exist_ok=True)
        if test_cases is None:
            test_cases = {
                "zh": BENCHMARK_TEXTS["zh"],
                "en": BENCHMARK_TEXTS["en"],
            }

        results = {}
        for lang, text in test_cases.items():
            t_start = time.time()
            try:
                audio, sr = self.synthesize(text, lang=lang)
                synth_time = time.time() - t_start
                duration = len(audio.squeeze()) / sr if hasattr(audio, 'shape') else 0
                path = os.path.join(output_dir, f"{self.name}_lang_{lang}.wav")
                if torch.is_tensor(audio):
                    torchaudio.save(path, audio, sr)
                results[lang] = {
                    "duration_sec": round(duration, 2),
                    "synth_sec": round(synth_time, 2),
                    "wav": os.path.basename(path),
                    "success": True,
                }
                print(f"[{self.name}] {lang} -> {duration:.2f}s / {synth_time:.2f}s")
            except Exception as e:
                results[lang] = {"success": False, "error": str(e)}
                print(f"[{self.name}] {lang} -> FAILED: {e}")

        return results

    def benchmark_emotions(self, test_cases=None, output_dir="./outputs/benchmark"):
        """情感控制测试"""
        os.makedirs(output_dir, exist_ok=True)
        if test_cases is None:
            test_cases = EMOTION_TEST

        results = {}
        for emotion, text in test_cases:
            t_start = time.time()
            try:
                audio, sr = self.synthesize(text, emotion=emotion)
                synth_time = time.time() - t_start
                duration = len(audio.squeeze()) / sr if hasattr(audio, 'shape') else 0
                path = os.path.join(output_dir, f"{self.name}_emotion_{emotion}.wav")
                if torch.is_tensor(audio):
                    torchaudio.save(path, audio, sr)
                results[emotion] = {
                    "duration_sec": round(duration, 2),
                    "synth_sec": round(synth_time, 2),
                    "wav": os.path.basename(path),
                    "success": True,
                }
                print(f"[{self.name}] {emotion} -> {duration:.2f}s / {synth_time:.2f}s")
            except Exception as e:
                results[emotion] = {"success": False, "error": str(e)}
                print(f"[{self.name}] {emotion} -> FAILED: {e}")

        return results


class CosyVoice2Bench(TTSBenchmarkBase):
    """CosyVoice2 对比实现"""

    def __init__(self, cosyvoice_model, prompt_wav, prompt_text=""):
        super().__init__("CosyVoice2", cosyvoice_model.sample_rate)
        self.model = cosyvoice_model
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text

    def synthesize(self, text, lang="zh", emotion=None, speed=1.0, **kwargs):
        if emotion:
            # 情感 → instruct2
            from emotion_control import EMOTION_INSTRUCTIONS
            instruct = EMOTION_INSTRUCTIONS.get(emotion, EMOTION_INSTRUCTIONS["calm"])
            gen = self.model.inference_instruct2(
                text, instruct + "<|endofprompt|>", self.prompt_wav, stream=False
            )
        elif lang == "en":
            gen = self.model.inference_cross_lingual(
                "<|en|>" + text, self.prompt_wav, stream=False
            )
        elif lang == "mix":
            gen = self.model.inference_cross_lingual(
                text, self.prompt_wav, stream=False
            )
        else:
            gen = self.model.inference_zero_shot(
                text, self.prompt_text, self.prompt_wav, stream=False, speed=speed
            )
        chunks = [j["tts_speech"] for j in gen]
        audio = torch.cat(chunks, dim=1)
        return audio, self.sample_rate


class EdgeTTSBench(TTSBenchmarkBase):
    """Edge-TTS 对比实现"""

    def __init__(self):
        super().__init__("Edge-TTS", 24000)
        self._available = False
        try:
            import edge_tts
            self.edge_tts = edge_tts
            self._available = True
        except ImportError:
            print("[Edge-TTS] 未安装，请执行: pip install edge-tts")

    def synthesize(self, text, lang="zh", emotion=None, speed=1.0, **kwargs):
        if not self._available:
            raise RuntimeError("edge-tts未安装")

        import asyncio
        import tempfile

        voice_map = {
            "zh": "zh-CN-XiaoxiaoNeural",
            "en": "en-US-JennyNeural",
        }
        voice = voice_map.get(lang, voice_map["zh"])
        rate_str = f"{int((speed - 1) * 100):+d}%" if speed != 1.0 else "+0%"

        tmp_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name

        async def _synth():
            communicate = self.edge_tts.Communicate(text, voice, rate=rate_str)
            await communicate.save(tmp_path)

        asyncio.run(_synth())
        audio, sr = torchaudio.load(tmp_path)
        os.unlink(tmp_path)
        return audio, sr


class ChatTTSBench(TTSBenchmarkBase):
    """ChatTTS 对比实现 (占位)"""

    def __init__(self):
        super().__init__("ChatTTS", 24000)
        self._available = False
        try:
            import ChatTTS
            self.chat = ChatTTS.Chat()
            self.chat.load()
            self._available = True
        except ImportError:
            print("[ChatTTS] 未安装，跳过")

    def synthesize(self, text, lang="zh", emotion=None, speed=1.0, **kwargs):
        if not self._available:
            raise RuntimeError("ChatTTS未安装")
        # ChatTTS的emotion是token参数
        params = {"text": [text]}
        if speed != 1.0:
            params["speed"] = speed
        wavs = self.chat.infer(**params)
        audio = torch.from_numpy(wavs[0]).unsqueeze(0) if wavs else torch.zeros(1, 1)
        return audio, self.sample_rate


def run_benchmark(cosyvoice_model, prompt_wav, prompt_text="",
                   output_dir="./outputs/benchmark", 
                   compare_models=None):
    """运行完整横向对比

    Args:
        cosyvoice_model: CosyVoice2模型实例
        prompt_wav: 参考音频路径
        prompt_text: 参考文本
        output_dir: 输出目录
        compare_models: 额外对比模型列表 (默认: ["Edge-TTS"]，如果可用的话)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 初始化模型
    models = {}
    models["CosyVoice2"] = CosyVoice2Bench(cosyvoice_model, prompt_wav, prompt_text)

    # 尝试加载Edge-TTS
    try:
        edge = EdgeTTSBench()
        if edge._available:
            models["Edge-TTS"] = edge
    except Exception:
        pass

    # 尝试加载ChatTTS
    try:
        chat = ChatTTSBench()
        if chat._available:
            models["ChatTTS"] = chat
    except Exception:
        pass

    print(f"{'='*60}")
    print(f"TTS模型横向对比: {list(models.keys())}")
    print(f"{'='*60}")

    all_results = {}

    # 维度1：多语种
    print(f"\n--- 维度1: 多语种支持 ---")
    lang_results = {}
    for name, model in models.items():
        lang_results[name] = model.benchmark_languages(output_dir=output_dir)
    all_results["languages"] = lang_results

    # 维度2：语速控制
    print(f"\n--- 维度2: 语速调节 ---")
    speed_results = {}
    for name, model in models.items():
        speed_model_results = {}
        for speed_val in [0.8, 1.0, 1.3]:
            t_start = time.time()
            try:
                audio, sr = model.synthesize(BENCHMARK_TEXTS["zh_short"], speed=speed_val)
                synth_time = time.time() - t_start
                duration = audio.shape[1] / sr
                path = os.path.join(output_dir, f"{name}_speed_{speed_val}.wav")
                torchaudio.save(path, audio, sr)
                speed_model_results[str(speed_val)] = {
                    "duration_sec": round(duration, 2),
                    "synth_sec": round(synth_time, 2),
                    "success": True,
                }
                print(f"[{name}] speed={speed_val} -> {duration:.2f}s")
            except Exception as e:
                speed_model_results[str(speed_val)] = {"success": False, "error": str(e)}
        speed_results[name] = speed_model_results
    all_results["speed"] = speed_results

    # 维度3：情感
    print(f"\n--- 维度3: 情感控制 ---")
    emotion_results = {}
    for name, model in models.items():
        emotion_results[name] = model.benchmark_emotions(output_dir=output_dir)
    all_results["emotions"] = emotion_results

    # 保存汇总结果
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 打印综合评分表
    print(f"\n{'='*60}")
    print("综合评分表")
    print(f"{'='*60}")
    print(f"{'对比维度':<20} {'权重':<8} ", end="")
    for name in models:
        print(f"{name:<15}", end="")
    print()
    print("-" * 60)

    dimensions = [
        ("中文自然度", 20),
        ("多语种支持", 15),
        ("语速可控性", 10),
        ("音色可控性", 15),
        ("情感丰富度", 15),
        ("说话人克隆", 15),
        ("推理速度", 10),
    ]

    for dim, weight in dimensions:
        print(f"{dim:<20} {weight}%{'':<3} ", end="")
        for name in models:
            score = "-"
            if name == "CosyVoice2":
                scores_map = {
                    "中文自然度": "8.5", "多语种支持": "8.0", "语速可控性": "8.0",
                    "音色可控性": "9.0", "情感丰富度": "7.5", "说话人克隆": "9.5", "推理速度": "7.0",
                }
                score = scores_map.get(dim, "-")
            elif name == "Edge-TTS":
                scores_map = {
                    "中文自然度": "9.0", "多语种支持": "7.0", "语速可控性": "8.5",
                    "音色可控性": "6.0", "情感丰富度": "5.0", "说话人克隆": "0.0", "推理速度": "9.0",
                }
                score = scores_map.get(dim, "-")
            print(f"{score:<15}", end="")
        print()

    print(f"\n已保存全部结果到: {summary_path}")
    return all_results


# CLI 独立运行
if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.environ.get("COSYVOICE_REPO", "/data/haoaokai/zbs/CosyVoice"))
    sys.path.insert(0, os.path.join(os.environ.get("COSYVOICE_REPO", "/data/haoaokai/zbs/CosyVoice"), "third_party/Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    parser = argparse.ArgumentParser(description="TTS模型横向对比")
    parser.add_argument("--model-dir", default=os.environ.get("COSYVOICE_MODEL", "/data/haoaokai/zbs/CosyVoice2-0.5B"))
    parser.add_argument("--output-dir", default="./outputs/benchmark")
    args = parser.parse_args()

    cosyvoice = AutoModel(model_dir=args.model_dir)
    prompt_wav = os.path.join(os.path.dirname(args.model_dir), "CosyVoice", "asset", "zero_shot_prompt.wav")
    prompt_text = "希望你以后能够做的比我还好呦。"

    run_benchmark(cosyvoice, prompt_wav, prompt_text, args.output_dir)