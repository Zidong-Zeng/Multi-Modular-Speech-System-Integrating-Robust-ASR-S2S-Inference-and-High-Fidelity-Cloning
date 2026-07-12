# -*- coding: utf-8 -*-
"""
进阶① 多语种支持验证
测试 CosyVoice2 对 zh/en/ja/yue/ko/mix 多语种生成能力
"""
import os, sys, json, time
import torch
import torchaudio
import soundfile as sf
import numpy as np

# 语言token映射
LANG_TOKEN = {
    "zh": "<|zh|>",
    "en": "<|en|>",
    "ja": "<|ja|>",
    "yue": "<|yue|>",
    "ko": "<|ko|>",
}

# 默认测试文本
TEST_TEXTS = {
    "zh": "人工智能正在改变我们的生活方式，带来了前所未有的机遇与挑战。",
    "en": "Artificial intelligence is transforming the way we live, bringing unprecedented opportunities and challenges.",
    "ja": "人工知能は私たちの生活様式を変革し、前例のない機会と課題をもたらしています。",
    "yue": "人工智能改变紧我哋嘅生活方式，带嚟前所未有嘅机遇同挑战。",
    "ko": "인공지능은 우리의 생활 방식을 변화시키고 있으며, 전례 없는 기회와 도전을 가져오고 있습니다.",
    "mix": "今天的 meeting 我们主要讨论一下 AI 的 roadmap，特别是 Q4 的 delivery timeline。",
}

# 参考音频路径 (默认)
DEFAULT_PROMPT_WAV = None  # 动态设置
DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"


class MultilingualTester:
    """多语种合成测试器"""

    def __init__(self, cosyvoice_model, prompt_wav, prompt_text=None, output_dir="./outputs/multilingual"):
        self.model = cosyvoice_model
        self.sr = cosyvoice_model.sample_rate
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text or DEFAULT_PROMPT_TEXT
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def synthesize_lang(self, lang, text, output_name=None):
        """合成单个语种"""
        t_start = time.time()
        if lang == "zh":
            gen = self.model.inference_zero_shot(
                text, self.prompt_text, self.prompt_wav, stream=False
            )
        elif lang == "mix":
            # 中英混合用 cross_lingual
            gen = self.model.inference_cross_lingual(
                text, self.prompt_wav, stream=False
            )
        else:
            # 其他语言用跨语言接口
            token = LANG_TOKEN.get(lang, "")
            gen = self.model.inference_cross_lingual(
                token + text, self.prompt_wav, stream=False
            )
        
        chunks = [j["tts_speech"] for j in gen]
        audio = torch.cat(chunks, dim=1)
        synth_time = time.time() - t_start
        duration = audio.shape[1] / self.sr

        if output_name is None:
            output_name = f"lang_{lang}.wav"
        path = os.path.join(self.output_dir, output_name)
        torchaudio.save(path, audio, self.sr)

        return {
            "lang": lang,
            "text": text,
            "wav": os.path.basename(path),
            "duration_sec": round(duration, 2),
            "synth_sec": round(synth_time, 2),
            "rtf": round(synth_time / max(duration, 0.01), 3),
        }

    def test_all_languages(self, langs=None, repeat=3):
        """批量测试所有语种"""
        if langs is None:
            langs = list(TEST_TEXTS.keys())
        
        all_results = []
        print(f"{'='*60}")
        print(f"多语种测试: {langs} × {repeat}")
        print(f"{'='*60}")

        for lang in langs:
            text = TEST_TEXTS.get(lang, TEST_TEXTS["zh"])
            for r in range(repeat):
                result = self.synthesize_lang(lang, text, f"lang_{lang}_{r+1}.wav")
                all_results.append(result)
                print(f"[{lang}] ({r+1}/{repeat}) '{text[:30]}...' "
                      f"-> {result['duration_sec']}s音频 / {result['synth_sec']}s / RTF={result['rtf']}")

        # 保存汇总
        summary_path = os.path.join(self.output_dir, "multilingual_summary.json")
        summary = {
            "model": "CosyVoice2-0.5B",
            "test_languages": langs,
            "results": all_results,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n已保存汇总到: {summary_path}")
        return all_results

    def evaluate(self, results):
        """按语种统计平均RTF和音频时长"""
        from collections import defaultdict
        stats = defaultdict(list)
        for r in results:
            stats[r["lang"]].append(r)
        
        print(f"\n{'='*60}")
        print("多语种评估汇总")
        print(f"{'='*60}")
        print(f"{'语种':<8} {'样本数':<6} {'平均RTF':<10} {'平均时长(s)':<12}")
        print("-" * 40)
        for lang, items in sorted(stats.items()):
            avg_rtf = np.mean([x["rtf"] for x in items])
            avg_dur = np.mean([x["duration_sec"] for x in items])
            print(f"{lang:<8} {len(items):<6} {avg_rtf:<10.3f} {avg_dur:<12.2f}")
        return stats


def run_multilingual_test(cosyvoice_model, prompt_wav, prompt_text=None, 
                           output_dir="./outputs/multilingual", repeat=1):
    """便捷函数：一键多语种测试"""
    tester = MultilingualTester(cosyvoice_model, prompt_wav, prompt_text, output_dir)
    results = tester.test_all_languages(repeat=repeat)
    tester.evaluate(results)
    return results


# CLI 独立运行
if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.environ.get("COSYVOICE_REPO", "/data/haoaokai/zbs/CosyVoice"))
    sys.path.insert(0, os.path.join(os.environ.get("COSYVOICE_REPO", "/data/haoaokai/zbs/CosyVoice"), "third_party/Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    parser = argparse.ArgumentParser(description="多语种合成测试")
    parser.add_argument("--model-dir", default=os.environ.get("COSYVOICE_MODEL", "/data/haoaokai/zbs/CosyVoice2-0.5B"))
    parser.add_argument("--prompt-wav", default=None)
    parser.add_argument("--output-dir", default="./outputs/multilingual")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    cosyvoice = AutoModel(model_dir=args.model_dir)
    prompt_wav = args.prompt_wav or os.path.join(args.model_dir, "..", "CosyVoice", "asset", "zero_shot_prompt.wav")
    
    run_multilingual_test(cosyvoice, prompt_wav, output_dir=args.output_dir, repeat=args.repeat)