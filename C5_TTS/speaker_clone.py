# -*- coding: utf-8 -*-
"""
进阶④ 说话人控制 (Zero-Shot克隆)
验证 zero-shot 语音克隆效果，测试不同参考音频条件下的说话人还原度
- 不同参考时长 (1s/3s/5s/10s)
- 跨性别克隆
- 相似度评估 (speaker embedding cosine similarity)

重要：当参考音频内容未知时（如来自其他语种），自动走 cross_lingual 模式，
避免 prompt_text 与实际音频不匹配导致的输出乱码。
"""
import os, sys, json, time
import torch
import torchaudio
import numpy as np

# 默认的评估用测试文本
DEFAULT_EVAL_TEXT = "你好，欢迎使用智能语音合成系统。今天我们将一起探讨人工智能的发展趋势。"


class SpeakerCloneTester:
    """Zero-Shot说话人克隆测试器"""

    def __init__(self, cosyvoice_model, prompt_wav, prompt_text=""):
        self.model = cosyvoice_model
        self.sr = cosyvoice_model.sample_rate
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text

    def clone_speaker(self, text, ref_wav, ref_text="", zero_shot_spk_id="",
                       cross_lingual=None):
        """Zero-shot 说话人克隆合成

        Args:
            text: 目标文本
            ref_wav: 参考音频路径
            ref_text: 参考音频的文本内容（如果不知道，留空则自动走 cross_lingual）
            zero_shot_spk_id: 已注册音色ID
            cross_lingual: 是否强制使用 cross_lingual 模式。
                           None=自动(无ref_text时走cross_lingual)
        """
        # 智能判断：如果 ref_text 为空，说明音频内容未知，走 cross_lingual 保底
        if cross_lingual is None:
            cross_lingual = (not ref_text) and (not zero_shot_spk_id)

        if zero_shot_spk_id:
            # 方式A: 用缓存的 speaker embedding（已注册，快且可靠）
            gen = self.model.inference_zero_shot(
                text, "", "", zero_shot_spk_id=zero_shot_spk_id, stream=False
            )
        elif cross_lingual:
            # 方式B: cross_lingual（不需要prompt_text，从音频提取音色+跨语言合成）
            gen = self.model.inference_cross_lingual(
                text, ref_wav, stream=False
            )
        else:
            # 方式C: zero_shot（需要prompt_text与音频内容一致）
            gen = self.model.inference_zero_shot(
                text, ref_text, ref_wav, stream=False
            )
        chunks = [j["tts_speech"] for j in gen]
        audio = torch.cat(chunks, dim=1)
        return audio

    def register_speaker(self, spk_id, ref_wav, ref_text=""):
        """注册说话人到音色库"""
        success = self.model.add_zero_shot_spk(ref_text, ref_wav, spk_id)
        if success:
            self.model.save_spkinfo()
            print(f"[说话人] 已注册: {spk_id}")
        return success

    def save_trimmed_audio(self, ref_wav, duration_sec, output_path):
        """截取音频并保存"""
        waveform, sr = torchaudio.load(ref_wav)
        n_samples = int(duration_sec * sr)
        if waveform.shape[1] > n_samples:
            waveform = waveform[:, :n_samples]
        torchaudio.save(output_path, waveform, sr)
        return output_path

    def compute_similarity(self, audio1, audio2, sr=None):
        """计算两段音频的说话人相似度 (基于 CosyVoice speaker embedding)

        使用模型自带的 spk_encoder 提取说话人向量，然后计算余弦相似度。
        结果范围通常 0.5~1.0，越高越像同一个说话人。
        """
        if sr is None:
            sr = self.sr
        try:
            # 如果有 speaker encoder（CosyVoice2 有 spk_encoder 属性）
            spk_enc = getattr(self.model, 'spk_encoder', None)
            if spk_enc is not None:
                import torchaudio.functional as F_audio

                def _extract_embedding(wav, orig_sr):
                    if not torch.is_tensor(wav):
                        wav = torch.from_numpy(np.array(wav)).squeeze().float()
                    else:
                        wav = wav.squeeze().float()
                    if wav.dim() == 1:
                        wav = wav.unsqueeze(0)  # (1, T)
                    # 重采样到 16kHz（speaker encoder 通常用 16kHz）
                    if orig_sr != 16000:
                        wav_16k = F_audio.resample(wav, orig_sr, 16000)
                    else:
                        wav_16k = wav
                    # 提取 embedding
                    with torch.no_grad():
                        emb = spk_enc(wav_16k.to(next(spk_enc.parameters()).device))
                    return emb.squeeze().cpu()

                emb1 = _extract_embedding(audio1, sr)
                emb2 = _extract_embedding(audio2, sr)
                sim = float(torch.nn.functional.cosine_similarity(
                    emb1.unsqueeze(0), emb2.unsqueeze(0)
                ))
                return sim

            # 备选方案：用前端 component 提取
            frontend = getattr(self.model, 'frontend', None)
            if frontend and hasattr(frontend, 'spk_encoder'):
                # 老版本 CosyVoice
                pass

            # 兜底：model 上没有 speaker encoder 就用 Mel 频谱相似度
            import torchaudio.transforms as T
            mel = T.MelSpectrogram(sample_rate=sr, n_mels=80).to(
                audio1.device if torch.is_tensor(audio1) else 'cpu'
            )
            if not torch.is_tensor(audio1):
                a1 = torch.from_numpy(np.array(audio1)).squeeze().float()
            else:
                a1 = audio1.squeeze().float()
            if not torch.is_tensor(audio2):
                a2 = torch.from_numpy(np.array(audio2)).squeeze().float()
            else:
                a2 = audio2.squeeze().float()
            min_len = min(len(a1), len(a2))
            a1 = a1[:min_len].unsqueeze(0)
            a2 = a2[:min_len].unsqueeze(0)
            m1 = mel(a1).squeeze(0).flatten()
            m2 = mel(a2).squeeze(0).flatten()
            similarity = float(torch.nn.functional.cosine_similarity(
                m1.unsqueeze(0), m2.unsqueeze(0)
            ))
            return similarity
        except Exception as e:
            return -1.0

    def test_duration_impact(self, ref_wav, ref_text=None, eval_text=None,
                              output_dir="./outputs/speaker_clone", durations=None):
        """测试不同参考音频时长对克隆效果的影响

        注意：因为参考音频内容未知（来自外部数据），统一用 cross_lingual 模式
        """
        os.makedirs(output_dir, exist_ok=True)
        if eval_text is None:
            eval_text = DEFAULT_EVAL_TEXT
        if durations is None:
            durations = [1, 3, 5, 10]
        if ref_text is None:
            ref_text = ""

        results = []
        print(f"\n{'='*60}")
        print(f"参考音频时长影响测试: {durations} (cross_lingual模式)")
        print(f"{'='*60}")

        # 基准：完整参考音频 (cross_lingual)
        t_start = time.time()
        audio_full = self.clone_speaker(eval_text, ref_wav, ref_text=ref_text,
                                         cross_lingual=True)
        full_time = time.time() - t_start
        full_dur = audio_full.shape[1] / self.sr
        path = os.path.join(output_dir, "speaker_full.wav")
        torchaudio.save(path, audio_full, self.sr)
        print(f"[full] {full_dur:.2f}s音频 / {full_time:.2f}s合成")

        # 不同时长截断测试
        for d in durations:
            trimmed_path = os.path.join(output_dir, f"_tmp_trimmed_{d}s.wav")
            self.save_trimmed_audio(ref_wav, d, trimmed_path)

            t_start = time.time()
            audio = self.clone_speaker(eval_text, trimmed_path, ref_text="",
                                        cross_lingual=True)
            synth_time = time.time() - t_start
            duration = audio.shape[1] / self.sr
            path = os.path.join(output_dir, f"speaker_dur_{d}s.wav")
            torchaudio.save(path, audio, self.sr)

            # 计算相似度
            sim = self.compute_similarity(audio_full, audio)

            result = {
                "duration_sec": d,
                "audio_sec": round(duration, 2),
                "synth_sec": round(synth_time, 2),
                "wav": os.path.basename(path),
                "similarity": round(sim, 4),
            }
            results.append(result)
            print(f"[{d}s] -> {duration:.2f}s音频 / {synth_time:.2f}s合成 / 相似度={sim:.4f}")

            # 清理临时文件
            if os.path.exists(trimmed_path):
                os.remove(trimmed_path)

        with open(os.path.join(output_dir, "duration_results.json"), "w", encoding="utf-8") as f:
            json.dump({"ref_wav": ref_wav, "results": results}, f, ensure_ascii=False, indent=2)

        return results

    def test_cross_gender(self, ref_wavs, eval_text=None,
                           output_dir="./outputs/speaker_clone"):
        """测试多说话人克隆效果 (用cross_lingual避免prompt_text不匹配)"""
        os.makedirs(output_dir, exist_ok=True)
        if eval_text is None:
            eval_text = DEFAULT_EVAL_TEXT

        results = []
        print(f"\n{'='*60}")
        print(f"多说话人克隆测试: {len(ref_wavs)}个参考音频 (cross_lingual模式)")
        print(f"{'='*60}")

        for name, ref_wav in ref_wavs.items():
            if not os.path.exists(ref_wav):
                print(f"[WARNING] 参考音频不存在: {ref_wav}")
                continue

            t_start = time.time()
            # 用cross_lingual因为外部音频内容不确定
            audio = self.clone_speaker(eval_text, ref_wav, ref_text="",
                                        cross_lingual=True)
            synth_time = time.time() - t_start
            duration = audio.shape[1] / self.sr
            path = os.path.join(output_dir, f"speaker_{name}.wav")
            torchaudio.save(path, audio, self.sr)

            results.append({
                "speaker": name,
                "ref_wav": ref_wav,
                "audio_sec": round(duration, 2),
                "synth_sec": round(synth_time, 2),
                "wav": os.path.basename(path),
            })
            print(f"[{name}] -> {duration:.2f}s音频 / {synth_time:.2f}s合成")

        with open(os.path.join(output_dir, "cross_gender_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results


def run_speaker_clone_test(cosyvoice_model, prompt_wav, prompt_text="",
                            ref_wavs=None, output_dir="./outputs/speaker_clone"):
    """一键运行说话人克隆测试"""
    tester = SpeakerCloneTester(cosyvoice_model, prompt_wav, prompt_text)

    # 1. 测试不同时长 — 默认识别内容，直接用内置prompt_wav
    print(f"[INFO] 时长测试使用: {prompt_wav}")
    if prompt_text:
        tester.test_duration_impact(prompt_wav, ref_text=prompt_text,
                                     output_dir=output_dir)
    else:
        tester.test_duration_impact(prompt_wav, output_dir=output_dir)

    # 2. 多说话人测试
    if ref_wavs and len(ref_wavs) > 0:
        tester.test_cross_gender(ref_wavs, output_dir=output_dir)

    print(f"\n测试完成，结果保存在: {output_dir}")


# CLI 独立运行
if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.environ.get("COSYVOICE_REPO", "/data/haoaokai/zbs/CosyVoice"))
    sys.path.insert(0, os.path.join(os.environ.get("COSYVOICE_REPO", "/data/haoaokai/zbs/CosyVoice"), "third_party/Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    parser = argparse.ArgumentParser(description="说话人克隆测试")
    parser.add_argument("--model-dir", default=os.environ.get("COSYVOICE_MODEL", "/data/haoaokai/zbs/CosyVoice2-0.5B"))
    parser.add_argument("--ref-audio-dir", default=None, help="参考音频目录")
    parser.add_argument("--ref-wav", default=None, help="单个参考音频路径")
    parser.add_argument("--output-dir", default="./outputs/speaker_clone")
    parser.add_argument("--duration-test", action="store_true", help="运行时长影响测试")
    parser.add_argument("--all", action="store_true", help="运行全部测试")
    args = parser.parse_args()

    cosyvoice = AutoModel(model_dir=args.model_dir)
    prompt_wav = args.ref_wav or os.path.join(
        os.path.dirname(args.model_dir), "CosyVoice", "asset", "zero_shot_prompt.wav"
    )
    prompt_text = "希望你以后能够做的比我还好呦。"

    tester = SpeakerCloneTester(cosyvoice, prompt_wav, prompt_text)

    if args.duration_test or args.all:
        tester.test_duration_impact(prompt_wav, ref_text=prompt_text,
                                     output_dir=args.output_dir)

    if args.ref_audio_dir and os.path.isdir(args.ref_audio_dir):
        audio_files = [f for f in os.listdir(args.ref_audio_dir) if f.endswith(('.wav', '.mp3'))]
        ref_wavs = {}
        for af in audio_files[:8]:
            name = os.path.splitext(af)[0]
            ref_wavs[name] = os.path.join(args.ref_audio_dir, af)
        if ref_wavs:
            tester.test_cross_gender(ref_wavs, output_dir=args.output_dir)