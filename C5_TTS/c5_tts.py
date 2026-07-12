# -*- coding: utf-8 -*-
"""
C5 文本转语音 (TTS) —— 完整版 (基础要求 + 进阶要求前5条)
模型：CosyVoice2-0.5B

进阶要求前五条:
  ① 多语种支持验证
  ② 语速与音色控制
  ③ 情感控制能力验证
  ④ 说话人控制 (Zero-Shot克隆)
  ⑤ 不同TTS模型横向对比

运行示例:
  # 基础合成
  python c5_tts.py --text "你好世界" -o output.wav

  # 进阶① 多语种测试
  python c5_tts.py --multilingual-test

  # 进阶② 语速测试
  python c5_tts.py --speed-test

  # 进阶③ 情感测试
  python c5_tts.py --emotion-test

  # 进阶④ 说话人克隆测试
  python c5_tts.py --speaker-test

  # 进阶⑤ 横向对比
  python c5_tts.py --benchmark

  # 全部进阶测试
  python c5_tts.py --all-advanced

  # 流式合成
  python c5_tts.py --text "你好" --stream -o stream.wav

  # 语速调节
  python c5_tts.py --text "你好" --speed 1.5 -o fast.wav

  # 中英混合
  python c5_tts.py --text "今天的meeting讨论AI" --lang mix -o mix.wav

## 链接操作
conda activate cosyvoice
export COSYVOICE_REPO=/root/siton-tmp/lmy/CosyVoice
export COSYVOICE_MODEL=/root/siton-tmp/lmy/CosyVoice2-0.5B
  # C3
python c5_tts.py --dataset /root/siton-tmp/assignment_C/C3_cascade/outputs/c3-corrected/cremd200-tiny-ds/c3_predictions.json -o /root/siton-tmp/lmy/C5_TTS/test
# C4
python c5_tts.py --dataset /root/siton-tmp/assignment_C/C4_end2end/outputs -o /root/siton-tmp/lmy/C5_TTS/test

"""

import os, sys, json, time, argparse

# ── 路径配置 ──────────────────────────────────────────────────
COSYVOICE_REPO = os.environ.get("COSYVOICE_REPO", "/data/haoaokai/zbs/CosyVoice")
MODEL_DIR = os.environ.get("COSYVOICE_MODEL", "/data/haoaokai/zbs/CosyVoice2-0.5B")
AUDIO_DATA_DIR = os.environ.get(
    "AUDIO_DATA_DIR",
    "/root/siton-pub/production_practice/CS/assignment_C/common_data/audio",
)
C3_DATASET_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "C3_cascade", "outputs", "cascade_results.json",
)

# 路径校验
for _name, _p in [("COSYVOICE_REPO", COSYVOICE_REPO), ("COSYVOICE_MODEL", MODEL_DIR)]:
    if not os.path.isdir(_p):
        sys.exit(
            f"[C5] 找不到 {_name}: {_p}\n"
            f"请设置环境变量:\n"
            f"  export COSYVOICE_REPO=/你的路径/CosyVoice\n"
            f"  export COSYVOICE_MODEL=/你的路径/CosyVoice2-0.5B"
        )
sys.path.insert(0, COSYVOICE_REPO)
sys.path.insert(0, os.path.join(COSYVOICE_REPO, "third_party/Matcha-TTS"))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

# ── 项目路径 ──────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "outputs")
LOGDIR = os.path.join(HERE, "..", "log")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(LOGDIR, exist_ok=True)

# 默认音色
PROMPT_WAV = os.path.join(COSYVOICE_REPO, "asset", "zero_shot_prompt.wav")
PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

__version__ = "3.0.0"


# ── 日志 ──────────────────────────────────────────────────────
from datetime import datetime


def _write_log_entry(log_data):
    """追加一条日志到 log/runtime.log（JSON Lines 格式）"""
    log_path = os.path.join(LOGDIR, "runtime.log")
    log_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 日志写入失败不影响主流程


# ── 辅助函数 ──────────────────────────────────────────────────
def load_model(model_dir=None, verbose=True):
    """加载 CosyVoice2 模型"""
    model_dir = model_dir or MODEL_DIR
    if verbose:
        print(f"[C5] 加载模型: {model_dir}")
    t0 = time.time()
    cosyvoice = AutoModel(model_dir=model_dir)
    if verbose:
        print(f"[C5] 模型加载完成，耗时 {time.time() - t0:.1f}s，采样率 {cosyvoice.sample_rate}Hz")
    return cosyvoice


def synth_and_save(gen, out_path, sr):
    """运行生成器，拼接并保存音频，返回 (耗时, 路径, 时长)"""
    t = time.time()
    chunks = [j["tts_speech"] for j in gen]
    if not chunks:
        print("[WARNING] 生成器未产生任何音频块")
        return 0, out_path, 0
    audio = torch.cat(chunks, dim=1)
    torchaudio.save(out_path, audio, sr)
    synth_time = time.time() - t
    duration = audio.shape[1] / sr
    return synth_time, out_path, duration


# ── 基础合成 ──────────────────────────────────────────────────
def basic_synthesize(cosyvoice, text, output_path, lang="zh", speed=1.0,
                     voice_ref=None, prompt_text=None, prompt_wav=None, stream=False,
                     emotion=None, instruct=None, voice_id=None,
                     log_id=None):
    """基础合成：文本 → WAV

    Args:
        emotion: 情感标签 (happy/sad/angry/gentle/calm/excited/tired) → 走instruct2
        instruct: 自定义指令文本 (优先级高于emotion)
        voice_ref: 参考音频路径 (音色克隆)
        voice_id:  已注册音色ID (如 '中年男') → zero_shot_spk_id方式
        log_id:   日志记录ID（批量模式下传入，单句模式自动用文件名）
    """
    prompt_text = prompt_text or PROMPT_TEXT
    prompt_wav = prompt_wav or PROMPT_WAV

    # ── 解析参考音频 (优先级: voice_ref文件 > voice_id注册 > 默认) ──
    ref_wav = prompt_wav
    if voice_ref and os.path.exists(voice_ref):
        ref_wav = voice_ref
    elif voice_id and voice_id in cosyvoice.frontend.spk2info:
        # 从voice_bank解析注册音频路径
        voice_bank_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voice_bank", "voice_bank.json")
        if os.path.exists(voice_bank_file):
            with open(voice_bank_file, "r", encoding="utf-8") as f:
                vb = json.load(f)
            for v in vb:
                if v.get("name") == voice_id or v.get("spk_id") == voice_id:
                    ref_wav = v.get("wav", prompt_wav)
                    break

    t_start = time.time()

    # ── 指令优先级: instruct > emotion > voice_id > voice_ref > 普通合成 ──
    if instruct:
        # 自定义指令模式 (最高优先级) — 用解析出的ref_wav
        gen = cosyvoice.inference_instruct2(
            text, instruct + "<|endofprompt|>", ref_wav, stream=stream
        )
    elif emotion:
        # 情感标签 → 映射到instruct2指令 — 用解析出的ref_wav
        from emotion_control import EMOTION_INSTRUCTIONS
        emo_instruct = EMOTION_INSTRUCTIONS.get(emotion, EMOTION_INSTRUCTIONS.get("calm", ""))
        gen = cosyvoice.inference_instruct2(
            text, emo_instruct + "<|endofprompt|>", ref_wav, stream=stream
        )
    elif voice_id and voice_id in cosyvoice.frontend.spk2info:
        # 使用已注册音色 (zero_shot_spk_id)
        gen = cosyvoice.inference_zero_shot(
            text, "", "", zero_shot_spk_id=voice_id, stream=stream, speed=speed
        )
    elif voice_ref and os.path.exists(voice_ref):
        # 使用指定参考音频文件 (内容未知→cross_lingual，避免prompt_text不匹配导致乱码)
        gen = cosyvoice.inference_cross_lingual(
            text, voice_ref, stream=stream
        )
    elif lang == "en":
        gen = cosyvoice.inference_cross_lingual(
            "<|en|>" + text, prompt_wav, stream=stream
        )
    elif lang == "mix":
        gen = cosyvoice.inference_cross_lingual(
            text, prompt_wav, stream=stream
        )
    else:
        gen = cosyvoice.inference_zero_shot(
            text, prompt_text, prompt_wav, stream=stream, speed=speed
        )

    # 流式模式：逐块写入
    if stream:
        import soundfile as sf
        chunks = []
        with sf.SoundFile(output_path, 'w', cosyvoice.sample_rate, channels=1) as sf_out:
            for chunk in gen:
                audio_chunk = chunk['tts_speech'].squeeze().cpu().numpy()
                sf_out.write(audio_chunk)
                chunks.append(chunk['tts_speech'])
        total_time = time.time() - t_start
        all_audio = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 1)
        duration = all_audio.shape[1] / cosyvoice.sample_rate
        print(f"[流式合成] 首包后 {total_time:.2f}s, 总时长 {duration:.2f}s")
        return output_path, duration, total_time
    else:
        synth_time, output_path, duration = synth_and_save(gen, output_path, cosyvoice.sample_rate)
        total_time = time.time() - t_start

        # ── 写入日志 ──
        _write_log_entry({
            "type": "single",
            "id": log_id or os.path.splitext(os.path.basename(output_path))[0],
            "text": text[:100],
            "output": output_path,
            "lang": lang,
            "speed": speed,
            "emotion": emotion,
            "audio_duration_sec": round(duration, 2),
            "synth_time_sec": round(total_time, 3),
            "rtf": round(total_time / max(duration, 0.01), 3),
        })

        return output_path, duration, synth_time


def batch_synthesize(cosyvoice, dataset_path, output_dir=None, lang="zh", speed=1.0,
                     voice_ref=None, emotion=None, instruct=None):
    """批量合成（从C3/C4的JSON结果）

    Args:
        voice_ref: 参考音频路径
        emotion: 情感标签
        instruct: 自定义instruct2指令
    """
    if output_dir is None:
        output_dir = os.path.join(OUTDIR, "batch")
    os.makedirs(output_dir, exist_ok=True)

    # ── 加载数据（支持单文件JSON / 目录批量JSON / 单条JSON） ──
    items = []
    if os.path.isdir(dataset_path):
        # 目录模式：遍历所有 .json 文件
        json_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.json')])
        for jf in json_files:
            fp = os.path.join(dataset_path, jf)
            try:
                data = json.load(open(fp, encoding="utf-8"))
                if isinstance(data, list):
                    items.extend(data)
                elif isinstance(data, dict):
                    items.append(data)
            except Exception as e:
                print(f"[WARNING] 跳过 {jf}: {e}")
    elif os.path.exists(dataset_path):
        data = json.load(open(dataset_path, encoding="utf-8"))
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
    else:
        print(f"[WARNING] 数据集不存在: {dataset_path}")
        return []

    if not items:
        print("[WARNING] 没有找到有效数据")
        return []

    results = []
    t_batch = time.time()

    for i, r in enumerate(items):
        # ── 文本字段提取（按优先级：C3/C4 多字段兼容） ──
        text = (r.get("translation_zh") or
                r.get("e2e_translation_zh") or
                r.get("c3_cascade_translation_zh") or
                r.get("text_output") or
                r.get("label_text") or
                r.get("reference_en") or
                r.get("text"))
        if not text or not text.strip():
            continue
        # 清理空格
        text = " ".join(text.split())

        # ── ID 提取 ──
        item_id = r.get("id") or os.path.splitext(os.path.basename(
            r.get("audio", r.get("input_audio", f"item{i}"))))[0]

        # ── 情感字段提取 ──
        emo = r.get("emotion", r.get("label_emotion", emotion))
        VALID_EMOTIONS = {"happy", "sad", "angry", "gentle", "calm", "excited", "tired"}
        if emo and emo.lower() in VALID_EMOTIONS:
            emo = emo.lower()
        elif emo and (("ANG" in emo.upper()) or ("angry" in str(item_id).lower())):
            emo = "angry"
        elif emo and ("HAP" in emo.upper()):
            emo = "happy"
        elif emo and ("SAD" in emo.upper()):
            emo = "sad"
        else:
            emo = emotion  # 不覆盖外部传入的 emotion，API 层面由用户控制

        out_path = os.path.join(output_dir, f"s2s_{item_id}.wav")
        _, duration, synth_time = basic_synthesize(
            cosyvoice, text, out_path, lang=lang, speed=speed,
            voice_ref=voice_ref, emotion=emo, instruct=instruct,
            log_id=item_id,
        )
        results.append({
            "id": item_id, "text": text[:60],
            "emotion": emo,
            "wav": os.path.basename(out_path),
            "audio_sec": round(duration, 2),
            "synth_sec": round(synth_time, 2),
        })
        emo_tag = f"[{emo}]" if emo and emo in VALID_EMOTIONS else ""
        if len(text) > 30:
            snippet = text[:30] + "..."
        else:
            snippet = text
        print(f"[{item_id}] {emo_tag} '{snippet}' -> {duration:.2f}s音频/{synth_time:.2f}s")

    batch_time = time.time() - t_batch
    print(f"\n批量合成 {len(results)} 条，总耗时 {batch_time:.1f}s，"
          f"平均 {batch_time / max(len(results), 1):.2f}s/条")

    # ── 写入批量汇总日志 ──
    _write_log_entry({
        "type": "batch_summary",
        "total_items": len(results),
        "total_time_sec": round(batch_time, 1),
        "avg_time_sec": round(batch_time / max(len(results), 1), 3),
        "output_dir": output_dir,
    })

    # 保存汇总JSON
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "total_time": round(batch_time, 1)}, f,
                  ensure_ascii=False, indent=2)
    return results


# ── 进阶功能入口 ──────────────────────────────────────────────
def run_multilingual_test(cosyvoice):
    """进阶①：多语种支持验证"""
    from multilingual import run_multilingual_test as _run
    output_dir = os.path.join(OUTDIR, "multilingual")
    _run(cosyvoice, PROMPT_WAV, PROMPT_TEXT, output_dir, repeat=1)


def run_speed_test(cosyvoice):
    """进阶②：语速控制测试"""
    from speed_timbre import run_speed_test as _run
    output_dir = os.path.join(OUTDIR, "speed_timbre")
    _run(cosyvoice, PROMPT_WAV, PROMPT_TEXT, output_dir)


def run_timbre_test(cosyvoice, ref_audio_dir=None):
    """进阶②：音色控制测试"""
    from speed_timbre import run_timbre_test as _run
    output_dir = os.path.join(OUTDIR, "speed_timbre")
    if ref_audio_dir is None:
        ref_audio_dir = AUDIO_DATA_DIR if os.path.isdir(AUDIO_DATA_DIR) else None
    _run(cosyvoice, PROMPT_WAV, PROMPT_TEXT, ref_audio_dir, output_dir)


def run_emotion_test(cosyvoice):
    """进阶③：情感控制验证"""
    from emotion_control import run_emotion_test as _run
    output_dir = os.path.join(OUTDIR, "emotion")
    _run(cosyvoice, PROMPT_WAV, output_dir, repeat=1)


def run_emotion_intensity_test(cosyvoice):
    """进阶③：情感强度梯度测试"""
    from emotion_control import run_intensity_test as _run
    output_dir = os.path.join(OUTDIR, "emotion")
    _run(cosyvoice, PROMPT_WAV, output_dir)


def run_speaker_clone_test(cosyvoice, ref_audio_dir=None):
    """进阶④：说话人控制（Zero-Shot克隆）"""
    from speaker_clone import SpeakerCloneTester
    output_dir = os.path.join(OUTDIR, "speaker_clone")
    tester = SpeakerCloneTester(cosyvoice, PROMPT_WAV, PROMPT_TEXT)

    # 时长影响测试
    tester.test_duration_impact(PROMPT_WAV, PROMPT_TEXT, output_dir=output_dir)

    # 多说话人
    if ref_audio_dir and os.path.isdir(ref_audio_dir):
        audio_files = [f for f in os.listdir(ref_audio_dir) if f.endswith(('.wav', '.mp3'))]
        if audio_files:
            ref_wavs = {}
            for af in audio_files[:8]:
                name = os.path.splitext(af)[0]
                ref_wavs[name] = os.path.join(ref_audio_dir, af)
            tester.test_cross_gender(ref_wavs, output_dir=output_dir)


def run_benchmark(cosyvoice):
    """进阶⑤：不同TTS模型横向对比"""
    from benchmark import run_benchmark as _run
    output_dir = os.path.join(OUTDIR, "benchmark")
    _run(cosyvoice, PROMPT_WAV, PROMPT_TEXT, output_dir)


def run_all_advanced(cosyvoice, ref_audio_dir=None):
    """一键运行全部进阶测试"""
    print("=" * 70)
    print("C5_TTS 全量进阶测试")
    print("=" * 70)

    # ① 多语种
    print("\n\n### 进阶① 多语种支持验证 ###")
    try:
        run_multilingual_test(cosyvoice)
    except Exception as e:
        print(f"[ERROR] 进阶①失败: {e}")

    # ② 语速
    print("\n\n### 进阶② 语速控制 ###")
    try:
        run_speed_test(cosyvoice)
    except Exception as e:
        print(f"[ERROR] 进阶②-语速失败: {e}")

    # ② 音色
    print("\n\n### 进阶② 音色控制 ###")
    try:
        run_timbre_test(cosyvoice, ref_audio_dir)
    except Exception as e:
        print(f"[ERROR] 进阶②-音色失败: {e}")

    # ③ 情感
    print("\n\n### 进阶③ 情感控制 ###")
    try:
        run_emotion_test(cosyvoice)
    except Exception as e:
        print(f"[ERROR] 进阶③失败: {e}")

    # ④ 说话人
    print("\n\n### 进阶④ 说话人克隆 ###")
    try:
        run_speaker_clone_test(cosyvoice, ref_audio_dir)
    except Exception as e:
        print(f"[ERROR] 进阶④失败: {e}")

    # ⑤ 横向对比
    print("\n\n### 进阶⑤ 横向对比 ###")
    try:
        run_benchmark(cosyvoice)
    except Exception as e:
        print(f"[ERROR] 进阶⑤失败: {e}")

    print("\n" + "=" * 70)
    print(f"全部测试完成！输出目录: {os.path.abspath(OUTDIR)}")
    print("=" * 70)


# ── CLI 主入口 ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=f"C5_TTS v{__version__} — CosyVoice2-0.5B 语音合成 (基础+进阶)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础单句合成
  python c5_tts.py --text "你好世界" -o output.wav

  # 批量合成 (C3级联S2S)
  python c5_tts.py --dataset ../../C3_cascade/outputs/cascade_results.json

  # 语速调节
  python c5_tts.py --text "你好" --speed 1.5 -o fast.wav

  # 流式合成
  python c5_tts.py --text "你好" --stream -o stream.wav

  # 进阶① 多语种
  python c5_tts.py --multilingual-test

  # 进阶② 语速+音色
  python c5_tts.py --speed-test
  python c5_tts.py --timbre-test

  # 进阶③ 情感
  python c5_tts.py --emotion-test

  # 进阶④ 说话人克隆
  python c5_tts.py --speaker-test

  # 进阶⑤ 横向对比
  python c5_tts.py --benchmark

  # 一键全部进阶
  python c5_tts.py --all-advanced
        """,
    )

    # ── 基础参数 ──────────────────────────────────────────────
    ap.add_argument("--text", default=None, help="直接输入合成文本（单句模式）")
    ap.add_argument("--dataset", default=C3_DATASET_DEFAULT,
                    help="C3/C4 JSON结果路径 (默认: C3级联翻译结果)")
    ap.add_argument("-o", "--output", default=None, help="输出WAV路径")
    ap.add_argument("--speed", type=float, default=1.0, help="语速倍率 (0.5~2.0)")
    ap.add_argument("--lang", default="zh", choices=["zh", "en", "mix"],
                    help="语言模式")
    ap.add_argument("--voice", "--voice-ref", dest="voice_ref", default=None,
                    help="参考音频路径 (音色克隆)")
    ap.add_argument("--voice-id", default=None,
                    help="已注册音色ID (如 '中年男')，用 --list-voices 查看")
    ap.add_argument("--stream", action="store_true", help="流式合成")
    ap.add_argument("--emotion", default=None,
                    choices=["happy", "sad", "angry", "gentle", "calm", "excited", "tired"],
                    help="情感标签 (happy/sad/angry/gentle/calm/excited/tired)")
    ap.add_argument("--instruct", default=None,
                    help="自定义instruct2指令 (如 '用悲伤的中年男声朗读')")

    # ── 进阶功能开关 ──────────────────────────────────────────
    ap.add_argument("--multilingual-test", action="store_true",
                    help="进阶①: 多语种支持验证")
    ap.add_argument("--speed-test", action="store_true",
                    help="进阶②: 语速控制测试")
    ap.add_argument("--timbre-test", action="store_true",
                    help="进阶②: 音色控制测试")
    ap.add_argument("--emotion-test", action="store_true",
                    help="进阶③: 情感控制验证")
    ap.add_argument("--emotion-intensity", action="store_true",
                    help="进阶③: 情感强度梯度测试")
    ap.add_argument("--speaker-test", action="store_true",
                    help="进阶④: 说话人克隆测试")
    ap.add_argument("--benchmark", action="store_true",
                    help="进阶⑤: TTS模型横向对比")
    ap.add_argument("--all-advanced", action="store_true",
                    help="一键运行全部进阶测试")
    ap.add_argument("--ref-audio-dir", default=None,
                    help="参考音频目录 (用于音色/说话人测试)")

    # ── 音色库管理 ────────────────────────────────────────────
    ap.add_argument("--list-voices", action="store_true", help="列出已注册音色")
    ap.add_argument("--register-voice", default=None, help="注册新音色名称")
    ap.add_argument("--register-wav", default=None, help="注册音色的参考音频")
    ap.add_argument("--register-text", default="", help="注册音色的参考文本")

    args = ap.parse_args()

    # 加载模型
    cosyvoice = load_model()

    # ── 音色库管理 ────────────────────────────────────────────
    if args.list_voices:
        voices = list(cosyvoice.frontend.spk2info.keys())
        print(f"已注册音色 ({len(voices)}): {voices}")
        return

    if args.register_voice and args.register_wav:
        from speed_timbre import TimbreController
        tc = TimbreController(cosyvoice)
        tc.register_voice(args.register_voice, args.register_wav, args.register_text)
        print(f"[OK] 音色已注册: {args.register_voice}")
        return

    # ── 进阶功能分支 ──────────────────────────────────────────
    ref_audio_dir = args.ref_audio_dir
    if ref_audio_dir is None and os.path.isdir(AUDIO_DATA_DIR):
        ref_audio_dir = AUDIO_DATA_DIR

    if args.all_advanced:
        run_all_advanced(cosyvoice, ref_audio_dir)
        return

    if args.multilingual_test:
        run_multilingual_test(cosyvoice)

    if args.speed_test:
        run_speed_test(cosyvoice)

    if args.timbre_test:
        run_timbre_test(cosyvoice, ref_audio_dir)

    if args.emotion_test:
        run_emotion_test(cosyvoice)

    if args.emotion_intensity:
        run_emotion_intensity_test(cosyvoice)

    if args.speaker_test:
        run_speaker_clone_test(cosyvoice, ref_audio_dir)

    if args.benchmark:
        run_benchmark(cosyvoice)

    # 如果指定了进阶测试，跳过基础合成
    advanced_flags = [
        args.multilingual_test, args.speed_test, args.timbre_test,
        args.emotion_test, args.emotion_intensity,
        args.speaker_test, args.benchmark,
    ]
    if any(advanced_flags):
        print(f"\n所有进阶测试完成，输出目录: {os.path.abspath(OUTDIR)}")
        return

    # ── 基础合成 ──────────────────────────────────────────────
    if args.text:
        # 单句模式
        output_path = args.output or os.path.join(OUTDIR, "output.wav")
        result_path, duration, synth_time = basic_synthesize(
            cosyvoice, args.text, output_path,
            lang=args.lang, speed=args.speed,
            voice_ref=args.voice_ref, stream=args.stream,
            emotion=args.emotion, instruct=args.instruct,
            voice_id=args.voice_id,
        )
        print(f"[OK] {os.path.basename(result_path)} "
              f"({duration:.2f}s音频 / {synth_time:.2f}s合成)")
    elif args.dataset and (os.path.exists(args.dataset) or os.path.isdir(args.dataset)):
        # 批量模式
        batch_output_dir = args.output or os.path.join(OUTDIR, "batch")
        batch_synthesize(cosyvoice, args.dataset, output_dir=batch_output_dir,
                         lang=args.lang, speed=args.speed)

        # 同时测试两种语言（基础需求③）
        print(f"\n{'='*60}")
        print("基础需求③: 测试两种语言 (中文 + 英文)")
        print(f"{'='*60}")
        zh_out = os.path.join(OUTDIR, "lang_zh.wav")
        _, dur, t = basic_synthesize(cosyvoice, "今天天气很好，我们一起去公园散步吧。", zh_out)
        print(f"[中文] -> zh.wav ({dur:.2f}s音频 / {t:.2f}s合成)")

        en_out = os.path.join(OUTDIR, "lang_en.wav")
        _, dur, t = basic_synthesize(cosyvoice, "The weather is really nice today.", en_out, lang="en")
        print(f"[英文] -> en.wav ({dur:.2f}s音频 / {t:.2f}s合成)")
    else:
        # 无输入：显示帮助
        ap.print_help()
        print(f"\n[C5_TTS v{__version__}] 提示: 使用 --text 单句合成，或 --dataset 批量合成")

    print(f"\n输出目录: {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()