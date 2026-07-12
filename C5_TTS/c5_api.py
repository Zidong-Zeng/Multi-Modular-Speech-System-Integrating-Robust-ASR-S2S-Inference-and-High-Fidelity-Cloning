# -*- coding: utf-8 -*-
"""
C5_TTS 统一对外API接口模块
=============================
为组内其他模块(C1~C4)和外部系统提供标准化的TTS调用接口

模块定位：
  C5是整套多模态语音大模型实训闭环的输出收尾模块，依托TTS技术将文本转为可播放语音。
  本模块提供统一的API接口，供其他模块通过函数调用、JSON输入输出或命令行方式集成。

接口设计原则：
  1. 最小依赖：仅依赖 c5_tts.py 中的 basic_synthesize / batch_synthesize / load_model
  2. 明确输入输出：每个接口都有清晰的输入参数和返回值结构
  3. 容错健壮：对异常输入有兜底处理，返回标准化的错误信息
  4. 可独立运行：每个接口均可通过命令行参数独立调用

=============================
接口清单
=============================
  接口1: synthesize()           — 单句文本→语音文件（基础合成，JSON输入输出可选）
  接口2: batch_synthesize()     — 批量文本→语音文件（对接C3/C4输出）
  接口3: synthesize_stream()    — 流式合成（边生成边返回音频块）
  接口4: list_voices()          — 列出可用音色/说话人
  接口5: get_module_info()      — 返回模块元信息（版本、能力、模型说明）
  接口6: generate_voice_with_control() — 高级合成（情感+语速+音色+说话人一站式）
  接口7: run_pipeline_from_json() — JSON文件驱动的批量管线（对接上游模块）

=============================
预留接口（供后续扩展）
=============================
  预留接口A: register_custom_voice()    — 动态注册自定义音色（预计v3.1）
  预留接口B: fine_tune_voice()          — 微调音色模型（预计v3.2，需额外训练脚本）
  预留接口C: voice_conversion()         — 语音转换（音色A→音色B，预计v4.0）
  预留接口D: text_frontend_normalize()  — 文本前端归一化接口（预计v3.1）
  预留接口E: streaming_websocket()      — WebSocket流式推送接口（预计v3.2）

=============================
使用示例
=============================
  # 方式1: Python函数调用
  from c5_api import C5TTSAPI
  api = C5TTSAPI()
  result = api.synthesize("你好世界", output_path="hello.wav")

  # 方式2: 命令行JSON模式
  echo '{"text": "你好世界", "output": "hello.wav"}' | python c5_api.py --json-stdin

  # 方式3: 作为模块被C3/C4调用
  from c5_api import synthesize_and_return_path
  wav_path = synthesize_and_return_path(translation_text, lang="zh")

=============================
版本历史
=============================
  v3.0.0 (2025-07): 初始对外API版本，封装基础+进阶全部能力
"""

import os, sys, json, time, argparse
from typing import Dict, List, Optional, Union, Generator, Any

# ── 将当前目录和CosyVoice路径加入sys.path ──────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
COSYVOICE_REPO = os.environ.get("COSYVOICE_REPO", "/data/haoaokai/zbs/CosyVoice")
MODEL_DIR = os.environ.get("COSYVOICE_MODEL", "/data/haoaokai/zbs/CosyVoice2-0.5B")

if os.path.isdir(COSYVOICE_REPO):
    sys.path.insert(0, COSYVOICE_REPO)
    sys.path.insert(0, os.path.join(COSYVOICE_REPO, "third_party/Matcha-TTS"))

__version__ = "3.0.0"
__author__ = "C5_TTS Team"
__module_name__ = "C5_TTS_API"


# ================================================================
# 接口数据结构定义
# ================================================================

class SynthesizeRequest:
    """接口1: 单句合成请求参数

    Attributes:
        text:         待合成文本（必填，支持中/英/日/粤/韩/混合）
        output:       输出WAV路径（选填，默认自动生成）
        lang:         语言模式 zh/en/ja/yue/ko/mix（默认zh）
        speed:        语速倍率 0.5~2.0（默认1.0）
        emotion:      情感标签 happy/sad/angry/gentle/calm/excited/tired
        instruct:     自定义instruct2指令（优先级高于emotion）
        voice_ref:    参考音频路径（音色克隆）
        voice_id:     已注册音色ID
        stream:       是否流式输出
    """
    def __init__(self, text: str, output: str = None, lang: str = "zh",
                 speed: float = 1.0, emotion: str = None, instruct: str = None,
                 voice_ref: str = None, voice_id: str = None, stream: bool = False):
        self.text = text
        self.output = output
        self.lang = lang
        self.speed = speed
        self.emotion = emotion
        self.instruct = instruct
        self.voice_ref = voice_ref
        self.voice_id = voice_id
        self.stream = stream

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class SynthesizeResponse:
    """接口1: 单句合成响应

    Attributes:
        success:      是否成功
        wav_path:     输出音频文件路径
        audio_duration_sec: 音频时长(秒)
        synth_time_sec:    合成耗时(秒)
        rtf:          实时率 (synth_time / audio_duration)
        error:        错误信息（仅失败时）
        request:      原始请求参数回显
    """
    def __init__(self, success: bool = True, wav_path: str = "",
                 audio_duration_sec: float = 0, synth_time_sec: float = 0,
                 error: str = "", request: Dict = None):
        self.success = success
        self.wav_path = wav_path
        self.audio_duration_sec = audio_duration_sec
        self.synth_time_sec = synth_time_sec
        self.rtf = synth_time_sec / max(audio_duration_sec, 0.01)
        self.error = error
        self.request = request or {}

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "wav_path": self.wav_path,
            "audio_duration_sec": round(self.audio_duration_sec, 2),
            "synth_time_sec": round(self.synth_time_sec, 2),
            "rtf": round(self.rtf, 3),
            "error": self.error,
            "request": self.request,
        }


class BatchSynthesizeRequest:
    """接口2: 批量合成请求参数

    Attributes:
        dataset_path: JSON文件路径（含 translation_zh / e2e_translation_zh / text 字段）
        output_dir:   输出目录（选填，默认 outputs/batch/）
        lang:         语言模式
        speed:        语速倍率
        voice_ref:    参考音频路径
        emotion:      情感标签
        instruct:     自定义指令
    """
    def __init__(self, dataset_path: str, output_dir: str = None,
                 lang: str = "zh", speed: float = 1.0,
                 voice_ref: str = None, emotion: str = None, instruct: str = None):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.lang = lang
        self.speed = speed
        self.voice_ref = voice_ref
        self.emotion = emotion
        self.instruct = instruct


class BatchSynthesizeResponse:
    """接口2: 批量合成响应

    Attributes:
        success:       是否成功
        total_items:   总条目数
        total_time_sec: 批量总耗时
        avg_time_sec:  平均每条耗时
        results:       每条结果列表 [{"id":..., "wav":..., "audio_sec":..., "synth_sec":...}]
        error:         错误信息
    """
    def __init__(self, success: bool = True, total_items: int = 0,
                 total_time_sec: float = 0, results: List[Dict] = None, error: str = ""):
        self.success = success
        self.total_items = total_items
        self.total_time_sec = total_time_sec
        self.avg_time_sec = total_time_sec / max(total_items, 1)
        self.results = results or []
        self.error = error

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "total_items": self.total_items,
            "total_time_sec": round(self.total_time_sec, 2),
            "avg_time_sec": round(self.avg_time_sec, 3),
            "results": self.results,
            "error": self.error,
        }


class ModuleInfo:
    """接口5: 模块元信息

    Attributes:
        module_name:      模块名称
        version:          版本号
        model:            底层模型名称
        sample_rate:      音频采样率(Hz)
        supported_languages:  支持的语言列表
        supported_emotions:   支持的情感列表
        supported_speed_range: 语速范围
        capabilities:     能力清单（中文描述）
        reserved_interfaces:  预留接口说明
        input_format:     支持的输入格式
        output_format:    输出格式
    """
    def __init__(self, sample_rate: int = 24000):
        self.module_name = __module_name__
        self.version = __version__
        self.model = "CosyVoice2-0.5B"
        self.sample_rate = sample_rate
        self.supported_languages = ["zh", "en", "ja", "yue", "ko", "mix"]
        self.supported_emotions = ["happy", "sad", "angry", "gentle", "calm", "excited", "tired"]
        self.supported_speed_range = "0.5 ~ 2.0"
        self.capabilities = {
            "基础合成": "单句文本→WAV语音文件，支持中英双语",
            "批量合成": "批量文本→批量语音（对接C3/C4输出JSON）",
            "多语种": "支持中文/英文/日语/粤语/韩语/中英混合",
            "语速控制": "0.5x~2.0x原生调速，5档预设",
            "音色控制": "通过参考音频实现音色切换，支持音色库注册",
            "情感控制": "7种情感(happy/sad/angry/gentle/calm/excited/tired)，支持强度梯度",
            "说话人克隆": "Zero-Shot说话人音色复刻，支持跨性别",
            "流式合成": "边生成边输出音频块，适配实时对话场景",
            "实验评估": "RTF实时率/声学特征F0/可视化图表",
        }
        self.reserved_interfaces = {
            "register_custom_voice": "动态注册自定义音色（预计v3.1）",
            "fine_tune_voice": "微调音色模型（预计v3.2，需额外训练脚本）",
            "voice_conversion": "语音转换：音色A→音色B（预计v4.0）",
            "text_frontend_normalize": "文本前端归一化接口（预计v3.1）",
            "streaming_websocket": "WebSocket流式推送接口（预计v3.2）",
        }
        self.input_format = {
            "function_call": "Python函数调用，传入文本/参数",
            "json_stdin": "通过stdin传入JSON请求",
            "command_line": "命令行参数",
            "json_file": "JSON文件路径（批量模式）",
        }
        self.output_format = {
            "audio": "24kHz WAV单声道音频文件",
            "json": "JSON格式结果（含路径、耗时、RTF）",
            "stream": "流式音频块生成器",
        }

    def to_dict(self) -> Dict:
        return {
            "module_name": self.module_name,
            "version": self.version,
            "model": self.model,
            "sample_rate": self.sample_rate,
            "supported_languages": self.supported_languages,
            "supported_emotions": self.supported_emotions,
            "supported_speed_range": self.supported_speed_range,
            "capabilities": self.capabilities,
            "reserved_interfaces": self.reserved_interfaces,
            "input_format": self.input_format,
            "output_format": self.output_format,
        }


# ================================================================
# 核心API类
# ================================================================

class C5TTSAPI:
    """C5_TTS统一对外API

    封装CosyVoice2-0.5B的全部能力，提供标准化接口供组内其他模块调用。

    使用方式:
        api = C5TTSAPI()                          # 自动加载模型
        api = C5TTSAPI(model_dir="/path/to/model") # 指定模型路径
        result = api.synthesize("你好世界")
        print(result.to_dict())

    注意:
        模型加载耗时约27秒（GPU），请在模块初始化时一次性加载。
        显存占用约2-3GB，确保GPU有充足显存。
        采样率固定为24000Hz，输出为单声道WAV。
    """

    def __init__(self, model_dir: str = None, cosyvoice_repo: str = None,
                 verbose: bool = True):
        """
        Args:
            model_dir:      CosyVoice2-0.5B权重目录路径。
                            默认从环境变量COSYVOICE_MODEL读取。
            cosyvoice_repo: CosyVoice官方仓库路径。
                            默认从环境变量COSYVOICE_REPO读取。
            verbose:        是否打印加载日志
        """
        self.model_dir = model_dir or MODEL_DIR
        self.cosyvoice_repo = cosyvoice_repo or COSYVOICE_REPO
        self.verbose = verbose
        self._model = None
        self._prompt_wav = None
        self._prompt_text = None
        self._sample_rate = 24000

        # 懒加载：首次调用任意接口时才加载模型
        self._loaded = False

    # ── 模型加载 ──────────────────────────────────────────────
    def _ensure_loaded(self):
        """确保模型已加载（懒加载模式）"""
        if self._loaded:
            return

        if self.verbose:
            print(f"[C5_API] 加载模型: {self.model_dir}")

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"模型目录不存在: {self.model_dir}\n"
                f"请设置环境变量 COSYVOICE_MODEL 或传入 model_dir 参数"
            )

        # 动态导入c5_tts中的load_model
        from c5_tts import load_model, PROMPT_WAV, PROMPT_TEXT
        self._model = load_model(self.model_dir, verbose=self.verbose)
        self._prompt_wav = PROMPT_WAV
        self._prompt_text = PROMPT_TEXT
        self._sample_rate = self._model.sample_rate
        self._loaded = True

        if self.verbose:
            print(f"[C5_API] 模型就绪，采样率={self._sample_rate}Hz")

    # ════════════════════════════════════════════════════════════
    # 接口1: synthesize() — 单句文本→语音
    # ════════════════════════════════════════════════════════════
    def synthesize(self, text: str, output: str = None, lang: str = "zh",
                   speed: float = 1.0, emotion: str = None,
                   instruct: str = None, voice_ref: str = None,
                   voice_id: str = None, stream: bool = False) -> SynthesizeResponse:
        """单句文本语音合成

        本接口是C5_TTS最核心的基础合成接口，接收文本并输出WAV音频文件。
        支持情感、语速、音色、说话人等全部进阶控制参数。

        Args:
            text:     待合成文本。支持中/英/日/粤/韩/中英混合。
                      示例: "今天天气真好" / "Hello world" / "今日の天気は良い"
            output:   输出WAV路径。默认自动生成到 outputs/ 目录。
            lang:     语言模式。zh(中文)/en(英文)/ja(日语)/yue(粤语)/ko(韩语)/mix(混合)
            speed:    语速倍率。0.5(半速) ~ 2.0(双倍速)，推荐1.0。
            emotion:  情感标签。happy/sad/angry/gentle/calm/excited/tired
            instruct: 自定义instruct2指令文本。优先级高于emotion。
                      示例: "用悲伤的中年男声朗读，语速缓慢"
            voice_ref: 参考音频路径（用于音色克隆）。
                       提供3秒以上清晰人声，说话人音色将被复刻到合成语音。
            voice_id:  已注册音色ID（通过register_voice预先注册）。
                       使用 --list-voices 查看可用音色列表。
            stream:    是否启用流式合成模式。

        Returns:
            SynthesizeResponse: 包含 success, wav_path, audio_duration_sec,
                               synth_time_sec, rtf 等字段

        Raises:
            ValueError:  文本为空
            FileNotFoundError: 模型未加载成功

        Example:
            >>> api = C5TTSAPI()
            >>> resp = api.synthesize("你好世界")
            >>> print(f"音频: {resp.wav_path}, RTF: {resp.rtf}")

            >>> resp = api.synthesize("太棒了", emotion="happy")
            >>> resp = api.synthesize("Hello", lang="en", speed=1.5)
        """
        if not text or not text.strip():
            return SynthesizeResponse(
                success=False, error="输入文本为空",
                request={"text": text, "lang": lang}
            )

        self._ensure_loaded()

        # 自动生成输出路径
        if output is None:
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            output = os.path.join(
                os.path.dirname(HERE), "outputs", "api",
                f"synth_{lang}_{text_hash}.wav"
            )
        os.makedirs(os.path.dirname(output), exist_ok=True)

        try:
            from c5_tts import basic_synthesize
            wav_path, audio_dur, synth_time = basic_synthesize(
                self._model, text, output,
                lang=lang, speed=speed,
                voice_ref=voice_ref,
                prompt_text=self._prompt_text,
                prompt_wav=self._prompt_wav,
                emotion=emotion, instruct=instruct,
                voice_id=voice_id, stream=stream,
            )
            return SynthesizeResponse(
                success=True,
                wav_path=wav_path,
                audio_duration_sec=audio_dur,
                synth_time_sec=synth_time,
                request={"text": text[:50], "lang": lang, "speed": speed,
                         "emotion": emotion, "voice_id": voice_id},
            )
        except Exception as e:
            return SynthesizeResponse(
                success=False,
                error=str(e),
                request={"text": text[:50], "lang": lang},
            )

    # ════════════════════════════════════════════════════════════
    # 接口2: batch_synthesize() — 批量文本→语音
    # ════════════════════════════════════════════════════════════
    def batch_synthesize(self, dataset_path: str, output_dir: str = None,
                         lang: str = "zh", speed: float = 1.0,
                         voice_ref: str = None, emotion: str = None,
                         instruct: str = None) -> BatchSynthesizeResponse:
        """批量文本语音合成（对接C3/C4输出JSON）

        读取上游模块(C3级联翻译 / C4端到端翻译)的JSON输出文件，
        逐条提取文本并合成为语音，生成 s2s_{id}.wav 文件。

        支持的JSON字段（自动识别优先级）:
            translation_zh      — C3级联翻译中文结果（最高优先级）
            e2e_translation_zh  — C4端到端翻译中文结果
            text                — 通用文本字段

        Args:
            dataset_path: JSON文件路径。格式为 [{"id":..., "translation_zh":..., ...}, ...]
            output_dir:   输出目录。默认 outputs/batch/
            lang:         语言模式
            speed:        语速倍率
            voice_ref:    统一参考音频路径（所有条目共用同一音色）
            emotion:      统一情感标签
            instruct:     统一自定义指令

        Returns:
            BatchSynthesizeResponse: 包含 total_items, total_time_sec,
                                    avg_time_sec, results 等字段

        Example:
            >>> api = C5TTSAPI()
            >>> resp = api.batch_synthesize(
            ...     "../../C3_cascade/outputs/cascade_results.json",
            ...     emotion="gentle"
            ... )
            >>> print(f"合成{resp.total_items}条，平均{resp.avg_time_sec:.2f}s/条")
        """
        self._ensure_loaded()

        if not os.path.exists(dataset_path):
            return BatchSynthesizeResponse(
                success=False,
                error=f"数据集文件不存在: {dataset_path}"
            )

        try:
            from c5_tts import batch_synthesize as _batch_synth
            results = _batch_synth(
                self._model, dataset_path,
                output_dir=output_dir,
                lang=lang, speed=speed,
                voice_ref=voice_ref,
                emotion=emotion, instruct=instruct,
            )

            total_time = sum(r.get("synth_sec", 0) for r in results)
            return BatchSynthesizeResponse(
                success=True,
                total_items=len(results),
                total_time_sec=total_time,
                results=results,
            )
        except Exception as e:
            return BatchSynthesizeResponse(
                success=False, error=str(e)
            )

    # ════════════════════════════════════════════════════════════
    # 接口3: synthesize_stream() — 流式合成
    # ════════════════════════════════════════════════════════════
    def synthesize_stream(self, text: str, lang: str = "zh",
                          emotion: str = None, voice_ref: str = None
                          ) -> Generator[bytes, None, None]:
        """流式语音合成（边生成边返回音频块）

        适用于实时对话场景，逐块返回PCM音频数据。
        调用方可边接收边播放，降低首字延迟。

        Args:
            text:      待合成文本
            lang:      语言模式
            emotion:   情感标签
            voice_ref: 参考音频路径

        Yields:
            bytes: 每个音频块的PCM数据（16-bit, mono, 24000Hz）

        Example:
            >>> api = C5TTSAPI()
            >>> for chunk in api.synthesize_stream("你好世界欢迎来到AI时代"):
            ...     audio_player.feed(chunk)  # 实时播放
        """
        self._ensure_loaded()
        import numpy as np

        try:
            from c5_tts import basic_synthesize

            # 使用临时文件作为流式中转
            import tempfile
            tmp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

            basic_synthesize(
                self._model, text, tmp_path,
                lang=lang, emotion=emotion,
                voice_ref=voice_ref,
                stream=True,
            )

            # 读取并逐块yield
            import wave
            with wave.open(tmp_path, 'rb') as wf:
                chunk_size = 1024 * 4  # 4KB per chunk
                while True:
                    data = wf.readframes(chunk_size)
                    if not data:
                        break
                    yield data

            os.unlink(tmp_path)
        except Exception as e:
            raise RuntimeError(f"流式合成失败: {e}")

    # ════════════════════════════════════════════════════════════
    # 接口4: list_voices() — 列出可用音色
    # ════════════════════════════════════════════════════════════
    def list_voices(self) -> List[Dict]:
        """列出所有可用的音色/说话人

        包括：
          1. CosyVoice模型中预注册的说话人（spk2info）
          2. voice_bank/voice_bank.json中用户自定义注册的音色

        Returns:
            [{"name": "音色名称", "spk_id": "说话人ID", "source": "model/voice_bank"}, ...]

        Example:
            >>> api = C5TTSAPI()
            >>> voices = api.list_voices()
            >>> for v in voices:
            ...     print(f"{v['name']} (来源: {v['source']})")
        """
        self._ensure_loaded()

        voices = []

        # 1. 模型内置说话人
        try:
            for spk_id in self._model.frontend.spk2info.keys():
                voices.append({
                    "name": spk_id,
                    "spk_id": spk_id,
                    "source": "model",
                })
        except Exception:
            pass

        # 2. voice_bank自定义音色
        voice_bank_path = os.path.join(
            os.path.dirname(HERE), "voice_bank", "voice_bank.json"
        )
        if os.path.exists(voice_bank_path):
            try:
                with open(voice_bank_path, "r", encoding="utf-8") as f:
                    vb = json.load(f)
                for v in vb:
                    voices.append({
                        "name": v.get("name", "unknown"),
                        "spk_id": v.get("spk_id", v.get("name", "")),
                        "source": "voice_bank",
                        "wav": v.get("wav", ""),
                    })
            except Exception:
                pass

        return voices

    # ════════════════════════════════════════════════════════════
    # 接口5: get_module_info() — 模块元信息
    # ════════════════════════════════════════════════════════════
    def get_module_info(self) -> ModuleInfo:
        """返回C5_TTS模块的完整元信息

        包括版本、能力清单、支持的参数范围、预留接口等。
        供组内其他模块和验收文档使用。

        Returns:
            ModuleInfo: 模块元信息对象，可调用 .to_dict() 转JSON

        Example:
            >>> api = C5TTSAPI()
            >>> info = api.get_module_info()
            >>> print(json.dumps(info.to_dict(), ensure_ascii=False, indent=2))
        """
        self._ensure_loaded()
        return ModuleInfo(sample_rate=self._sample_rate)

    # ════════════════════════════════════════════════════════════
    # 接口6: generate_voice_with_control() — 高级合成
    # ════════════════════════════════════════════════════════════
    def generate_voice_with_control(self, text: str,
                                     output: str = None,
                                     speed: float = 1.0,
                                     emotion: str = None,
                                     voice_ref: str = None,
                                     voice_id: str = None,
                                     instruct: str = None
                                     ) -> SynthesizeResponse:
        """高级合成：一站式语音生成（情感+语速+音色+说话人）

        本接口是 synthesize() 的增强版别名，语义更明确地表达"控制参数合成"。
        为组内C3/C4模块提供最简化调用方式。

        Args:
            text:      待合成文本
            output:    输出路径
            speed:     语速 0.5~2.0
            emotion:   情感标签
            voice_ref: 参考音频路径（音色克隆）
            voice_id:  已注册音色ID
            instruct:  自定义instruct2指令

        Returns:
            SynthesizeResponse

        Example:
            >>> api = C5TTSAPI()
            >>> resp = api.generate_voice_with_control(
            ...     "今天的项目进展顺利",
            ...     emotion="happy", speed=1.2, voice_id="中年男"
            ... )
        """
        return self.synthesize(
            text=text, output=output,
            lang="zh", speed=speed,
            emotion=emotion, instruct=instruct,
            voice_ref=voice_ref, voice_id=voice_id,
        )

    # ════════════════════════════════════════════════════════════
    # 接口7: run_pipeline_from_json() — JSON驱动批量管线
    # ════════════════════════════════════════════════════════════
    def run_pipeline_from_json(self, json_path: str,
                                output_dir: str = None) -> Dict:
        """从JSON文件驱动批量合成管线

        单个JSON文件描述全部合成任务，适合自动化流水线调用。
        JSON格式见下方示例。

        JSON请求格式:
        {
            "pipeline_name": "c3_s2s_pipeline",
            "tasks": [
                {
                    "id": "sample_001",
                    "text": "今天天气很好",
                    "lang": "zh",
                    "speed": 1.0,
                    "emotion": "gentle",
                    "voice_ref": "/path/to/ref.wav",
                    "output": "/path/to/output.wav"
                },
                {
                    "id": "sample_002",
                    "text": "The weather is nice",
                    "lang": "en",
                    "speed": 1.2
                }
            ],
            "global_params": {
                "default_speed": 1.0,
                "default_lang": "zh"
            }
        }

        Args:
            json_path:  JSON管线描述文件路径
            output_dir: 统一输出目录（可被task级别output覆盖）

        Returns:
            {"success": True/False, "total": N, "completed": N,
             "failed": N, "results": [...]}

        Example:
            >>> api = C5TTSAPI()
            >>> result = api.run_pipeline_from_json("pipeline_tasks.json")
            >>> print(f"完成 {result['completed']}/{result['total']} 条")
        """
        self._ensure_loaded()

        if not os.path.exists(json_path):
            return {"success": False, "error": f"文件不存在: {json_path}"}

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            return {"success": False, "error": f"JSON解析失败: {e}"}

        tasks = config.get("tasks", [])
        global_params = config.get("global_params", {})
        default_output_dir = output_dir or os.path.join(
            os.path.dirname(HERE), "outputs", "pipeline"
        )
        os.makedirs(default_output_dir, exist_ok=True)

        results = []
        completed = 0
        failed = 0

        for task in tasks:
            task_id = task.get("id", f"task_{len(results)}")
            text = task.get("text", "")
            if not text:
                results.append({"id": task_id, "success": False, "error": "文本为空"})
                failed += 1
                continue

            output = task.get("output") or os.path.join(
                default_output_dir, f"pipeline_{task_id}.wav"
            )

            resp = self.synthesize(
                text=text,
                output=output,
                lang=task.get("lang", global_params.get("default_lang", "zh")),
                speed=task.get("speed", global_params.get("default_speed", 1.0)),
                emotion=task.get("emotion"),
                voice_ref=task.get("voice_ref"),
                voice_id=task.get("voice_id"),
                instruct=task.get("instruct"),
            )

            results.append({
                "id": task_id,
                "success": resp.success,
                "wav_path": resp.wav_path,
                "audio_duration_sec": resp.audio_duration_sec,
                "synth_time_sec": resp.synth_time_sec,
                "rtf": resp.rtf,
                "error": resp.error,
            })

            if resp.success:
                completed += 1
            else:
                failed += 1

        return {
            "success": failed == 0,
            "pipeline_name": config.get("pipeline_name", "unnamed"),
            "total": len(tasks),
            "completed": completed,
            "failed": failed,
            "results": results,
        }


# ================================================================
# 便捷函数（无状态，直接调用）
# ================================================================

def synthesize_and_return_path(text: str, output: str = None,
                                lang: str = "zh", speed: float = 1.0,
                                **kwargs) -> str:
    """最简调用：合成语音并返回文件路径

    专为C3/C4模块设计的极简接口。只需传入翻译文本，返回WAV文件路径。
    内部自动管理模型加载和生命周期。

    Args:
        text:   待合成文本
        output: 输出路径（可选）
        lang:   语言
        speed:  语速
        **kwargs: 传递给 synthesize() 的其他参数

    Returns:
        str: WAV文件路径

    Raises:
        RuntimeError: 合成失败

    Example:
        >>> # 在C3_cascade中调用
        >>> from C5_TTS.code.c5_api import synthesize_and_return_path
        >>> wav = synthesize_and_return_path(c3_translation, lang="zh")
        >>> print(f"语音文件: {wav}")
    """
    api = C5TTSAPI(verbose=False)
    resp = api.synthesize(text, output=output, lang=lang, speed=speed, **kwargs)
    if not resp.success:
        raise RuntimeError(f"TTS合成失败: {resp.error}")
    return resp.wav_path


# ================================================================
# JSON stdin/stdout 模式（供命令行管道调用）
# ================================================================

def _handle_json_stdin():
    """从stdin读取JSON请求，执行合成，输出JSON结果到stdout"""
    try:
        raw = sys.stdin.read()
        request = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"JSON解析错误: {e}"}))
        return

    api = C5TTSAPI()

    # 判断请求类型
    if "dataset_path" in request:
        # 批量合成请求
        resp = api.batch_synthesize(
            dataset_path=request["dataset_path"],
            output_dir=request.get("output_dir"),
            lang=request.get("lang", "zh"),
            speed=request.get("speed", 1.0),
            voice_ref=request.get("voice_ref"),
            emotion=request.get("emotion"),
        )
        print(json.dumps(resp.to_dict(), ensure_ascii=False, indent=2))
    elif request.get("action") == "list_voices":
        voices = api.list_voices()
        print(json.dumps(voices, ensure_ascii=False, indent=2))
    elif request.get("action") == "module_info":
        info = api.get_module_info()
        print(json.dumps(info.to_dict(), ensure_ascii=False, indent=2))
    elif "text" in request:
        # 单句合成请求
        resp = api.synthesize(
            text=request["text"],
            output=request.get("output"),
            lang=request.get("lang", "zh"),
            speed=request.get("speed", 1.0),
            emotion=request.get("emotion"),
            instruct=request.get("instruct"),
            voice_ref=request.get("voice_ref"),
            voice_id=request.get("voice_id"),
            stream=request.get("stream", False),
        )
        print(json.dumps(resp.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"success": False, "error": "未知请求类型，请包含 'text' 或 'dataset_path' 或 'action' 字段"}))


# ================================================================
# CLI主入口
# ================================================================

def main():
    ap = argparse.ArgumentParser(
        description=f"C5_TTS API v{__version__} — 统一对外TTS接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 函数调用模式（Python）
  python -c "from c5_api import C5TTSAPI; api=C5TTSAPI(); print(api.synthesize('你好').to_dict())"

  # JSON stdin模式（管道调用）
  echo '{"text": "你好世界", "lang": "zh"}' | python c5_api.py --json-stdin

  # 批量JSON驱动管线
  python c5_api.py --pipeline pipeline_tasks.json

  # 查看模块信息
  python c5_api.py --module-info

  # 列出可用音色
  python c5_api.py --list-voices
        """,
    )

    ap.add_argument("--json-stdin", action="store_true",
                    help="从stdin读取JSON请求，执行后输出JSON结果")
    ap.add_argument("--pipeline", default=None,
                    help="JSON管线描述文件路径（接口7）")
    ap.add_argument("--module-info", action="store_true",
                    help="打印模块元信息（接口5）")
    ap.add_argument("--list-voices", action="store_true",
                    help="列出所有可用音色（接口4）")
    ap.add_argument("--model-dir", default=None,
                    help="CosyVoice2模型目录（覆盖环境变量）")

    args = ap.parse_args()

    if args.json_stdin:
        _handle_json_stdin()
        return

    api = C5TTSAPI(model_dir=args.model_dir, verbose=True)

    if args.module_info:
        info = api.get_module_info()
        print(json.dumps(info.to_dict(), ensure_ascii=False, indent=2))
        return

    if args.list_voices:
        voices = api.list_voices()
        print(f"可用音色 ({len(voices)}):")
        for v in voices:
            print(f"  - {v['name']} (来源: {v['source']})")
        return

    if args.pipeline:
        result = api.run_pipeline_from_json(args.pipeline)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 默认显示帮助
    ap.print_help()
    print(f"\n[C5_TTS API v{__version__}] 请指定操作模式，或使用 --json-stdin 管道调用")


if __name__ == "__main__":
    main()