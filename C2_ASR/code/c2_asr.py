# -*- coding: utf-8 -*-
"""
C2 ASR entrypoint with explicit VAD -> chunk -> ASR flow.

Supported engines:
1. transformers pipeline
2. WhisperX with alignment and optional diarization
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
import re
import sys
import tempfile
import time
from typing import Any, Callable, Sequence


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
DEFAULT_PYANNOTE_SEGMENTATION_MODEL = "/root/siton-tmp/assignment_C/model/pyannote-segmentation-3.0"


def _maybe_offline() -> None:
    argv = sys.argv
    offline = "--offline" in argv
    for flag in ("--model", "--asr_model", "--llm"):
        if flag in argv:
            index = argv.index(flag) + 1
            if index < len(argv) and os.path.isdir(argv[index]):
                offline = True
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


_maybe_offline()

HERE = os.path.dirname(os.path.abspath(__file__))


def normalize(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(left: list[str], right: list[str]) -> int:
    prev = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        curr = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    ref_words = [word for text in refs for word in normalize(text).split()]
    hyp_words = [word for text in hyps for word in normalize(text).split()]
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def compute_cer(refs: list[str], hyps: list[str]) -> float:
    ref_chars = list("".join(normalize(text).replace(" ", "") for text in refs))
    hyp_chars = list("".join(normalize(text).replace(" ", "") for text in hyps))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def load_torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("torch is required for ASR inference but is not installed.") from exc


def load_transformers_pipeline():
    try:
        module = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Install transformers before using the transformers ASR engine."
        ) from exc
    return module.pipeline


def load_transformers_module():
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Install transformers before using the whisper_nbest ASR mode."
        ) from exc


def load_whisperx_module(importer: Callable[[], Any] | None = None):
    importer = importer or (lambda: importlib.import_module("whisperx"))
    try:
        return importer()
    except ImportError as exc:
        raise RuntimeError(
            "whisperx is not installed. Install whisperx and its alignment dependencies before using this engine."
        ) from exc


def load_pyannote_pipeline():
    try:
        module = importlib.import_module("pyannote.audio")
    except ImportError as exc:
        raise RuntimeError("pyannote.audio is not installed. Install pyannote.audio before using --diarize.") from exc
    return module.Pipeline


def load_silero_segmenter():
    module = importlib.import_module("vad_stage1_silero")
    return module.SileroSegmenter


def load_chunking_helpers():
    module = importlib.import_module("vad_stage2")
    return module.build_dynamic_chunks, module.export_chunk_audio


def resolve_hf_token(cli_token: str | None, diarize: bool, pyannote_model: str | None = None) -> str | None:
    token = cli_token or os.environ.get("HF_TOKEN")
    if diarize and pyannote_model:
        return token
    if diarize and not token:
        raise ValueError("HF token is required when diarization is enabled. Use --hf_token or HF_TOKEN.")
    return token



def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/root/siton-tmp/assignment_C/C2_ASR/data/cremd-test200.json")
    ap.add_argument("--model", default="/root/siton-tmp/assignment_C/model/whisper-large-v3")
    ap.add_argument("--asr_mode", default="onebest", choices=["onebest", "nbest"], help="C2 output mode: onebest writes asr_predictions.json; nbest writes asr_nbest_predictions.json for C3")
    ap.add_argument("--vad_backend", default="silero", choices=["energy", "silero"], help="VAD backend used before dynamic chunking")
    ap.add_argument("--engine", default="transformers", choices=["transformers", "whisperx"], help="1-best ASR engine used when --asr_mode onebest")
    ap.add_argument("--start", type=int, default=0, help="璧峰涓嬫爣(榛樿0)")
    ap.add_argument("--n", type=int, default=0, help="鍙窇鍓嶅嚑鏉★紱榛樿 0 = 璺戞暟鎹泦鍏ㄩ儴")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--nbest", type=int, default=5, help="nbest mode 杈撳嚭鐨勬瘡涓?chunk 鍊欓€夋暟閲?)
    ap.add_argument("--beam_size", type=int, default=5, help="nbest mode beam search 瀹藉害")
    ap.add_argument("--nbest_diversity_penalty", type=float, default=0.1, help="N-best 涓嶈冻鏃?diverse beam search 鐨勬儵缃氱郴鏁?)
    ap.add_argument("--nbest_use_diverse_beam", action="store_true", help="N-best 涓嶈冻鏃跺皾璇?diverse beam search")
    ap.add_argument("--nbest_sample_multiplier", type=int, default=8, help="sampling fallback 姣忚疆閲囨牱鍊欓€夊€嶆暟")
    ap.add_argument("--nbest_sample_temperatures", default="0.7,1.0,1.3", help="sampling fallback 娓╁害鍒楄〃锛岄€楀彿鍒嗛殧")
    ap.add_argument("--language", default=None, help="璇█浠ｇ爜锛涗笉浼犲垯鐢辨ā鍨嬭嚜鍔ㄥ鐞?)
    ap.add_argument("--compute_type", default="float16", help="WhisperX/faster-whisper compute type")
    ap.add_argument("--diarize", action="store_true", help="鍚敤 PyAnnote 璇磋瘽浜鸿仛绫?)
    ap.add_argument("--hf_token", default=None, help="PyAnnote/WhisperX 璇磋瘽浜鸿仛绫绘墍闇€ Hugging Face token")
    ap.add_argument(
        "--pyannote_model",
        default="/root/siton-tmp/assignment_C/model/pyannote-speaker-diarization-3.1",
        help="鏈湴 pyannote speaker-diarization 妯″瀷鐩綍鎴?config.yaml锛涗负绌烘椂鎵嶈蛋 Hugging Face repo",
    )
    ap.add_argument(
        "--pyannote_segmentation_model",
        default=DEFAULT_PYANNOTE_SEGMENTATION_MODEL,
        help="鏈湴 pyannote segmentation 妯″瀷鐩綍锛涚敤浜庢浛鎹?diarization config 鍐呴儴鐨勮繙绔?pyannote/segmentation-3.0 寮曠敤",
    )
    ap.add_argument("--vad_threshold", type=float, default=0.35)
    ap.add_argument("--vad_min_speech_ms", type=int, default=250)
    ap.add_argument("--vad_min_silence_ms", type=int, default=200)
    ap.add_argument("--vad_speech_pad_ms", type=int, default=80)
    ap.add_argument("--chunk_max_ms", type=int, default=30000)
    ap.add_argument("--chunk_min_ms", type=int, default=1000)
    ap.add_argument("--chunk_merge_gap_ms", type=int, default=500)
    ap.add_argument("--chunk_overlap_ms", type=int, default=500)
    ap.add_argument("--offline", action="store_true", help="绂荤嚎妯″紡锛氫笉鑱旂綉锛岀敤鏈湴/缂撳瓨宸蹭笅杞界殑妯″瀷")
    ap.add_argument("--outdir", default="/root/siton-tmp/assignment_C/C2_ASR/outputs/c2_vad_asr/cremd-200-v3")
    return ap

def load_dataset_rows(dataset_path: str, start: int = 0, n: int = 0) -> tuple[list[dict], str]:
    with open(dataset_path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError("dataset JSON must be a list of objects")
    data_root = os.path.dirname(os.path.abspath(dataset_path))
    selected = rows[start:] if n <= 0 else rows[start : start + n]
    return selected, data_root


def resolve_audio_paths(samples: list[dict], data_root: str) -> list[str]:
    return [os.path.join(data_root, sample["audio"]) for sample in samples]


def resolve_generation_kwargs(language: str | None) -> dict[str, str]:
    kwargs = {"task": "transcribe"}
    if language:
        kwargs["language"] = language
    return kwargs


def build_transformers_prediction_row(sample: dict, hypothesis: str) -> dict:
    return {
        "id": sample["id"],
        "audio": sample["audio"],
        "reference": sample.get("text", ""),
        "hypothesis": hypothesis,
        "emotion": sample.get("emotion"),
        "engine": "transformers",
    }


def build_chunked_prediction_row(sample: dict, chunks: list[dict], hypothesis: str, engine: str) -> dict:
    speakers = sorted({str(chunk.get("speaker")) for chunk in chunks if chunk.get("speaker")})
    return {
        "id": sample["id"],
        "audio": sample["audio"],
        "reference": sample.get("text", ""),
        "hypothesis": hypothesis,
        "emotion": sample.get("emotion"),
        "engine": engine,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "speakers": speakers,
    }


def build_whisper_nbest_prediction_row(sample: dict, chunks: list[dict], hypothesis: str, model: str) -> dict:
    row = build_chunked_prediction_row(sample, chunks, hypothesis, engine="whisper_nbest")
    row["model"] = model
    return row


def build_whisperx_prediction_row(sample: dict, result: dict, stage_times: dict[str, float]) -> dict:
    segments = result.get("segments", [])
    texts = [str(segment.get("text", "")).strip() for segment in segments if str(segment.get("text", "")).strip()]
    words = [word for segment in segments for word in segment.get("words", [])]
    speakers = sorted({str(segment.get("speaker")) for segment in segments if segment.get("speaker")})
    return {
        "id": sample["id"],
        "audio": sample["audio"],
        "reference": sample.get("text", ""),
        "hypothesis": " ".join(texts).strip(),
        "emotion": sample.get("emotion"),
        "engine": "whisperx",
        "language": result.get("language"),
        "segments": segments,
        "words": words,
        "speakers": speakers,
        "stage_times": stage_times,
    }


def build_chunks_from_segments(
    segments: list[dict],
    audio_ms: int,
    max_chunk_ms: int,
    min_chunk_ms: int,
    merge_gap_ms: int,
    overlap_ms: int,
    chunk_builder: Callable[..., list[dict]] | None = None,
) -> list[dict]:
    if chunk_builder is None:
        chunk_builder, _ = load_chunking_helpers()
    return chunk_builder(
        segments,
        audio_ms=audio_ms,
        max_chunk_ms=max_chunk_ms,
        min_chunk_ms=min_chunk_ms,
        merge_gap_ms=merge_gap_ms,
        overlap_ms=overlap_ms,
    )


def cap_chunk_durations(chunks: list[dict], max_duration_ms: int) -> list[dict]:
    capped = []
    for chunk in chunks:
        row = dict(chunk)
        duration_ms = int(row["end_ms"]) - int(row["start_ms"])
        if duration_ms > max_duration_ms:
            row["end_ms"] = int(row["start_ms"]) + max_duration_ms
            row["duration_ms"] = max_duration_ms
        else:
            row["duration_ms"] = duration_ms
        capped.append(row)
    return capped


def normalize_pyannote_segments(diarization_result: Any) -> list[dict]:
    if isinstance(diarization_result, list):
        return [
            {"start": float(row["start"]), "end": float(row["end"]), "speaker": str(row["speaker"])}
            for row in diarization_result
        ]
    segments = []
    for turn, _track, speaker in diarization_result.itertracks(yield_label=True):
        segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
    return segments


def assign_speakers_to_chunks(chunks: list[dict], diarization_segments: list[dict]) -> list[dict]:
    assigned = []
    for chunk in chunks:
        row = dict(chunk)
        start_s = int(row.get("start_ms", 0)) / 1000.0
        end_s = int(row.get("end_ms", 0)) / 1000.0
        best_speaker = None
        best_overlap = 0.0
        for segment in diarization_segments:
            overlap = max(0.0, min(end_s, float(segment["end"])) - max(start_s, float(segment["start"])))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(segment["speaker"])
        if best_speaker:
            row["speaker"] = best_speaker
            row["speaker_overlap_s"] = round(best_overlap, 3)
        assigned.append(row)
    return assigned


def normalize_nbest_entries(candidates: list[Any], top_n: int) -> list[dict]:
    entries = []
    seen = set()
    for candidate in candidates:
        if isinstance(candidate, dict):
            text = str(candidate.get("text", "")).strip()
            score = candidate.get("avg_logprob", candidate.get("score"))
        else:
            text = str(candidate).strip()
            score = None
        if not text:
            continue
        key = normalize(text)
        if key in seen:
            continue
        seen.add(key)
        row = {"rank": len(entries) + 1, "text": text}
        if score is not None:
            row["avg_logprob"] = round(float(score), 6)
        entries.append(row)
        if len(entries) >= top_n:
            break
    return entries


def parse_float_list(value: str | Sequence[float] | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return tuple(float(item) for item in items) or default
    return tuple(float(item) for item in value) or default


class HFWhisperNBestBackend:
    def __init__(
        self,
        model_path: str,
        nbest: int = 5,
        beam_size: int = 5,
        language: str | None = None,
        diversity_penalty: float = 0.1,
        sample_temperature: float = 0.7,
        sample_top_p: float = 0.95,
        sample_multiplier: int = 8,
        sample_temperatures: Sequence[float] | None = None,
        use_diverse_beam: bool = False,
        torch_module=None,
        transformers_module=None,
    ):
        if nbest <= 0:
            raise ValueError("nbest must be positive")
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")
        self.model_path = model_path
        self.nbest = nbest
        self.beam_size = max(beam_size, nbest)
        self.language = language
        self.diversity_penalty = diversity_penalty
        self.sample_temperature = sample_temperature
        self.sample_top_p = sample_top_p
        self.sample_multiplier = max(1, int(sample_multiplier))
        self.sample_temperatures = tuple(sample_temperatures) if sample_temperatures else (sample_temperature,)
        self.use_diverse_beam = use_diverse_beam
        self.torch = torch_module
        self.transformers = transformers_module
        self.model = None
        self.processor = None
        self.device = None
        self.model_dtype = None
        self.last_warnings = []

    def transcribe_nbest(self, audio_path: str) -> list[dict]:
        self.last_warnings = []
        self._load()
        read_wav_mono = importlib.import_module("vad_stage1_energy").read_wav_mono
        audio, sample_rate = read_wav_mono(audio_path, target_sample_rate=16000)
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=self.model_dtype)
        candidates = self._generate_candidates(input_features, self._generation_kwargs())
        entries = normalize_nbest_entries(candidates, self.nbest)

        if len(entries) < self.nbest and self.use_diverse_beam:
            diverse_kwargs = self._diverse_generation_kwargs()
            if diverse_kwargs is not None:
                try:
                    candidates.extend(self._generate_candidates(input_features, diverse_kwargs))
                except RuntimeError as exc:
                    self.last_warnings.append(f"diverse_beam_failed: {exc}")
                entries = normalize_nbest_entries(candidates, self.nbest)
        for temperature in self.sample_temperatures:
            if len(entries) >= self.nbest:
                break
            sample_kwargs = self._sampling_generation_kwargs(temperature)
            candidates.extend(self._generate_candidates(input_features, sample_kwargs))
            entries = normalize_nbest_entries(candidates, self.nbest)
        return entries

    def _generation_kwargs(self) -> dict:
        kwargs = {
            "num_beams": self.beam_size,
            "num_return_sequences": self.nbest,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if self.language and hasattr(self.processor, "get_decoder_prompt_ids"):
            kwargs["forced_decoder_ids"] = self.processor.get_decoder_prompt_ids(
                language=self.language,
                task="transcribe",
            )
        return kwargs

    def _diverse_generation_kwargs(self) -> dict | None:
        group_count = min(self.nbest, self.beam_size)
        while group_count > 1 and self.beam_size % group_count != 0:
            group_count -= 1
        if group_count <= 1:
            return None
        kwargs = self._generation_kwargs()
        kwargs["num_beam_groups"] = group_count
        kwargs["diversity_penalty"] = self.diversity_penalty
        return kwargs

    def _sampling_generation_kwargs(self, temperature: float) -> dict:
        kwargs = self._generation_kwargs()
        kwargs["num_beams"] = 1
        kwargs["num_return_sequences"] = self.nbest * self.sample_multiplier
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
        kwargs["top_p"] = self.sample_top_p
        kwargs.pop("num_beam_groups", None)
        kwargs.pop("diversity_penalty", None)
        return kwargs

    def _generate_candidates(self, input_features, generation_kwargs: dict) -> list[dict]:
        with self.torch.no_grad():
            generated = self.model.generate(input_features, **generation_kwargs)

        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        texts = self.processor.batch_decode(sequences, skip_special_tokens=True)
        scores = getattr(generated, "sequences_scores", None)
        candidates = []
        for index, text in enumerate(texts):
            row = {"text": str(text).strip()}
            if scores is not None:
                row["avg_logprob"] = float(scores[index].detach().cpu())
            candidates.append(row)
        return candidates

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        self.torch = self.torch or load_torch()
        self.transformers = self.transformers or load_transformers_module()
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.model_dtype = self.torch.float16 if self.device == "cuda" else self.torch.float32
        model_class = getattr(self.transformers, "AutoModelForSpeechSeq2Seq")
        processor_class = getattr(self.transformers, "AutoProcessor")
        self.processor = processor_class.from_pretrained(self.model_path)
        self.model = model_class.from_pretrained(self.model_path, torch_dtype=self.model_dtype)
        self.model.to(self.device)
        self.model.eval()
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.return_dict_in_generate = True
            self.model.generation_config.output_scores = True


def localize_pyannote_pipeline_config(model_path: str, segmentation_model_path: str | None = None) -> str:
    if not model_path:
        return model_path

    config_path = model_path
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.yaml")
    if not os.path.isfile(config_path):
        return model_path

    segmentation_model_path = segmentation_model_path or DEFAULT_PYANNOTE_SEGMENTATION_MODEL
    if not segmentation_model_path or not os.path.exists(segmentation_model_path):
        return config_path

    with open(config_path, encoding="utf-8") as f:
        config_text = f.read()

    replacements = {
        "pyannote/segmentation-3.0": segmentation_model_path,
        '"pyannote/segmentation-3.0"': f'"{segmentation_model_path}"',
        "'pyannote/segmentation-3.0'": f"'{segmentation_model_path}'",
    }
    localized_text = config_text
    for old, new in replacements.items():
        localized_text = localized_text.replace(old, new)

    if localized_text == config_text:
        return config_path

    fd, localized_config = tempfile.mkstemp(prefix="pyannote-local-", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(localized_text)
    return localized_config


def build_pyannote_diarizer(
    token: str | None,
    device: str,
    model_path: str | None = None,
    pipeline_class=None,
    segmentation_model_path: str | None = None,
):
    Pipeline = pipeline_class or load_pyannote_pipeline()
    if model_path:
        pipeline_config = localize_pyannote_pipeline_config(model_path, segmentation_model_path)
        pipeline = Pipeline.from_pretrained(pipeline_config)
    else:
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    if hasattr(pipeline, "to"):
        try:
            import torch

            pipeline.to(torch.device(device))
        except Exception:
            pipeline.to(device)
    return pipeline


def build_vad_segmenter_factory(args):
    vad_backend = getattr(args, "vad_backend", "silero")
    if vad_backend == "energy":
        module = importlib.import_module("vad_stage1_energy")

        class _EnergySegmenter:
            def run_file(self, audio_path: str, output_json: str | None = None) -> dict:
                return module.run_vad_file(
                    audio_path,
                    output_json=output_json,
                    threshold=args.vad_threshold,
                    min_speech_ms=args.vad_min_speech_ms,
                    min_silence_ms=args.vad_min_silence_ms,
                    speech_pad_ms=args.vad_speech_pad_ms,
                    sample_rate=16000,
                )

        return _EnergySegmenter

    if vad_backend != "silero":
        raise ValueError(f"unsupported vad_backend: {vad_backend}")

    return lambda: load_silero_segmenter()(
        sample_rate=16000,
        threshold=args.vad_threshold,
        min_speech_ms=args.vad_min_speech_ms,
        min_silence_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms,
        device="cpu",
    )


def build_vad_chunks_for_audio(
    audio_path: str,
    args,
    segmenter_factory: Callable[[], Any] | None = None,
    chunk_builder: Callable[..., list[dict]] | None = None,
) -> tuple[list[dict], int]:
    segmenter_factory = segmenter_factory or build_vad_segmenter_factory(args)
    segmenter = segmenter_factory()
    vad_result = segmenter.run_file(audio_path, output_json=None)
    audio_ms = int(vad_result.get("summary", {}).get("audio_ms", 0))
    chunks = build_chunks_from_segments(
        segments=vad_result.get("segments", []),
        audio_ms=audio_ms,
        max_chunk_ms=args.chunk_max_ms,
        min_chunk_ms=args.chunk_min_ms,
        merge_gap_ms=args.chunk_merge_gap_ms,
        overlap_ms=args.chunk_overlap_ms,
        chunk_builder=chunk_builder,
    )
    # Keep exported chunk audio below Whisper's 30s long-form threshold.
    chunks = cap_chunk_durations(chunks, max_duration_ms=min(args.chunk_max_ms - 500, 29500))
    return chunks, audio_ms


def run_transformers_engine(
    samples: list[dict],
    audio_paths: list[str],
    args,
    torch_module=None,
    pipeline_factory=None,
    segmenter_factory: Callable[[], Any] | None = None,
    chunk_exporter: Callable[[str, dict, str, str], str] | None = None,
    chunk_builder: Callable[..., list[dict]] | None = None,
    diarizer_factory: Callable[..., Any] | None = None,
) -> tuple[list[dict], float]:
    torch = torch_module or load_torch()
    pipeline = pipeline_factory or load_transformers_pipeline()
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"鍔犺浇妯″瀷: {args.model}  (engine=transformers, device={'cuda:0' if device == 0 else 'cpu'}, dtype={dtype})")
    t0 = time.time()
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        torch_dtype=dtype,
        device=device,
    )
    print(f"妯″瀷鍔犺浇鑰楁椂: {time.time() - t0:.1f}s")
    if device == 0 and hasattr(torch.cuda, "get_device_name") and hasattr(torch.cuda, "memory_allocated"):
        print(f"GPU: {torch.cuda.get_device_name(0)}  鏄惧瓨鍗犵敤: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"\n寮€濮嬫壒閲忚瘑鍒?{len(audio_paths)} 鏉￠煶棰?(batch_size={args.batch_size}) ...")
    started = time.time()
    predictions = []
    if chunk_exporter is None:
        _, chunk_exporter = load_chunking_helpers()
    diarizer = None
    if args.diarize:
        token = resolve_hf_token(args.hf_token, diarize=True, pyannote_model=args.pyannote_model)
        diarizer_factory = diarizer_factory or build_pyannote_diarizer
        diarizer = diarizer_factory(
            token,
            "cuda" if device == 0 else "cpu",
            model_path=args.pyannote_model,
            segmentation_model_path=getattr(args, "pyannote_segmentation_model", DEFAULT_PYANNOTE_SEGMENTATION_MODEL),
        )

    for sample, audio_path in zip(samples, audio_paths):
        chunks, _audio_ms = build_vad_chunks_for_audio(
            audio_path,
            args,
            segmenter_factory=segmenter_factory,
            chunk_builder=chunk_builder,
        )
        chunk_dir = os.path.join(args.outdir, "stage2_chunks")
        chunk_paths = [
            chunk_exporter(audio_path, chunk, chunk_dir, str(sample.get("id") or "item"))
            for chunk in chunks
        ]
        for chunk, chunk_audio in zip(chunks, chunk_paths):
            result = asr(
                chunk_audio,
                batch_size=1,
                generate_kwargs=resolve_generation_kwargs(args.language),
            )
            chunk["text"] = str(result["text"]).strip()
            chunk["chunk_audio"] = chunk_audio
        if diarizer is not None:
            diarization_segments = normalize_pyannote_segments(diarizer(audio_path))
            chunks = assign_speakers_to_chunks(chunks, diarization_segments)
        hypothesis = " ".join(chunk.get("text", "") for chunk in chunks).strip()
        predictions.append(build_chunked_prediction_row(sample, chunks, hypothesis, engine="transformers"))

    infer_time = time.time() - started
    print(f"鎵归噺鎺ㄧ悊瀹屾垚锛岃€楁椂 {infer_time:.1f}s锛屽钩鍧?{infer_time / len(audio_paths) * 1000:.0f} ms/鏉?)
    return predictions, infer_time


def run_whisper_nbest_engine(
    samples: list[dict],
    audio_paths: list[str],
    args,
    nbest_backend=None,
    segmenter_factory: Callable[[], Any] | None = None,
    chunk_exporter: Callable[[str, dict, str, str], str] | None = None,
    chunk_builder: Callable[..., list[dict]] | None = None,
) -> tuple[list[dict], float]:
    nbest_backend = nbest_backend or HFWhisperNBestBackend(
        args.model,
        nbest=args.nbest,
        beam_size=args.beam_size,
        language=args.language,
        diversity_penalty=getattr(args, "nbest_diversity_penalty", 0.1),
        sample_multiplier=getattr(args, "nbest_sample_multiplier", 8),
        sample_temperatures=parse_float_list(
            getattr(args, "nbest_sample_temperatures", None),
            default=(0.7, 1.0, 1.3),
        ),
        use_diverse_beam=getattr(args, "nbest_use_diverse_beam", False),
    )
    print(
        f"鍔犺浇妯″瀷: {args.model}  "
        f"(mode=nbest, nbest={args.nbest}, beam_size={args.beam_size})"
    )
    print(f"\n寮€濮嬮€?chunk 鐢熸垚 N-best锛歿len(audio_paths)} 鏉￠煶棰?...")
    started = time.time()
    predictions = []
    if chunk_exporter is None:
        _, chunk_exporter = load_chunking_helpers()

    for sample, audio_path in zip(samples, audio_paths):
        chunks, _audio_ms = build_vad_chunks_for_audio(
            audio_path,
            args,
            segmenter_factory=segmenter_factory,
            chunk_builder=chunk_builder,
        )
        chunk_dir = os.path.join(args.outdir, "stage2_chunks")
        chunk_paths = [
            chunk_exporter(audio_path, chunk, chunk_dir, str(sample.get("id") or "item"))
            for chunk in chunks
        ]
        for chunk, chunk_audio in zip(chunks, chunk_paths):
            nbest = normalize_nbest_entries(nbest_backend.transcribe_nbest(chunk_audio), args.nbest)
            if not nbest:
                nbest = [{"rank": 1, "text": ""}]
            chunk["text"] = nbest[0]["text"]
            chunk["nbest"] = nbest
            backend_warnings = list(getattr(nbest_backend, "last_warnings", []))
            if len(nbest) < args.nbest:
                backend_warnings.append("unique_nbest_candidates_less_than_requested")
                chunk["requested_nbest"] = args.nbest
                chunk["actual_nbest"] = len(nbest)
            if backend_warnings:
                chunk["nbest_warnings"] = backend_warnings
            chunk["chunk_audio"] = chunk_audio
        hypothesis = " ".join(chunk.get("text", "") for chunk in chunks).strip()
        predictions.append(build_whisper_nbest_prediction_row(sample, chunks, hypothesis, model=args.model))

    infer_time = time.time() - started
    print(f"N-best 鎺ㄧ悊瀹屾垚锛岃€楁椂 {infer_time:.1f}s锛屽钩鍧?{infer_time / len(audio_paths) * 1000:.0f} ms/鏉?)
    return predictions, infer_time


def run_whisperx_engine(
    samples: list[dict],
    audio_paths: list[str],
    args,
    torch_module=None,
    whisperx_module=None,
    segmenter_factory: Callable[[], Any] | None = None,
    chunk_exporter: Callable[[str, dict, str, str], str] | None = None,
    chunk_builder: Callable[..., list[dict]] | None = None,
) -> tuple[list[dict], float]:
    torch = torch_module or load_torch()
    whisperx = whisperx_module or load_whisperx_module()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    token = resolve_hf_token(args.hf_token, args.diarize, pyannote_model=args.pyannote_model)
    language = args.language or None

    print(f"鍔犺浇妯″瀷: {args.model}  (engine=whisperx, device={device}, compute_type={args.compute_type})")
    t0 = time.time()
    model = whisperx.load_model(args.model, device, compute_type=args.compute_type, language=language)
    print(f"妯″瀷鍔犺浇鑰楁椂: {time.time() - t0:.1f}s")

    diarizer = None
    if args.diarize:
        try:
            diarizer = whisperx.DiarizationPipeline(use_auth_token=token, device=device)
        except TypeError:
            diarizer = whisperx.DiarizationPipeline(token=token, device=device)

    align_cache: dict[str, tuple[Any, Any]] = {}
    predictions = []
    started = time.time()
    if chunk_exporter is None:
        _, chunk_exporter = load_chunking_helpers()

    for sample, audio_path in zip(samples, audio_paths):
        stage_times: dict[str, float] = {}
        chunks, _audio_ms = build_vad_chunks_for_audio(
            audio_path,
            args,
            segmenter_factory=segmenter_factory,
            chunk_builder=chunk_builder,
        )
        chunk_dir = os.path.join(args.outdir, "stage2_chunks")
        aligned_segments = []

        for chunk in chunks:
            chunk_audio = chunk_exporter(audio_path, chunk, chunk_dir, str(sample.get("id") or "item"))
            audio = whisperx.load_audio(chunk_audio)

            t_stage = time.time()
            result = model.transcribe(audio, batch_size=args.batch_size)
            stage_times["transcribe"] = stage_times.get("transcribe", 0.0) + round(time.time() - t_stage, 3)

            lang = result.get("language") or language or "en"
            if lang not in align_cache:
                align_cache[lang] = whisperx.load_align_model(language_code=lang, device=device)
            model_a, metadata = align_cache[lang]

            t_stage = time.time()
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            stage_times["align"] = stage_times.get("align", 0.0) + round(time.time() - t_stage, 3)

            offset_s = chunk["start_ms"] / 1000.0
            chunk_texts = []
            for segment in result.get("segments", []):
                shifted = dict(segment)
                shifted["start"] = float(shifted.get("start", 0.0)) + offset_s
                shifted["end"] = float(shifted.get("end", 0.0)) + offset_s
                shifted_words = []
                for word in shifted.get("words", []):
                    shifted_word = dict(word)
                    if "start" in shifted_word:
                        shifted_word["start"] = float(shifted_word["start"]) + offset_s
                    if "end" in shifted_word:
                        shifted_word["end"] = float(shifted_word["end"]) + offset_s
                    shifted_words.append(shifted_word)
                shifted["words"] = shifted_words
                aligned_segments.append(shifted)
                text = str(shifted.get("text", "")).strip()
                if text:
                    chunk_texts.append(text)
            chunk["text"] = " ".join(chunk_texts).strip()
            chunk["chunk_audio"] = chunk_audio

        final_result = {"language": language or "en", "segments": aligned_segments}

        if diarizer is not None:
            diar_audio = whisperx.load_audio(audio_path)
            t_stage = time.time()
            diarize_segments = diarizer(diar_audio)
            stage_times["diarize"] = round(time.time() - t_stage, 3)

            t_stage = time.time()
            final_result = whisperx.assign_word_speakers(diarize_segments, final_result)
            stage_times["assign_speakers"] = round(time.time() - t_stage, 3)

        row = build_whisperx_prediction_row(sample, final_result, stage_times)
        row["num_chunks"] = len(chunks)
        row["chunks"] = chunks
        predictions.append(row)

    infer_time = time.time() - started
    print(f"WhisperX 鎺ㄧ悊瀹屾垚锛岃€楁椂 {infer_time:.1f}s锛屽钩鍧?{infer_time / len(audio_paths) * 1000:.0f} ms/鏉?)
    return predictions, infer_time


def summarize_predictions(predictions: list[dict], references: list[str], infer_time: float, model: str, engine: str) -> dict:
    hypotheses = [prediction.get("hypothesis", "") for prediction in predictions]
    wer = compute_wer(references, hypotheses)
    cer = compute_cer(references, hypotheses)
    return {
        "engine": engine,
        "model": model,
        "num_samples": len(predictions),
        "WER": round(wer, 4),
        "CER": round(cer, 4),
        "infer_time_sec": round(infer_time, 1),
        "ms_per_sample": round(infer_time / len(predictions) * 1000, 1) if predictions else 0.0,
    }


def save_outputs(predictions: list[dict], summary: dict, outdir: str, asr_mode: str = "onebest") -> tuple[str, str]:
    os.makedirs(outdir, exist_ok=True)
    predictions_name = "asr_nbest_predictions.json" if asr_mode == "nbest" else "asr_predictions.json"
    out_json = os.path.join(outdir, predictions_name)
    summary_json = os.path.join(outdir, "c2_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out_json, summary_json


def print_examples(samples: list[dict], predictions: list[dict], limit: int = 5) -> None:
    print("\n---- 璇嗗埆鏍蜂緥 ----")
    for sample, prediction in list(zip(samples, predictions))[:limit]:
        print(f"[{sample['id']}]")
        print(f"  鍙傝€? {sample.get('text', '')}")
        print(f"  璇嗗埆: {prediction.get('hypothesis', '')}")


def run(args) -> dict:
    samples, data_root = load_dataset_rows(args.dataset, start=args.start, n=args.n)
    audio_paths = resolve_audio_paths(samples, data_root)
    references = [sample.get("text", "") for sample in samples]

    print(f"鏁版嵁闆? {os.path.abspath(args.dataset)}")
    print(f"鍏?{len(samples)} 鏉″緟澶勭悊鏍锋湰")

    if args.asr_mode == "nbest":
        predictions, infer_time = run_whisper_nbest_engine(samples, audio_paths, args)
        engine_name = "whisper_nbest"
    elif args.engine == "transformers":
        predictions, infer_time = run_transformers_engine(samples, audio_paths, args)
        engine_name = "transformers"
    else:
        predictions, infer_time = run_whisperx_engine(samples, audio_paths, args)
        engine_name = "whisperx"

    summary = summarize_predictions(predictions, references, infer_time, args.model, engine_name)
    print(f"\n==== 璇勬祴缁撴灉锛坽len(samples)} 鏉★級====")
    print(f"WER (璇嶉敊璇巼): {summary['WER'] * 100:.2f}%")
    print(f"CER (瀛楃閿欒鐜?: {summary['CER'] * 100:.2f}%")
    print_examples(samples, predictions)
    out_json, summary_json = save_outputs(predictions, summary, args.outdir, asr_mode=args.asr_mode)
    print(f"\n宸蹭繚瀛? {out_json} (渚汣3浣跨敤) 鍜?{summary_json}")
    return summary


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
