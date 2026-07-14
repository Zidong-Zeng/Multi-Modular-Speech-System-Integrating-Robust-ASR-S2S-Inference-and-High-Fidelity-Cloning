"""Adapters between the demo job protocol and existing C1-C5 scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import AppConfig, ConfigError


ASR_MODEL_PATH_KEYS = {
    "tiny": "asr_tiny",
    "large-v3": "asr_large_v3",
}


def _value_after(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _script(config: AppConfig, name: str) -> str:
    return str(config.script_for(name))


def _asr_model_path(config: AppConfig, asr_model: str) -> str:
    if asr_model not in ASR_MODEL_PATH_KEYS:
        raise ConfigError(f"unsupported asr_model: {asr_model}")
    key = ASR_MODEL_PATH_KEYS[asr_model]
    if key in config.model_paths:
        return config.model_for(key)
    if asr_model == "large-v3":
        return config.model_for("asr")
    raise ConfigError(f"model.{key} is required")


def build_c1_command(config: AppConfig, input_path: str, outdir: str) -> list[str]:
    return [
        config.python_for("c1"),
        _script(config, "c1"),
        "--input",
        str(input_path),
        "--outdir",
        str(outdir),
        "--no-vad",
        "--no-speed",
        "--no-volume",
        "--no-noise",
    ]


def build_c2_command(
    config: AppConfig,
    dataset_path: str,
    outdir: str,
    *,
    vad_enabled: bool,
    pyannote_enabled: bool,
    correction_enabled: bool,
    asr_model: str,
) -> list[str]:
    nbest = "5" if correction_enabled else "1"
    beam_size = "20" if correction_enabled else "1"
    command = [
        config.python_for("c2"),
        _script(config, "c2"),
        "--dataset",
        str(dataset_path),
        "--model",
        _asr_model_path(config, asr_model),
        "--outdir",
        str(outdir),
        "--asr_mode",
        "nbest",
        "--vad_backend",
        "silero" if vad_enabled else "none",
        "--nbest",
        nbest,
        "--beam_size",
        beam_size,
        "--n",
        "1",
        "--offline",
    ]
    if pyannote_enabled:
        command.append("--diarize")
        pyannote_model = config.model_paths.get("pyannote")
        if pyannote_model:
            command.extend(["--pyannote_model", pyannote_model])
        segmentation_model = config.model_paths.get("pyannote_segmentation")
        if segmentation_model:
            command.extend(["--pyannote_segmentation_model", segmentation_model])
    return command


def build_c3_command(
    config: AppConfig,
    c2_json: str,
    outdir: str,
    *,
    correction_enabled: bool,
    correction_backend: str,
) -> list[str]:
    if correction_backend not in {"local", "openai_compatible"}:
        raise ConfigError(f"unsupported correction_backend: {correction_backend}")
    command = [
        config.python_for("c3"),
        "-m",
        "c3.cli",
        "--c2_json",
        str(c2_json),
        "--outdir",
        str(outdir),
        "--correction_backend",
        correction_backend,
        "--translation_model",
        config.model_for("translation"),
    ]
    if not correction_enabled:
        command.append("--disable_correction")
    elif correction_backend == "local":
        command.extend(["--correction_model", config.model_for("correction")])
    else:
        api_base = str(config.correction_api_value("base_url", "") or "")
        api_model = str(config.correction_api_value("model", "") or "")
        api_key_env = str(config.correction_api_value("api_key_env", "DEMO_CORRECTION_API_KEY") or "DEMO_CORRECTION_API_KEY")
        if not api_base:
            raise ConfigError("correction.api.base_url is required when correction_backend is openai_compatible")
        if not api_model:
            raise ConfigError("correction.api.model is required when correction_backend is openai_compatible")
        command.extend([
            "--correction_api_base",
            api_base,
            "--correction_api_model",
            api_model,
            "--correction_api_key_env",
            api_key_env,
        ])
        for key, flag in (
            ("temperature", "--correction_temperature"),
            ("timeout", "--correction_timeout"),
            ("max_new_tokens", "--correction_max_new_tokens"),
        ):
            value = config.correction_api_value(key)
            if value is not None and value != "":
                command.extend([flag, str(value)])
    return command


def build_c3_env(config: AppConfig, *, correction_enabled: bool, correction_backend: str) -> dict[str, str]:
    if not correction_enabled or correction_backend != "openai_compatible":
        return {}
    api_key = str(config.correction_api_value("api_key", "") or "")
    if not api_key:
        raise ConfigError("correction.api.api_key is required when correction_backend is openai_compatible")
    api_key_env = str(config.correction_api_value("api_key_env", "DEMO_CORRECTION_API_KEY") or "DEMO_CORRECTION_API_KEY")
    return {api_key_env: api_key}


def build_c4_command(
    config: AppConfig,
    input_dir: str,
    outdir: str,
    dataset_path: str,
) -> list[str]:
    return [
        config.python_for("c4"),
        _script(config, "c4"),
        "--dataset",
        str(dataset_path),
        "--input_dir",
        str(input_dir),
        "--model",
        config.model_for("c4"),
        "--outdir",
        str(outdir),
        "--count",
        "1",
        "--offline",
    ]


def build_c5_command(config: AppConfig, dataset_path: str, outdir: str) -> list[str]:
    return [
        config.python_for("c5"),
        _script(config, "c5"),
        "--dataset",
        str(dataset_path),
        "--output",
        str(outdir),
    ]


def _load_json(path: str | Path) -> Any:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read artifact: {path}") from exc


def _first_row(data: Any, label: str) -> dict[str, Any]:
    if isinstance(data, list):
        if not data or not isinstance(data[0], dict):
            raise ConfigError(f"{label} artifact has no object rows")
        return dict(data[0])
    if isinstance(data, dict):
        return dict(data)
    raise ConfigError(f"{label} artifact must be an object or list")


def parse_c2_artifact(path: str | Path) -> dict[str, Any]:
    row = _first_row(_load_json(path), "C2")
    chunks = row.get("chunks")
    if not isinstance(chunks, list):
        raise ConfigError("C2 artifact is missing chunks")
    row["chunks"] = [dict(chunk) for chunk in chunks if isinstance(chunk, dict)]
    row["num_chunks"] = len(row["chunks"])
    return row


def parse_c3_artifact(path: str | Path) -> dict[str, Any]:
    row = _first_row(_load_json(path), "C3")
    row.setdefault("hypothesis", row.get("corrected_transcript_en", ""))
    row.setdefault("translation_zh", "")
    return row


def parse_c4_artifact(path: str | Path) -> dict[str, Any]:
    row = _first_row(_load_json(path), "C4")
    row["translation_zh"] = str(row.get("e2e_translation_zh", row.get("translation_zh", "")) or "")
    return row


def parse_c5_artifact(path: str | Path) -> dict[str, Any]:
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        row = _first_row(data["results"], "C5")
    else:
        row = _first_row(data, "C5")
    row["wav"] = str(row.get("wav", row.get("wav_path", "")) or "")
    return row


def _chunk_top1(chunk: dict[str, Any]) -> str:
    text = str(chunk.get("text", "") or "").strip()
    if text:
        return text
    nbest = chunk.get("nbest")
    if isinstance(nbest, list) and nbest:
        return str(nbest[0].get("text", "") or "").strip()
    return ""


def correction_bypass_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for original in rows:
        row = dict(original)
        chunks = row.get("chunks", [])
        transcript = " ".join(_chunk_top1(chunk) for chunk in chunks if isinstance(chunk, dict)).strip()
        row["asr_top1_transcript_en"] = transcript
        row["corrected_transcript_en"] = transcript
        row["translation_zh"] = ""
        row["correction_enabled"] = False
        row["correction_units"] = []
        result.append(row)
    return result
