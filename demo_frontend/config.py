"""Configuration and server-sample discovery for the offline demo frontend."""

from __future__ import annotations

import hashlib
import json
import os
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when the frontend configuration cannot be used safely."""


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    python_paths: dict[str, str] = field(default_factory=dict)
    model_paths: dict[str, str] = field(default_factory=dict)
    script_paths: dict[str, str] = field(default_factory=dict)
    sample_dir: Path | None = None
    jobs_dir: Path | None = None
    host: str = "127.0.0.1"
    port: int = 7860
    max_single_job: bool = True
    default_correction_backend: str = "local"
    correction_api: dict[str, Any] = field(default_factory=dict)

    def python_for(self, module: str) -> str:
        value = self.python_paths.get(module)
        if not value:
            raise ConfigError(f"python.{module} is required")
        return value

    def model_for(self, name: str) -> str:
        value = self.model_paths.get(name, "")
        if not value:
            raise ConfigError(f"model.{name} is required")
        return value

    def script_for(self, name: str) -> Path:
        value = self.script_paths.get(name)
        if not value:
            raise ConfigError(f"script.{name} is required")
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()

    def samples_root(self) -> Path:
        return (self.sample_dir or self.project_root).resolve()

    def jobs_root(self) -> Path:
        return (self.jobs_dir or self.project_root / "demo_frontend" / "jobs").resolve()

    def correction_api_value(self, name: str, default: Any = None) -> Any:
        return self.correction_api.get(name, default)


def _path_from_value(root: Path, value: Any, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def load_config(raw: Mapping[str, Any]) -> AppConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration must be a JSON object")
    root_value = raw.get("project_root")
    if not root_value:
        raise ConfigError("project_root is required")
    root = Path(str(root_value)).expanduser().resolve()
    python_paths = {str(key): str(value) for key, value in dict(raw.get("python", {})).items() if value}
    model_paths = {str(key): str(value) for key, value in dict(raw.get("models", {})).items() if value}
    script_paths = {str(key): str(value) for key, value in dict(raw.get("scripts", {})).items() if value}
    sample_dir = _path_from_value(root, raw.get("sample_dir"))
    jobs_dir = _path_from_value(root, raw.get("jobs_dir"))
    server = dict(raw.get("server", {}))
    correction = dict(raw.get("correction", {}))
    default_correction_backend = str(correction.get("default_backend", "local") or "local")
    if default_correction_backend not in {"local", "openai_compatible"}:
        raise ConfigError("correction.default_backend must be local or openai_compatible")
    correction_api = dict(correction.get("api", {}))
    if "api_base" in correction_api and "base_url" not in correction_api:
        correction_api["base_url"] = correction_api["api_base"]
    if "api_key_env" not in correction_api:
        correction_api["api_key_env"] = "DEMO_CORRECTION_API_KEY"
    try:
        port = int(server.get("port", raw.get("port", 7860)))
    except (TypeError, ValueError) as exc:
        raise ConfigError("server.port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ConfigError("server.port must be between 1 and 65535")
    host = str(server.get("host", raw.get("host", "127.0.0.1")))
    if not host:
        raise ConfigError("server.host cannot be empty")
    return AppConfig(
        project_root=root,
        python_paths=python_paths,
        model_paths=model_paths,
        script_paths=script_paths,
        sample_dir=sample_dir,
        jobs_dir=jobs_dir,
        host=host,
        port=port,
        max_single_job=bool(server.get("max_single_job", True)),
        default_correction_backend=default_correction_backend,
        correction_api=correction_api,
    )


def load_config_file(path: str | os.PathLike[str]) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read config: {config_path}") from exc
    if "project_root" not in raw:
        raw["project_root"] = str(config_path.parent.parent)
    else:
        configured_root = Path(str(raw["project_root"])).expanduser()
        if not configured_root.is_absolute():
            raw["project_root"] = str((config_path.parent / configured_root).resolve())
    return load_config(raw)


def _wav_metadata(path: Path) -> tuple[float, int, int]:
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            channels = wav.getnchannels()
    except (OSError, wave.Error) as exc:
        raise ConfigError(f"invalid wav sample: {path}") from exc
    if sample_rate <= 0:
        raise ConfigError(f"invalid sample rate in wav: {path}")
    return round(frames / sample_rate, 3), sample_rate, channels


def discover_samples(sample_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    root = Path(sample_dir).expanduser().resolve()
    if not root.is_dir():
        raise ConfigError(f"sample_dir does not exist: {root}")
    samples: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.wav")):
        if not path.is_file() or path.is_symlink():
            continue
        duration_sec, sample_rate, channels = _wav_metadata(path)
        audio_rel = path.relative_to(root).as_posix()
        sample_id = hashlib.sha1(audio_rel.encode("utf-8")).hexdigest()[:12]
        samples.append({
            "sample_id": sample_id,
            "name": path.stem,
            "audio_rel": audio_rel,
            "duration_sec": duration_sec,
            "sample_rate": sample_rate,
            "channels": channels,
            "long_audio": duration_sec > 30.0,
        })
    return samples
