"""Build per-job manifests and ordered C1-C5 stage specifications."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterable

from .adapters import (
    build_c1_command,
    build_c2_command,
    build_c3_command,
    build_c3_env,
    build_c4_command,
    build_c5_command,
    parse_c2_artifact,
    parse_c3_artifact,
    parse_c4_artifact,
    parse_c5_artifact,
)
from .config import AppConfig, ConfigError
from .orchestrator import JobRequest, StageSpec


def _within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_stage_factory(config: AppConfig) -> Callable[[Path, JobRequest], Iterable[StageSpec]]:
    def factory(job_dir: Path, request: JobRequest) -> list[StageSpec]:
        samples_root = config.samples_root()
        relative_audio = Path(str(request.sample.get("audio_rel", "")))
        if relative_audio.is_absolute() or ".." in relative_audio.parts:
            raise ValueError("sample path escapes configured sample directory")
        source_audio = (samples_root / relative_audio).resolve()
        if not _within(source_audio, samples_root) or not source_audio.is_file():
            raise ValueError(f"sample path is invalid: {relative_audio}")

        stage_dirs = {name: job_dir / name for name in ("c1", "c2", "c3", "c4", "c5")}
        for directory in stage_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        sample_id = str(request.sample.get("sample_id") or source_audio.stem)
        clean_audio = stage_dirs["c1"] / f"{source_audio.stem}_clean.wav"
        manifest_row = {
            "id": sample_id,
            "audio": str(clean_audio),
            "text": str(request.sample.get("reference", "") or ""),
        }
        if request.sample.get("emotion"):
            manifest_row["emotion"] = request.sample["emotion"]
        c2_manifest = job_dir / "c2_manifest.json"
        c4_manifest = job_dir / "c4_manifest.json"
        _write_json(c2_manifest, [manifest_row])
        _write_json(c4_manifest, [manifest_row])

        c2_json = stage_dirs["c2"] / "asr_nbest_predictions.json"
        c3_json = stage_dirs["c3"] / "c3_predictions.json"
        c3_code_dir = config.project_root / "C3_cascade" / "code"

        stages = [
            StageSpec(
                name="c1",
                command=build_c1_command(config, str(source_audio), str(stage_dirs["c1"])),
                python_path=config.python_for("c1"),
                cwd=config.project_root,
            ),
            StageSpec(
                name="c2",
                command=build_c2_command(
                    config,
                    str(c2_manifest),
                    str(stage_dirs["c2"]),
                    vad_enabled=request.vad_enabled,
                    pyannote_enabled=request.pyannote_enabled,
                    correction_enabled=request.correction_enabled,
                    asr_model=request.asr_model,
                ),
                python_path=config.python_for("c2"),
                cwd=config.project_root,
            ),
            StageSpec(
                name="c3",
                command=build_c3_command(
                    config,
                    str(c2_json),
                    str(stage_dirs["c3"]),
                    correction_enabled=request.correction_enabled,
                    correction_backend=request.correction_backend,
                ),
                python_path=config.python_for("c3"),
                cwd=c3_code_dir,
                env={
                    "PYTHONPATH": str(c3_code_dir),
                    **build_c3_env(
                        config,
                        correction_enabled=request.correction_enabled,
                        correction_backend=request.correction_backend,
                    ),
                },
            ),
            StageSpec(
                name="c5",
                command=build_c5_command(config, str(c3_json), str(stage_dirs["c5"])),
                python_path=config.python_for("c5"),
                cwd=config.project_root / "C5_TTS",
                env={
                    "COSYVOICE_REPO": config.model_for("cosyvoice_repo"),
                    "COSYVOICE_MODEL": config.model_for("cosyvoice"),
                },
            ),
        ]
        if request.c4_enabled:
            stages.append(
                StageSpec(
                    name="c4",
                    command=build_c4_command(
                        config,
                        str(stage_dirs["c1"]),
                        str(stage_dirs["c4"]),
                        str(c4_manifest),
                    ),
                    python_path=config.python_for("c4"),
                    cwd=config.project_root / "C4_e2e_package" / "C4_end2end" / "code",
                    critical=False,
                )
            )
        return stages

    return factory


def collect_job_result(job_dir: Path, *, c4_enabled: bool) -> dict:
    c2 = parse_c2_artifact(job_dir / "c2" / "asr_nbest_predictions.json")
    c3 = parse_c3_artifact(job_dir / "c3" / "c3_predictions.json")
    c5 = parse_c5_artifact(job_dir / "c5" / "batch_summary.json")
    result = {
        "asr": str(c2.get("hypothesis", "") or ""),
        "corrected": str(c3.get("hypothesis", c3.get("corrected_transcript_en", "")) or ""),
        "translation": str(c3.get("translation_zh", "") or ""),
        "chunks": c2.get("chunks", []),
        "correction_enabled": bool(c3.get("correction_enabled", True)),
        "c5_audio_rel": f"c5/{c5.get('wav', '')}" if c5.get("wav") else "",
    }
    if c4_enabled:
        try:
            c4 = parse_c4_artifact(job_dir / "c4" / "c4_results.json")
            result["c4_translation"] = str(c4.get("translation_zh", "") or "")
        except ConfigError as exc:
            result["c4_error"] = str(exc)
    return result
