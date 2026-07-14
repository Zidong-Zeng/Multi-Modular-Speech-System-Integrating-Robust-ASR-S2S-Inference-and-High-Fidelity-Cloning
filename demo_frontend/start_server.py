"""CLI entrypoint for the offline multimodal speech demo server."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo_frontend.config import AppConfig, ConfigError, discover_samples, load_config_file
from demo_frontend.orchestrator import JobManager
from demo_frontend.pipeline import collect_job_result, create_stage_factory
from demo_frontend.server import create_server


PYTHON_KEYS = ("c1", "c2", "c3", "c4", "c5")
SCRIPT_KEYS = ("c1", "c2", "c3", "c4", "c5")
MODEL_KEYS = ("asr_tiny", "asr_large_v3", "correction", "translation", "c4", "cosyvoice_repo", "cosyvoice")


def _path_exists(value: str | Path, *, file_only: bool = False) -> bool:
    try:
        path = Path(value)
        return path.is_file() if file_only else path.exists()
    except OSError:
        return False


def dry_run_report(config: AppConfig) -> dict:
    checks = []
    for key in PYTHON_KEYS:
        value = config.python_paths.get(key, "")
        checks.append({"name": f"python.{key}", "path": value, "exists": bool(value and _path_exists(value, file_only=True))})
    for key in SCRIPT_KEYS:
        value = config.script_paths.get(key, "")
        path = config.script_for(key) if value else Path("")
        checks.append({"name": f"script.{key}", "path": str(path) if value else "", "exists": bool(value and _path_exists(path, file_only=True))})
    for key in MODEL_KEYS:
        value = config.model_paths.get(key, "")
        checks.append({"name": f"model.{key}", "path": value, "exists": bool(value and _path_exists(value))})
    try:
        samples = discover_samples(config.samples_root())
        sample_error = ""
    except ConfigError as exc:
        samples = []
        sample_error = str(exc)
    return {
        "ready": all(item["exists"] for item in checks) and bool(samples),
        "project_root": str(config.project_root),
        "sample_dir": str(config.samples_root()),
        "sample_count": len(samples),
        "sample_error": sample_error,
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline multimodal speech demo frontend")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config_file(args.config)
    if args.host is not None:
        config = replace(config, host=args.host)
    if args.port is not None:
        if not 1 <= args.port <= 65535:
            raise SystemExit("--port must be between 1 and 65535")
        config = replace(config, port=args.port)
    if args.dry_run:
        print(json.dumps(dry_run_report(config), ensure_ascii=False, indent=2))
        return

    samples = discover_samples(config.samples_root())
    if not samples:
        raise SystemExit(f"No WAV samples found in {config.samples_root()}")
    def result_builder(job_dir, request, job_id):
        result = collect_job_result(job_dir, c4_enabled=request.c4_enabled)
        audio_rel = result.get("c5_audio_rel", "")
        if audio_rel:
            result["c5_audio_url"] = f"/api/jobs/{job_id}/media/{audio_rel}"
            result["c5_audio_name"] = Path(audio_rel).name
        return result

    manager = JobManager(config, create_stage_factory(config), result_builder=result_builder)
    server = create_server(config, manager, samples)
    host, port = server.server_address[:2]
    print(f"Demo frontend serving on http://{host}:{port}", flush=True)
    print(f"SSH tunnel example: ssh -N -L 18080:127.0.0.1:{port} user@server", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
