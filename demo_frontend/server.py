"""Dependency-free HTTP API and static file server for the demo frontend."""

from __future__ import annotations

import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

from .config import AppConfig
from .orchestrator import JobManager, JobRequest


SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_-]+$")
JOB_FIELDS = {"sample_id", "vad_enabled", "pyannote_enabled", "correction_enabled", "c4_enabled", "asr_model", "correction_backend"}
ASR_MODELS = {"tiny", "large-v3"}
CORRECTION_BACKENDS = {"local", "openai_compatible"}


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def safe_job_path(jobs_root: Path, job_id: str, relative_name: str) -> Path:
    root = Path(jobs_root).resolve()
    if not SAFE_JOB_ID.fullmatch(job_id):
        raise ValueError("invalid job id")
    relative = Path(unquote(relative_name))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid artifact path")
    raw = root / job_id / relative
    if any(part.is_symlink() for part in [raw, *raw.parents] if _is_within(part.resolve(), root)):
        raise ValueError("artifact symlinks are not allowed")
    candidate = raw.resolve()
    if not _is_within(candidate, (root / job_id).resolve()):
        raise ValueError("artifact path escapes job directory")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


class DemoHTTPServer(ThreadingHTTPServer):
    config: AppConfig
    manager: JobManager
    samples: list[dict[str, Any]]
    static_dir: Path


class DemoRequestHandler(BaseHTTPRequestHandler):
    server: DemoHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > 65536:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request JSON must be an object")
        return payload

    def _serve_static(self, route: str) -> None:
        name = "index.html" if route in ("", "/") else route.lstrip("/")
        if "/" in name or ".." in name:
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        path = (self.server.static_dir / name).resolve()
        if not _is_within(path, self.server.static_dir.resolve()) or not path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok", "offline": True})
            return
        if route == "/api/samples":
            self._send_json(HTTPStatus.OK, {"samples": self.server.samples})
            return
        if route == "/api/config":
            self._send_json(HTTPStatus.OK, {
                "default_correction_backend": self.server.config.default_correction_backend,
                "correction_api_configured": bool(
                    self.server.config.correction_api.get("base_url")
                    and self.server.config.correction_api.get("model")
                    and self.server.config.correction_api.get("api_key")
                ),
            })
            return
        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/artifact/(.+)", route)
        if match:
            try:
                artifact = safe_job_path(self.server.config.jobs_root(), match.group(1), match.group(2))
            except (ValueError, FileNotFoundError):
                self._send_error_json(HTTPStatus.NOT_FOUND, "artifact not found")
                return
            if artifact.suffix.lower() not in {".json", ".jsonl", ".txt", ".log"}:
                self._send_error_json(HTTPStatus.NOT_FOUND, "artifact not found")
                return
            body = artifact.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(artifact.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/media/(.+)", route)
        if match:
            try:
                media = safe_job_path(self.server.config.jobs_root(), match.group(1), match.group(2))
            except (ValueError, FileNotFoundError):
                self._send_error_json(HTTPStatus.NOT_FOUND, "media not found")
                return
            if media.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg"}:
                self._send_error_json(HTTPStatus.NOT_FOUND, "media not found")
                return
            body = media.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(media.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/events", route)
        if match:
            try:
                events = self.server.manager.events(match.group(1))
            except KeyError:
                self._send_error_json(HTTPStatus.NOT_FOUND, "job not found")
                return
            self._send_json(HTTPStatus.OK, {"events": events})
            return
        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)", route)
        if match:
            try:
                state = self.server.manager.get(match.group(1))
            except KeyError:
                self._send_error_json(HTTPStatus.NOT_FOUND, "job not found")
                return
            self._send_json(HTTPStatus.OK, state)
            return
        self._serve_static(route)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        if route == "/api/jobs":
            unknown = set(payload) - JOB_FIELDS
            if unknown:
                self._send_error_json(HTTPStatus.BAD_REQUEST, f"unknown fields: {', '.join(sorted(unknown))}")
                return
            sample_id = str(payload.get("sample_id", ""))
            sample = next((item for item in self.server.samples if item.get("sample_id") == sample_id), None)
            if sample is None:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "unknown sample_id")
                return
            options = {}
            for name, default in (
                ("vad_enabled", True),
                ("pyannote_enabled", False),
                ("correction_enabled", True),
                ("c4_enabled", False),
            ):
                value = payload.get(name, default)
                if not isinstance(value, bool):
                    self._send_error_json(HTTPStatus.BAD_REQUEST, f"{name} must be boolean")
                    return
                options[name] = value
            asr_model = str(payload.get("asr_model", "large-v3"))
            if asr_model not in ASR_MODELS:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "asr_model must be tiny or large-v3")
                return
            options["asr_model"] = asr_model
            correction_backend = str(payload.get("correction_backend", self.server.config.default_correction_backend))
            if correction_backend not in CORRECTION_BACKENDS:
                self._send_error_json(HTTPStatus.BAD_REQUEST, "correction_backend must be local or openai_compatible")
                return
            options["correction_backend"] = correction_backend
            try:
                job_id = self.server.manager.start(JobRequest(sample=dict(sample), **options))
            except (RuntimeError, ValueError) as exc:
                self._send_error_json(HTTPStatus.CONFLICT, str(exc))
                return
            self._send_json(HTTPStatus.ACCEPTED, {"job_id": job_id})
            return
        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9_-]+)/cancel", route)
        if match:
            cancelled = self.server.manager.cancel(match.group(1))
            self._send_json(HTTPStatus.OK if cancelled else HTTPStatus.CONFLICT, {"cancelled": cancelled})
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "not found")


def create_server(
    config: AppConfig,
    manager: JobManager,
    samples: Sequence[dict[str, Any]],
    *,
    static_dir: str | Path | None = None,
    address: tuple[str, int] | None = None,
) -> DemoHTTPServer:
    server = DemoHTTPServer(address or (config.host, config.port), DemoRequestHandler)
    server.config = config
    server.manager = manager
    server.samples = [dict(sample) for sample in samples]
    server.static_dir = Path(static_dir or Path(__file__).with_name("static")).resolve()
    return server
