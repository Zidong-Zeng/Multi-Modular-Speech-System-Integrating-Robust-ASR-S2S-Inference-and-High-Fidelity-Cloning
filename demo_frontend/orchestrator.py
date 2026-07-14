"""Process orchestration and durable job state for the offline demo."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import AppConfig


SAFE_JOB_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class JobRequest:
    sample: dict[str, Any]
    vad_enabled: bool = True
    pyannote_enabled: bool = False
    correction_enabled: bool = True
    c4_enabled: bool = False
    asr_model: str = "large-v3"
    correction_backend: str = "local"


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: list[str]
    python_path: str
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    critical: bool = True


Runner = Callable[[StageSpec, Path, Callable[[str], None], threading.Event], int]
Factory = Callable[[Path, JobRequest], Iterable[StageSpec]]
ResultBuilder = Callable[[Path, JobRequest, str], dict[str, Any]]


def _now() -> float:
    return round(time.time(), 3)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def default_runner(
    spec: StageSpec,
    log_path: Path,
    emit_log: Callable[[str], None],
    cancel_event: threading.Event,
) -> int:
    env = os.environ.copy()
    env.update(spec.env)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(spec.command)}\n")
        log.flush()
        process = subprocess.Popen(
            spec.command,
            cwd=str(spec.cwd),
            env=env,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        lines: queue.Queue[str | None] = queue.Queue()

        def read_lines() -> None:
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        stream_done = False
        while not stream_done:
            if cancel_event.is_set() and process.poll() is None:
                process.terminate()
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    break
                continue
            if line is None:
                stream_done = True
                continue
            clean_line = line.rstrip("\n")
            log.write(clean_line + "\n")
            log.flush()
            emit_log(clean_line)
        try:
            return_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
        reader.join(timeout=1)
        process.stdout.close()
        if cancel_event.is_set() and return_code == 0:
            return -15
        return return_code


class JobManager:
    def __init__(
        self,
        config: AppConfig,
        factory: Factory,
        runner: Runner | None = None,
        result_builder: ResultBuilder | None = None,
    ):
        self.config = config
        self.factory = factory
        self.runner = runner or default_runner
        self.result_builder = result_builder
        self.jobs_root = config.jobs_root()
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._active_job: str | None = None
        self._threads: dict[str, threading.Thread] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._recover_interrupted_jobs()

    def _recover_interrupted_jobs(self) -> None:
        for job_dir in self.jobs_root.iterdir():
            status_path = job_dir / "status.json"
            if not job_dir.is_dir() or not status_path.is_file() or not SAFE_JOB_NAME.fullmatch(job_dir.name):
                continue
            try:
                state = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if state.get("status") in {"queued", "running"}:
                self._set_status(
                    job_dir.name,
                    "interrupted",
                    previous_stage=state.get("stage", ""),
                    message="server restarted before job completion",
                )

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def _write_event(self, job_id: str, event: dict[str, Any], *, update_status: bool = False) -> None:
        event = {"timestamp": _now(), **event}
        job_dir = self._job_dir(job_id)
        with (job_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        if update_status:
            _write_json_atomic(job_dir / "status.json", event)

    def _set_status(self, job_id: str, status: str, **extra: Any) -> dict[str, Any]:
        state = {"job_id": job_id, "status": status, **extra}
        self._write_event(job_id, state, update_status=True)
        return state

    def start(self, request: JobRequest) -> str:
        duration = float(request.sample.get("duration_sec", 0) or 0)
        if duration > 30.0 and not request.vad_enabled:
            raise ValueError("VAD must be enabled for audio longer than 30 seconds")
        with self._lock:
            if self._active_job and self._threads.get(self._active_job, None) is not None:
                if self._threads[self._active_job].is_alive():
                    raise RuntimeError("another job is already running")
            job_id = uuid.uuid4().hex[:12]
            job_dir = self._job_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=False)
            _write_json_atomic(job_dir / "request.json", asdict(request))
            self._set_status(job_id, "queued")
            cancel_event = threading.Event()
            thread = threading.Thread(target=self._run, args=(job_id, request, cancel_event), daemon=True)
            self._active_job = job_id
            self._cancel_events[job_id] = cancel_event
            self._threads[job_id] = thread
            thread.start()
            return job_id

    def _run(self, job_id: str, request: JobRequest, cancel_event: threading.Event) -> None:
        job_dir = self._job_dir(job_id)
        warnings: list[dict[str, Any]] = []
        try:
            self._set_status(job_id, "running")
            for stage in self.factory(job_dir, request):
                if cancel_event.is_set():
                    self._set_status(job_id, "cancelled", stage=stage.name)
                    return
                self._set_status(job_id, "running", stage=stage.name)
                rc = self.runner(
                    stage,
                    job_dir / "pipeline.log",
                    lambda line: self._write_event(job_id, {"type": "log", "stage": stage.name, "message": line}),
                    cancel_event,
                )
                if cancel_event.is_set():
                    self._set_status(job_id, "cancelled", stage=stage.name, return_code=rc)
                    return
                if rc != 0:
                    if stage.critical:
                        self._set_status(job_id, "failed", stage=stage.name, return_code=rc)
                        return
                    warning = {"stage": stage.name, "return_code": rc, "message": "optional stage failed"}
                    warnings.append(warning)
                    self._write_event(job_id, {"type": "warning", **warning})
                    continue
                self._write_event(
                    job_id,
                    {"type": "stage", "status": "completed", "stage": stage.name, "return_code": rc},
                )
            extra = {}
            if self.result_builder is not None:
                extra["result"] = self.result_builder(job_dir, request, job_id)
            if warnings:
                extra["warnings"] = warnings
            self._set_status(job_id, "completed", **extra)
        except Exception as exc:
            self._set_status(job_id, "failed", error=str(exc))
        finally:
            with self._lock:
                self._cancel_events.pop(job_id, None)
                self._active_job = None

    def get(self, job_id: str) -> dict[str, Any]:
        path = self._job_dir(job_id) / "status.json"
        if not path.is_file():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def events(self, job_id: str) -> list[dict[str, Any]]:
        path = self._job_dir(job_id) / "events.jsonl"
        if not path.is_file():
            raise KeyError(job_id)
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def cancel(self, job_id: str) -> bool:
        event = self._cancel_events.get(job_id)
        if event is None:
            return False
        event.set()
        return True

    def wait(self, job_id: str, timeout: float | None = None) -> dict[str, Any]:
        thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout)
        return self.get(job_id)
