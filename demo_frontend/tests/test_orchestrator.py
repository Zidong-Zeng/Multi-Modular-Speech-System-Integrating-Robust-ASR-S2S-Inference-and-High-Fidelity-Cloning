import json
import tempfile
import threading
import time
import unittest
import sys
from pathlib import Path

from demo_frontend.config import AppConfig
from demo_frontend.orchestrator import JobManager, JobRequest, StageSpec, default_runner


class OrchestratorTests(unittest.TestCase):
    def make_config(self, root: Path) -> AppConfig:
        return AppConfig(project_root=root, jobs_dir=root / "jobs", max_single_job=True)

    def request(self, *, duration=2.0, c4=False, vad=True):
        return JobRequest(
            sample={"sample_id": "demo", "audio_rel": "demo.wav", "duration_sec": duration},
            vad_enabled=vad,
            pyannote_enabled=False,
            correction_enabled=True,
            c4_enabled=c4,
        )

    def test_pipeline_emits_ordered_events_and_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []

            def factory(job_dir, request):
                return [StageSpec(name=name, command=[name], python_path="python3", cwd=job_dir) for name in ("c1", "c2", "c3", "c5")]

            def runner(spec, log_path, emit, cancel_event):
                calls.append(spec.name)
                return 0

            manager = JobManager(self.make_config(root), factory, runner=runner, result_builder=lambda job_dir, request, job_id: {"ok": True})
            job_id = manager.start(self.request())
            state = manager.wait(job_id, timeout=2)
            self.assertEqual(state["status"], "completed")
            self.assertTrue(state["result"]["ok"])
            self.assertEqual(calls, ["c1", "c2", "c3", "c5"])
            events = [json.loads(line) for line in (root / "jobs" / job_id / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["status"], "queued")
            self.assertEqual(events[-1]["status"], "completed")
            completed_stages = [event["stage"] for event in events if event.get("type") == "stage" and event.get("status") == "completed"]
            self.assertEqual(completed_stages, ["c1", "c2", "c3", "c5"])

    def test_long_audio_without_vad_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager = JobManager(self.make_config(root), lambda *_: [])
            with self.assertRaisesRegex(ValueError, "VAD must be enabled"):
                manager.start(self.request(duration=31.0, vad=False))

    def test_c4_runs_after_main_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = []

            def factory(job_dir, request):
                names = ["c1", "c2", "c3", "c5"] + (["c4"] if request.c4_enabled else [])
                return [StageSpec(name=name, command=[name], python_path="python3", cwd=job_dir) for name in names]

            def runner(spec, log_path, emit, cancel_event):
                calls.append(spec.name)
                return 0

            manager = JobManager(self.make_config(root), factory, runner=runner)
            job_id = manager.start(self.request(c4=True))
            self.assertEqual(manager.wait(job_id, timeout=2)["status"], "completed")
            self.assertEqual(calls, ["c1", "c2", "c3", "c5", "c4"])

    def test_optional_c4_failure_does_not_fail_main_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def factory(job_dir, request):
                return [
                    StageSpec(name="c5", command=["c5"], python_path="python3", cwd=job_dir),
                    StageSpec(name="c4", command=["c4"], python_path="python3", cwd=job_dir, critical=False),
                ]

            def runner(spec, log_path, emit, cancel_event):
                return 1 if spec.name == "c4" else 0

            manager = JobManager(self.make_config(root), factory, runner=runner)
            job_id = manager.start(self.request(c4=True))
            state = manager.wait(job_id, timeout=2)
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["warnings"][0]["stage"], "c4")

    def test_cancel_marks_job_cancelled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            started = threading.Event()

            def factory(job_dir, request):
                return [StageSpec(name="c1", command=["c1"], python_path="python3", cwd=job_dir)]

            def runner(spec, log_path, emit, cancel_event):
                started.set()
                while not cancel_event.is_set():
                    time.sleep(0.01)
                return -15

            manager = JobManager(self.make_config(root), factory, runner=runner)
            job_id = manager.start(self.request())
            self.assertTrue(started.wait(1))
            self.assertTrue(manager.cancel(job_id))
            self.assertEqual(manager.wait(job_id, timeout=2)["status"], "cancelled")

    def test_default_runner_can_cancel_silent_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cancel = threading.Event()
            result = []
            spec = StageSpec(
                name="silent",
                command=[sys.executable, "-c", "import time; time.sleep(5)"],
                python_path=sys.executable,
                cwd=root,
            )
            thread = threading.Thread(target=lambda: result.append(default_runner(spec, root / "log.txt", lambda _: None, cancel)))
            thread.start()
            time.sleep(0.1)
            cancel.set()
            thread.join(1.5)
            self.assertFalse(thread.is_alive())
            self.assertNotEqual(result[0], 0)

    def test_startup_marks_stale_running_job_interrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_job = root / "jobs" / "oldjob"
            old_job.mkdir(parents=True)
            (old_job / "status.json").write_text(json.dumps({"job_id": "oldjob", "status": "running", "stage": "c2"}), encoding="utf-8")
            (old_job / "events.jsonl").write_text("", encoding="utf-8")
            manager = JobManager(self.make_config(root), lambda *_: [])
            state = manager.get("oldjob")
            self.assertEqual(state["status"], "interrupted")
            self.assertEqual(state["previous_stage"], "c2")


if __name__ == "__main__":
    unittest.main()
