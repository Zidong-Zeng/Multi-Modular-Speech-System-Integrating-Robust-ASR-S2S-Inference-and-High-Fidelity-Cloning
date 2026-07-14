import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib import error, request

from demo_frontend.config import AppConfig
from demo_frontend.server import create_server, safe_job_path


class FakeManager:
    def __init__(self):
        self.last_request = None

    def start(self, job_request):
        self.last_request = job_request
        return "job123"

    def get(self, job_id):
        if job_id != "job123":
            raise KeyError(job_id)
        return {"job_id": job_id, "status": "completed"}

    def events(self, job_id):
        return [{"job_id": job_id, "status": "completed"}]

    def cancel(self, job_id):
        return job_id == "job123"


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        static = root / "static"
        static.mkdir()
        (static / "index.html").write_text("<h1>demo</h1>", encoding="utf-8")
        self.config = AppConfig(project_root=root, jobs_dir=root / "jobs")
        self.samples = [{"sample_id": "sample1", "audio_rel": "demo.wav", "duration_sec": 2.0}]
        self.manager = FakeManager()
        self.server = create_server(self.config, self.manager, self.samples, static_dir=static, address=("127.0.0.1", 0))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.opener = request.build_opener(request.ProxyHandler({}))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def get_json(self, path):
        with self.opener.open(self.base + path) as response:
            return response.status, json.load(response)

    def post_json(self, path, payload):
        req = request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.opener.open(req) as response:
            return response.status, json.load(response)

    def test_health_samples_and_static_index(self):
        status, health = self.get_json("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["offline"])
        _, samples = self.get_json("/api/samples")
        self.assertEqual(samples["samples"][0]["sample_id"], "sample1")
        with self.opener.open(self.base + "/") as response:
            self.assertIn(b"demo", response.read())

    def test_create_job_accepts_boolean_options(self):
        status, payload = self.post_json("/api/jobs", {
            "sample_id": "sample1",
            "vad_enabled": True,
            "pyannote_enabled": False,
            "correction_enabled": True,
            "correction_backend": "openai_compatible",
            "c4_enabled": False,
            "asr_model": "tiny",
        })
        self.assertEqual(status, 202)
        self.assertEqual(payload["job_id"], "job123")
        self.assertTrue(self.manager.last_request.vad_enabled)
        self.assertEqual(self.manager.last_request.asr_model, "tiny")
        self.assertEqual(self.manager.last_request.correction_backend, "openai_compatible")

    def test_create_job_rejects_unknown_asr_model(self):
        with self.assertRaises(error.HTTPError) as caught:
            self.post_json("/api/jobs", {"sample_id": "sample1", "asr_model": "medium"})
        self.assertEqual(caught.exception.code, 400)

    def test_create_job_rejects_unknown_fields(self):
        with self.assertRaises(error.HTTPError) as caught:
            self.post_json("/api/jobs", {"sample_id": "sample1", "command": "rm -rf /"})
        self.assertEqual(caught.exception.code, 400)

    def test_job_status_events_and_cancel(self):
        _, state = self.get_json("/api/jobs/job123")
        self.assertEqual(state["status"], "completed")
        _, events = self.get_json("/api/jobs/job123/events")
        self.assertEqual(events["events"][0]["status"], "completed")
        status, cancelled = self.post_json("/api/jobs/job123/cancel", {})
        self.assertEqual(status, 200)
        self.assertTrue(cancelled["cancelled"])

    def test_safe_job_path_rejects_escape(self):
        root = Path(self.temp.name) / "jobs"
        job = root / "abc"
        job.mkdir(parents=True)
        (job / "result.json").write_text("{}", encoding="utf-8")
        self.assertEqual(safe_job_path(root, "abc", "result.json"), job / "result.json")
        with self.assertRaises(ValueError):
            safe_job_path(root, "abc", "../outside.json")
        with self.assertRaises(ValueError):
            safe_job_path(root, "../abc", "result.json")

    def test_media_route_serves_only_job_files(self):
        job = self.config.jobs_root() / "job123" / "c5"
        job.mkdir(parents=True)
        (job / "speech.wav").write_bytes(b"RIFFdemo")
        with self.opener.open(self.base + "/api/jobs/job123/media/c5/speech.wav") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"RIFFdemo")
        with self.assertRaises(error.HTTPError) as caught:
            self.opener.open(self.base + "/api/jobs/job123/media/../request.json")
        self.assertIn(caught.exception.code, (400, 404))

    def test_artifact_route_serves_json(self):
        job = self.config.jobs_root() / "job123"
        job.mkdir(parents=True, exist_ok=True)
        (job / "result.json").write_text('{"ok": true}', encoding="utf-8")
        with self.opener.open(self.base + "/api/jobs/job123/artifact/result.json") as response:
            self.assertEqual(response.headers.get_content_type(), "application/json")
            self.assertEqual(json.loads(response.read()), {"ok": True})


if __name__ == "__main__":
    unittest.main()
