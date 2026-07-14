import json
import tempfile
import unittest
from pathlib import Path

from demo_frontend.config import AppConfig
from demo_frontend.orchestrator import JobRequest
from demo_frontend.pipeline import collect_job_result, create_stage_factory


class PipelineFactoryTests(unittest.TestCase):
    def make_config(self, root: Path) -> AppConfig:
        return AppConfig(
            project_root=root,
            python_paths={"c1": "/py/c1", "c2": "/py/c23", "c3": "/py/c23", "c4": "/py/c4", "c5": "/py/c5"},
            model_paths={
                "asr": "/models/asr",
                "asr_tiny": "/models/asr-tiny",
                "asr_large_v3": "/models/asr-large",
                "correction": "/models/correction",
                "translation": "/models/translation",
                "c4": "/models/c4",
                "cosyvoice_repo": "/models/CosyVoice",
                "cosyvoice": "/models/CosyVoice2",
            },
            script_paths={
                "c1": "C1_audio_processing/c1_advanced_augmentation.py",
                "c2": "C2_ASR/code/c2_asr.py",
                "c3": "C3_cascade/code/c3/cli.py",
                "c4": "C4_e2e_package/C4_end2end/code/c4_e2e.py",
                "c5": "C5_TTS/c5_tts.py",
            },
            sample_dir=root / "samples",
            jobs_dir=root / "jobs",
        )

    def test_factory_writes_manifests_and_orders_optional_c4_last(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "samples").mkdir()
            (root / "samples" / "demo.wav").write_bytes(b"RIFF")
            request = JobRequest(
                sample={"sample_id": "sample1", "audio_rel": "demo.wav", "duration_sec": 2.0},
                vad_enabled=True,
                pyannote_enabled=True,
                correction_enabled=False,
                c4_enabled=True,
                asr_model="tiny",
            )
            job_dir = root / "jobs" / "job1"
            job_dir.mkdir(parents=True)
            stages = list(create_stage_factory(self.make_config(root))(job_dir, request))
            self.assertEqual([stage.name for stage in stages], ["c1", "c2", "c3", "c5", "c4"])
            c2_manifest = json.loads((job_dir / "c2_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(c2_manifest[0]["audio"], str(job_dir / "c1" / "demo_clean.wav"))
            self.assertIn("--disable_correction", stages[2].command)
            self.assertEqual(stages[1].command[stages[1].command.index("--model") + 1], "/models/asr-tiny")
            self.assertEqual(stages[4].python_path, "/py/c4")
            self.assertEqual(stages[3].env["COSYVOICE_MODEL"], "/models/CosyVoice2")

    def test_factory_rejects_sample_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "samples").mkdir()
            job_dir = root / "jobs" / "job1"
            job_dir.mkdir(parents=True)
            request = JobRequest(sample={"sample_id": "x", "audio_rel": "../secret.wav", "duration_sec": 2.0})
            with self.assertRaisesRegex(ValueError, "sample path"):
                list(create_stage_factory(self.make_config(root))(job_dir, request))

    def test_collect_job_result_joins_c2_c3_c5_and_optional_c4(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            for directory in ("c2", "c3", "c5", "c4"):
                (job / directory).mkdir()
            (job / "c2" / "asr_nbest_predictions.json").write_text(json.dumps([{"hypothesis": "hello", "chunks": [{"text": "hello", "start_ms": 0, "end_ms": 1000, "nbest": [{"text": "hello"}]}]}]), encoding="utf-8")
            (job / "c3" / "c3_predictions.json").write_text(json.dumps([{"hypothesis": "hello", "translation_zh": "你好"}]), encoding="utf-8")
            (job / "c5" / "batch_summary.json").write_text(json.dumps({"results": [{"wav": "s2s_demo.wav"}]}), encoding="utf-8")
            (job / "c4" / "c4_results.json").write_text(json.dumps([{"e2e_translation_zh": "你好（C4）"}]), encoding="utf-8")
            result = collect_job_result(job, c4_enabled=True)
            self.assertEqual(result["asr"], "hello")
            self.assertEqual(result["translation"], "你好")
            self.assertEqual(result["c5_audio_rel"], "c5/s2s_demo.wav")
            self.assertEqual(result["c4_translation"], "你好（C4）")

    def test_collect_job_result_keeps_main_result_when_c4_artifact_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            for directory in ("c2", "c3", "c5"):
                (job / directory).mkdir()
            (job / "c2" / "asr_nbest_predictions.json").write_text(json.dumps([{"hypothesis": "hello", "chunks": []}]), encoding="utf-8")
            (job / "c3" / "c3_predictions.json").write_text(json.dumps([{"hypothesis": "hello", "translation_zh": "你好"}]), encoding="utf-8")
            (job / "c5" / "batch_summary.json").write_text(json.dumps({"results": [{"wav": "s2s_demo.wav"}]}), encoding="utf-8")
            result = collect_job_result(job, c4_enabled=True)
            self.assertEqual(result["translation"], "你好")
            self.assertIn("c4_error", result)


if __name__ == "__main__":
    unittest.main()
