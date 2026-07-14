import json
import tempfile
import unittest
from pathlib import Path

from demo_frontend.adapters import (
    build_c1_command,
    build_c2_command,
    build_c3_command,
    build_c3_env,
    build_c4_command,
    build_c5_command,
    correction_bypass_rows,
    parse_c2_artifact,
    parse_c3_artifact,
    parse_c4_artifact,
    parse_c5_artifact,
)
from demo_frontend.config import AppConfig


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("/srv/assignment_C").resolve()
        self.config = AppConfig(
            project_root=self.root,
            python_paths={
                "c1": "/opt/conda/envs/speech_zxy/bin/python",
                "c2": "/opt/conda/envs/speech/bin/python",
                "c3": "/opt/conda/envs/speech/bin/python",
                "c4": "/opt/conda/envs/speech_tcx/bin/python",
                "c5": "/opt/conda/envs/cosyvoice/bin/python",
            },
            model_paths={
                "asr": "/models/whisper-large-v3",
                "asr_tiny": "/models/whisper-tiny",
                "asr_large_v3": "/models/whisper-large-v3",
                "correction": "/models/qwen3",
                "translation": "/models/qwen",
                "c4": "/models/seamless",
            },
            script_paths={
                "c1": "C1_audio_processing/c1_advanced_augmentation.py",
                "c2": "C2_ASR/code/c2_asr.py",
                "c3": "C3_cascade/code/c3/cli.py",
                "c4": "C4_e2e_package/C4_end2end/code/c4_e2e.py",
                "c5": "C5_TTS/c5_tts.py",
            },
        )

    def test_build_c1_command_disables_extra_augmentation(self):
        command = build_c1_command(self.config, "/input/demo.wav", "/jobs/1/c1")
        self.assertIsInstance(command, list)
        self.assertEqual(command[0], self.config.python_for("c1"))
        for flag in ("--no-vad", "--no-speed", "--no-volume", "--no-noise"):
            self.assertIn(flag, command)

    def test_build_c2_command_maps_vad_and_pyannote_flags(self):
        command = build_c2_command(self.config, "/jobs/1/manifest.json", "/jobs/1/c2", vad_enabled=True, pyannote_enabled=True, correction_enabled=True, asr_model="large-v3")
        self.assertIn("--vad_backend", command)
        self.assertIn("silero", command)
        self.assertIn("--diarize", command)
        self.assertIn("--asr_mode", command)
        self.assertIn("nbest", command)

    def test_build_c2_command_sets_search_size_from_correction_toggle(self):
        enabled = build_c2_command(self.config, "/jobs/1/manifest.json", "/jobs/1/c2", vad_enabled=True, pyannote_enabled=False, correction_enabled=True, asr_model="large-v3")
        disabled = build_c2_command(self.config, "/jobs/1/manifest.json", "/jobs/1/c2", vad_enabled=True, pyannote_enabled=False, correction_enabled=False, asr_model="large-v3")

        self.assertEqual(enabled[enabled.index("--nbest") + 1], "5")
        self.assertEqual(enabled[enabled.index("--beam_size") + 1], "20")
        self.assertEqual(disabled[disabled.index("--nbest") + 1], "1")
        self.assertEqual(disabled[disabled.index("--beam_size") + 1], "1")

    def test_build_c2_command_selects_asr_model_path(self):
        tiny = build_c2_command(self.config, "/jobs/1/manifest.json", "/jobs/1/c2", vad_enabled=True, pyannote_enabled=False, correction_enabled=False, asr_model="tiny")
        large = build_c2_command(self.config, "/jobs/1/manifest.json", "/jobs/1/c2", vad_enabled=True, pyannote_enabled=False, correction_enabled=False, asr_model="large-v3")

        self.assertEqual(tiny[tiny.index("--model") + 1], "/models/whisper-tiny")
        self.assertEqual(large[large.index("--model") + 1], "/models/whisper-large-v3")

    def test_build_c3_command_is_local_and_supports_bypass(self):
        command = build_c3_command(self.config, "/jobs/1/c2/asr_nbest_predictions.json", "/jobs/1/c3", correction_enabled=False, correction_backend="local")
        self.assertIn("--disable_correction", command)
        self.assertIn("--correction_backend", command)
        self.assertIn("local", command)
        self.assertNotIn("https://", " ".join(command))

    def test_build_c3_command_supports_openai_compatible_correction_backend(self):
        config = AppConfig(
            project_root=self.root,
            python_paths=self.config.python_paths,
            model_paths=self.config.model_paths,
            script_paths=self.config.script_paths,
            correction_api={
                "base_url": "https://api.example.com/v1",
                "model": "qwen-plus",
                "api_key_env": "DEMO_CORRECTION_API_KEY",
                "api_key": "secret-token",
                "temperature": 0.2,
                "timeout": 45,
                "max_new_tokens": 512,
            },
        )
        command = build_c3_command(config, "/jobs/1/c2/asr_nbest_predictions.json", "/jobs/1/c3", correction_enabled=True, correction_backend="openai_compatible")

        self.assertEqual(command[command.index("--correction_backend") + 1], "openai_compatible")
        self.assertEqual(command[command.index("--correction_api_base") + 1], "https://api.example.com/v1")
        self.assertEqual(command[command.index("--correction_api_model") + 1], "qwen-plus")
        self.assertEqual(command[command.index("--correction_api_key_env") + 1], "DEMO_CORRECTION_API_KEY")
        self.assertEqual(command[command.index("--correction_temperature") + 1], "0.2")
        self.assertEqual(command[command.index("--correction_timeout") + 1], "45")
        self.assertEqual(command[command.index("--correction_max_new_tokens") + 1], "512")
        self.assertNotIn("secret-token", command)

        env = build_c3_env(config, correction_enabled=True, correction_backend="openai_compatible")
        self.assertEqual(env, {"DEMO_CORRECTION_API_KEY": "secret-token"})

    def test_build_c4_and_c5_commands(self):
        c4 = build_c4_command(self.config, "/jobs/1/c1", "/jobs/1/c4", "/jobs/1/c4_manifest.json")
        c5 = build_c5_command(self.config, "/jobs/1/c3/c3_predictions.json", "/jobs/1/c5")
        self.assertIn("--dataset", c4)
        self.assertIn("--offline", c4)
        self.assertEqual(c5[0], self.config.python_for("c5"))
        self.assertIn("--dataset", c5)

    def test_parse_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            c2_path = root / "asr_nbest_predictions.json"
            c2_path.write_text(json.dumps([{"id": "demo", "hypothesis": "hello", "chunks": [{"chunk_id": 0, "start_ms": 0, "end_ms": 2000, "text": "hello", "nbest": [{"rank": 1, "text": "hello"}]}]}]), encoding="utf-8")
            c3_path = root / "c3_predictions.json"
            c3_path.write_text(json.dumps([{"id": "demo", "hypothesis": "hello", "translation_zh": "你好"}]), encoding="utf-8")
            c4_path = root / "c4_results.json"
            c4_path.write_text(json.dumps([{"id": "demo", "e2e_translation_zh": "你好"}]), encoding="utf-8")
            c5_path = root / "batch_summary.json"
            c5_path.write_text(json.dumps({"results": [{"id": "demo", "wav": "s2s_demo.wav"}]}), encoding="utf-8")
            self.assertEqual(parse_c2_artifact(c2_path)["chunks"][0]["text"], "hello")
            self.assertEqual(parse_c3_artifact(c3_path)["translation_zh"], "你好")
            self.assertEqual(parse_c4_artifact(c4_path)["translation_zh"], "你好")
            self.assertEqual(parse_c5_artifact(c5_path)["wav"], "s2s_demo.wav")

    def test_correction_bypass_uses_chunk_top1(self):
        rows = [{"id": "demo", "chunks": [{"text": "hello"}, {"text": "world"}]}]
        result = correction_bypass_rows(rows)
        self.assertEqual(result[0]["corrected_transcript_en"], "hello world")
        self.assertFalse(result[0]["correction_enabled"])


if __name__ == "__main__":
    unittest.main()
