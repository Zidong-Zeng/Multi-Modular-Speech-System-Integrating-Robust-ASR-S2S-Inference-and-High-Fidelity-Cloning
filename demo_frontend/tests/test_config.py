import json
import struct
import unittest
import wave
from pathlib import Path
import tempfile

from demo_frontend.config import ConfigError, discover_samples, load_config, load_config_file


def write_test_wav(path: Path, sample_rate: int, frames: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack("<" + "h" * frames, *([0] * frames)))


class ConfigTests(unittest.TestCase):
    def test_load_config_resolves_paths_and_python_env(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = load_config({
                "project_root": str(root),
                "python": {"c2": "/opt/conda/envs/speech/bin/python"},
            })
            self.assertEqual(config.project_root, root.resolve())
            self.assertEqual(config.python_for("c2"), "/opt/conda/envs/speech/bin/python")

    def test_sample_index_reads_wav_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_test_wav(root / "demo.wav", sample_rate=16000, frames=32000)
            samples = discover_samples(root)
            self.assertEqual(samples[0]["duration_sec"], 2.0)
            self.assertEqual(samples[0]["audio_rel"], "demo.wav")

    def test_config_rejects_missing_python(self):
        with tempfile.TemporaryDirectory() as temp:
            config = load_config({"project_root": temp, "python": {}})
            with self.assertRaisesRegex(ConfigError, r"python\.c2"):
                config.python_for("c2")

    def test_config_file_resolves_project_root_relative_to_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_dir = root / "demo_frontend"
            config_dir.mkdir()
            path = config_dir / "config.json"
            path.write_text(json.dumps({"project_root": "..", "python": {"c2": "/py/c2"}}), encoding="utf-8")
            config = load_config_file(path)
            self.assertEqual(config.project_root, root.resolve())

    def test_load_config_reads_openai_compatible_correction_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            config = load_config({
                "project_root": temp,
                "python": {"c3": "/py/c3"},
                "correction": {
                    "default_backend": "openai_compatible",
                    "api": {
                        "base_url": "https://api.example.com/v1",
                        "model": "qwen-plus",
                        "api_key": "secret-token",
                        "temperature": 0.1,
                        "timeout": 60,
                        "max_new_tokens": 256,
                    },
                },
            })
            self.assertEqual(config.default_correction_backend, "openai_compatible")
            self.assertEqual(config.correction_api["base_url"], "https://api.example.com/v1")
            self.assertEqual(config.correction_api["model"], "qwen-plus")
            self.assertEqual(config.correction_api["api_key"], "secret-token")
            self.assertEqual(config.correction_api["max_new_tokens"], 256)


if __name__ == "__main__":
    unittest.main()
