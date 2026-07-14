import unittest
from pathlib import Path


STATIC = Path(__file__).parents[1] / "static"


class StaticContractTests(unittest.TestCase):
    def test_static_files_expose_control_contract_without_external_assets(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        js = (STATIC / "app.js").read_text(encoding="utf-8")
        css = (STATIC / "styles.css").read_text(encoding="utf-8")
        for token in ("多模态语音识别系统", "sample-select", "asr-model-select", "vad-enabled", "pyannote-enabled", "correction-enabled", "correction-backend-select", "c4-enabled", "stage-c1", "stage-c2", "stage-c3", "stage-c5", "chunk-table"):
            self.assertIn(token, html)
        self.assertIn("/api/samples", js)
        self.assertIn("/api/config", js)
        self.assertIn("/api/jobs", js)
        self.assertIn("localStorage", js)
        self.assertIn("asr_model", js)
        self.assertNotRegex(html + js + css, r"https?://|//cdn|fonts.googleapis")

    def test_static_files_have_result_and_log_regions(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        for token in ("event-log", "asr-result", "correction-result", "translation-result", "c5-audio", "c4-result"):
            self.assertIn(token, html)


if __name__ == "__main__":
    unittest.main()
