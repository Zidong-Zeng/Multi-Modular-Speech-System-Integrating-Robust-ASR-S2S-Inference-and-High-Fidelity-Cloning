import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from demo_frontend.config import AppConfig
from demo_frontend.start_server import _path_exists, dry_run_report


class StartServerTests(unittest.TestCase):
    def test_dry_run_reports_paths_without_importing_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            samples = root / "samples"
            samples.mkdir()
            config = AppConfig(
                project_root=root,
                python_paths={"c1": "/missing/c1", "c2": "/missing/c2", "c3": "/missing/c3", "c4": "/missing/c4", "c5": "/missing/c5"},
                model_paths={"asr": "/models/asr"},
                script_paths={"c1": "c1.py"},
                sample_dir=samples,
            )
            report = dry_run_report(config)
            self.assertFalse(report["ready"])
            self.assertEqual(report["sample_count"], 0)
            self.assertTrue(any(item["name"] == "python.c2" for item in report["checks"]))

    def test_path_check_treats_permission_error_as_unavailable(self):
        with patch.object(Path, "exists", side_effect=PermissionError("denied")):
            self.assertFalse(_path_exists("/root/private-model"))


if __name__ == "__main__":
    unittest.main()
