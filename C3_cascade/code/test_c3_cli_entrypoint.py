import os
import subprocess
import sys
import unittest


class C3CliEntrypointTest(unittest.TestCase):
    def test_python_module_help_prints_cli_options(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.abspath(".")

        result = subprocess.run(
            [sys.executable, "-m", "c3.cli", "--help"],
            cwd=os.path.abspath("."),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--c2_json", result.stdout)
        self.assertIn("--correction_backend", result.stdout)


if __name__ == "__main__":
    unittest.main()
