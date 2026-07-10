import os
import tempfile
import unittest

import c2_asr


def parse_args(*args):
    return c2_asr.build_arg_parser().parse_args(list(args))


class C2EntrypointTest(unittest.TestCase):
    def test_c2_cli_defaults_are_safe_for_optional_features(self):
        args = parse_args()

        self.assertEqual(args.asr_mode, "onebest")
        self.assertEqual(args.vad_backend, "silero")
        self.assertIs(args.diarize, False)
        self.assertIs(args.offline, False)

    def test_c2_cli_can_enable_nbest_and_pyannote_explicitly(self):
        args = parse_args("--asr_mode", "nbest", "--vad_backend", "energy", "--diarize", "--offline")

        self.assertEqual(args.asr_mode, "nbest")
        self.assertEqual(args.vad_backend, "energy")
        self.assertIs(args.diarize, True)
        self.assertIs(args.offline, True)

    def test_save_outputs_uses_nbest_filename_for_nbest_mode(self):
        predictions = [{"id": "sample-1", "chunks": [{"nbest": [{"rank": 1, "text": "hello"}]}]}]
        summary = {"engine": "whisper_nbest"}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json, summary_json = c2_asr.save_outputs(predictions, summary, tmpdir, asr_mode="nbest")

        self.assertEqual(os.path.basename(out_json), "asr_nbest_predictions.json")
        self.assertEqual(os.path.basename(summary_json), "c2_summary.json")

    def test_save_outputs_uses_onebest_filename_by_default(self):
        predictions = [{"id": "sample-1", "hypothesis": "hello"}]
        summary = {"engine": "transformers"}

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json, _ = c2_asr.save_outputs(predictions, summary, tmpdir, asr_mode="onebest")

        self.assertEqual(os.path.basename(out_json), "asr_predictions.json")


if __name__ == "__main__":
    unittest.main()
