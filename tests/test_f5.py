import argparse
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from voice_lab import f5


class F5Test(unittest.TestCase):
    def test_render_resolves_every_weight_offline_by_commit(self):
        hub = Mock()
        hub.hf_hub_download.side_effect = lambda repo, filename, **_: f"/cache/{repo}/{filename}"
        with patch.dict("sys.modules", {"huggingface_hub": hub}), \
                patch.object(f5.platform, "system", return_value="Darwin"), \
                patch.object(f5.platform, "machine", return_value="arm64"):
            snapshots = f5._snapshots(local_only=True)
        for repo, (revision, filenames) in f5.WEIGHTS.items():
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
            self.assertEqual(snapshots[repo], Path("/cache", repo))
            for filename in filenames:
                hub.hf_hub_download.assert_any_call(repo, filename, revision=revision, local_files_only=True)

    def test_missing_models_leave_no_speech_or_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, transcript, output = root / "reference.wav", root / "text.txt", root / "render.wav"
            with wave.open(str(reference), "wb") as wav:
                wav.setparams((1, 2, f5.RATE, 0, "NONE", "not compressed"))
                wav.writeframes(b"\x01\x00" * (3 * f5.RATE))
            transcript.write_text("This is a test reference.", encoding="utf-8")
            parser = argparse.ArgumentParser()
            f5.configure(parser.add_subparsers())
            args = parser.parse_args(["f5-speak", "--reference", str(reference),
                                      "--transcript", str(transcript), "--text", "Hello.", "--output", str(output)])
            with patch.object(f5, "_snapshots", side_effect=RuntimeError("Missing cache")) as snapshots:
                with self.assertRaisesRegex(RuntimeError, "Missing cache"):
                    args.handler(args)
                snapshots.assert_called_once_with(local_only=True)
            self.assertEqual(set(root.iterdir()), {reference, transcript})
