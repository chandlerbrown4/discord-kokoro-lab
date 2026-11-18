import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

from voice_lab import fit


# Emulate two checkpoint improvements whose rounded scores are identical.
_TRAINER = """
import pathlib
import sys

assert sys.argv[sys.argv.index('--step_limit') + 1] == '101'
assert pathlib.Path('transcript.txt').read_text() == 'Approved reference.'
assert pathlib.Path('reference.wav').is_file()
run = pathlib.Path('out/fitted_reference_any_timestamp')
run.mkdir(parents=True)
(run / 'fitted_9_82.00_0.80_reference.pt').write_bytes(b'earlier improvement')
(run / 'fitted_100_82.00_0.80_reference.pt').write_bytes(b'later improvement')
"""
_EXPORTER = "import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2])"
_SCORER = '''
def penalty(self, features, tempo):
    features["tempo"] = float(tempo)
    for key, value in features.items():
        if True:
            diff = abs((value - self.target_features[key])/self.target_features[key])
'''


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.STDOUT, text=True).strip()


class FitTest(unittest.TestCase):
    def test_reports_no_improved_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "did not beat its starting voice"):
                fit._best_pack(Path(temp))

    def test_runs_pinned_trainer_and_publishes_latest_improvement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trainer = root / "trainer checkout"
            trainer.mkdir()
            _git(trainer, "init", "--quiet")
            (trainer / "main.py").write_text(_TRAINER, encoding="utf-8")
            for dirname in ("utilities", "voices"):
                folder = trainer / dirname
                folder.mkdir()
                (folder / "placeholder").touch()
            (trainer / "utilities" / "fitness_scorer.py").write_text(_SCORER, encoding="utf-8")
            _git(trainer, "add", ".")
            _git(trainer, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Trainer fixture")
            revision = _git(trainer, "rev-parse", "HEAD")
            (trainer / "main.py").write_text("raise SystemExit('Do not execute this worktree change')\n", encoding="utf-8")
            reference = root / "synthetic reference.wav"
            with wave.open(str(reference), "wb") as audio:
                audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\0" * 24000)
            transcript = root / "reviewed transcript.txt"
            transcript.write_text("Approved reference.\n", encoding="utf-8")
            output = root / "voice pack.pt"
            args = argparse.Namespace(reference=reference, transcript=transcript, trainer=trainer,
                                      output=output, trainer_python=Path(sys.executable), device="cpu", steps=101)
            with patch.object(fit, "TRAINER_REVISION", revision), patch.object(fit, "_EXPORT_PACK", _EXPORTER):
                self.assertEqual(fit.run(args), 0)
                with self.assertRaises(FileExistsError):
                    fit.run(args)

            self.assertEqual(output.read_bytes(), b"later improvement")
            self.assertFalse((trainer / "out").exists())
