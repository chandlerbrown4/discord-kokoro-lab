import json
import shutil
import tempfile
import unittest
import wave
from argparse import ArgumentParser
from pathlib import Path

from voice_lab import audition


@unittest.skipUnless(shutil.which("ffmpeg"), "uses FFmpeg")
class AuditionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name, period, peak in (("reference", 100, 3000), ("baseline", 100, 1000), ("candidate", 200, 6000)):
            with wave.open(str(self.root / f"{name}.wav"), "wb") as wav:
                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                period_bytes = (peak.to_bytes(2, "little", signed=True) * (period // 2)
                                + (-peak).to_bytes(2, "little", signed=True) * (period // 2))
                wav.writeframes(period_bytes * (24000 // period))

    def trial(self, dest, *extra):
        parser = ArgumentParser()
        audition.configure(parser.add_subparsers())
        args = parser.parse_args(["audition", "--reference", str(self.root / "reference.wav"),
                                  "--baseline", str(self.root / "baseline.wav"),
                                  "--candidate", str(self.root / "candidate.wav"),
                                  "--output", str(self.root / dest), "--seed", "91", *extra])
        args.handler(args)
        return self.root / dest

    def test_level_matches_preserves_order_and_can_replay(self):
        out = self.trial("first", "--different-text")
        again = self.trial("second", "--different-text")
        self.assertEqual((out / "listen.wav").read_bytes(), (again / "listen.wav").read_bytes())
        with wave.open(str(out / "listen.wav"), "rb") as wav:
            self.assertEqual(wav.getnframes(), 24000 * 3 + 18000 * 2)
            playback = wav.readframes(wav.getnframes())
        key = json.loads((out / "key.json").read_text())
        self.assertEqual({key["A"], key["B"]}, {"baseline", "candidate"})
        self.assertEqual(key["reference_text"], "different")
        for role, start in (("reference", 0), (key["A"], 42000), (key["B"], 84000)):
            half_period = 100 if role == "candidate" else 50
            self.assertEqual(int.from_bytes(playback[2*start:2*start+2], "little", signed=True), 3277)
            pos = 2 * (start + half_period)
            self.assertEqual(int.from_bytes(playback[pos:pos+2], "little", signed=True), -3277)

    def test_rejects_duplicate_audio_and_silence(self):
        with self.assertRaisesRegex(ValueError, "identical audio"):
            self.trial("dup", "--candidate", str(self.root / "baseline.wav"))
        with wave.open(str(self.root / "candidate.wav"), "wb") as wav:
            wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            wav.writeframes(bytes(48000))
        with self.assertRaisesRegex(ValueError, "no audible signal"):
            self.trial("empty")
        self.assertFalse((self.root / "empty").exists())
