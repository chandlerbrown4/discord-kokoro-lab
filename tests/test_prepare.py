import argparse
from array import array
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import wave

from voice_lab import prepare


def _track(path, spans):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = array("h")
    for seconds, hz in spans:
        frames = round(seconds * prepare.SOURCE_RATE)
        samples.extend(
            round(6000 * math.sin(2 * math.pi * hz * frame / prepare.SOURCE_RATE))
            for frame in range(frames)
        )
    if sys.byteorder == "big":
        samples.byteswap()
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, prepare.SOURCE_RATE, 0, "NONE", "not compressed"))
        stream.writeframes(samples.tobytes())


def _run(session, output, *options):
    parser = argparse.ArgumentParser()
    prepare.add_arguments(parser)
    prepare.run(parser.parse_args([str(session), str(output), "--no-transcribe", *options]))


@unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
class PrepareIntegrationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.session = self.root / "session"
        self.raw = self.session / "raw"
        self.raw.mkdir(parents=True)
        self.output = self.root / "prepared"

    def _write_session(self, participants, ended_at="2026-11-16T12:30:00Z"):
        metadata = {
            "schema_version": 1,
            "source": "discord",
            "participants": [
                {"id": speaker, "display_name": f"Speaker {speaker}", "consented": consented}
                for speaker, consented in participants
            ],
        }
        if ended_at is not None:
            metadata["ended_at"] = ended_at
        (self.session / "session.json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_splits_pauses_and_preserves_speaker_separation(self):
        self._write_session([("111", True), ("222", True), ("333", False), ("555", True)])
        first = self.raw / "111" / "first.wav"
        second = self.raw / "222" / "second.wav"
        _track(first, [(0.7, 0), (2, 210), (0.8, 0), (2, 210), (0.7, 0)])
        _track(second, [(2, 370)])
        _track(self.raw / "333" / "ignored.wav", [(2, 700)])
        (first.parent / "external.wav").symlink_to(second)
        (self.raw / "555").symlink_to(second.parent, target_is_directory=True)

        _run(self.session, self.output)

        metadata = json.loads((self.output / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["source"], "discord")
        self.assertEqual(metadata["skipped_unapproved_speaker_dirs"], 1)
        self.assertEqual({person["speaker_id"]: person["clips"] for person in metadata["speakers"]}, {"111": 2, "222": 1})
        self.assertEqual({path.name for path in self.output.iterdir()}, {"111", "222", "dataset.json"})
        for speaker, source, hz in (("111", first, 210), ("222", second, 370)):
            root = self.output / speaker
            rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            for row in rows:
                self.assertEqual(row["speaker_id"], speaker)
                self.assertEqual(row["source"], source.relative_to(self.session).as_posix())
                self.assertIsNone(row["text"])
                self.assertLess(row["duration_s"], 2.3)
                with wave.open(str(root / row["clip"]), "rb") as stream:
                    self.assertEqual((stream.getframerate(), stream.getnchannels(), stream.getsampwidth()), (24_000, 1, 2))
                    stream.setpos(12_000)
                    samples = array("h")
                    samples.frombytes(stream.readframes(24_000))
                if sys.byteorder == "big":
                    samples.byteswap()
                zero_crossings = sum(a <= 0 < b for a, b in zip(samples, samples[1:]))
                self.assertAlmostEqual(zero_crossings, hz, delta=2)

    def test_rejects_incomplete_discord_session_before_creating_output(self):
        self._write_session([("111", True)], ended_at=None)
        with self.assertRaisesRegex(ValueError, "ended_at"):
            _run(self.session, self.output)
        self.assertFalse(self.output.exists())

    def test_prepares_desktop_speaker_labels_and_preserves_recording_offsets(self):
        clip = "raw/1/000001.wav"
        _track(self.session / clip, [(2, 210)])
        metadata = {
            "schema_version": 1,
            "source": "discord-desktop",
            "participants": [{"id": "1", "display_name": "Speaker 1"}],
            "segments": [{"speaker_id": "1", "clip": clip, "start_s": 30.5, "end_s": 32.5}],
        }
        (self.session / "session.json").write_text(json.dumps(metadata), encoding="utf-8")

        _run(self.session, self.output)

        dataset = json.loads((self.output / "dataset.json").read_text(encoding="utf-8"))
        row = json.loads((self.output / "1" / "manifest.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(dataset["source"], "discord-desktop")
        self.assertEqual(row["source"], clip)
        self.assertEqual(row["recording_start_s"], 30.5)
        self.assertEqual(row["recording_end_s"], 32.5)

    def test_drops_silence_and_keeps_output_atomic(self):
        self._write_session([("111", True)])
        _track(self.raw / "111" / "gap.wav", [(4, 0)])
        with self.assertRaisesRegex(ValueError, "no usable clips"):
            _run(self.session, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.iterdir()), [self.session])


class SegmentationTests(unittest.TestCase):
    def test_long_audio_splits_at_short_quiet_breaks(self):
        clips = prepare._segments(22.0, [(10.6, 11.0)], 1.5, 12.0, 0.55)
        self.assertEqual(clips, [(0.0, 10.8), (10.8, 22.0)])


if __name__ == "__main__":
    unittest.main()
