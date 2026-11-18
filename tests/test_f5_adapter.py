import argparse
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from voice_lab import f5_adapter


class F5AdapterFixtureTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.speaker = base / "101"
        self.reference = base / "prompt"
        self.reference.mkdir()
        (self.speaker / "clips").mkdir(parents=True)
        self.write_wav(self.reference / "reference.wav", 40, 4)
        (self.reference / "transcript.txt").write_text("A reference.", encoding="utf-8")
        self.rows = []
        for i in range(1, 5):
            clip = f"clips/clip-{i}.wav"
            self.write_wav(self.speaker / clip, i, 1)
            self.rows.append({"speaker_id": "101", "clip": clip, "text": f"Take {i}"})
        self.clips = [r["clip"] for r in self.rows]

    @staticmethod
    def write_wav(path, level, seconds):
        with wave.open(str(path), "wb") as writer:
            writer.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            writer.writeframes(struct.pack("<h", level) * 24000 * seconds)

    def select(self):
        manifest = self.speaker / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        return f5_adapter._examples(self.speaker, self.clips, [self.reference])

    def test_selects_explicit_order_and_requires_reviewed_text(self):
        self.clips.reverse()
        prompts, examples = self.select()
        self.assertEqual(prompts[0][1], "A reference.")
        self.assertEqual([row[0] for row in examples], self.clips)
        self.assertEqual(examples[0][2], "Take 4.")
        self.rows[1]["text"] = None
        with self.assertRaisesRegex(ValueError, "transcript"):
            self.select()

    def test_excludes_reference_audio_and_duplicate_training_content(self):
        self.write_wav(self.speaker / self.clips[1], 1, 1)
        with self.assertRaisesRegex(ValueError, "distinct audio"):
            self.select()
        self.write_wav(self.speaker / self.clips[1], 40, 1)
        with self.assertRaisesRegex(ValueError, "distinct audio"):
            self.select()

    def test_default_recipe(self):
        parser = argparse.ArgumentParser()
        commands = parser.add_subparsers()
        f5_adapter.configure(commands)
        args = parser.parse_args(["f5-fit", "101", "bundle", "--reference", "prompt", "--clip", self.clips[0]])
        self.assertEqual((args.steps, args.seed), (24, 1701))
        self.assertEqual(f5_adapter._layout()["quantization_bits"], 8)
        with self.assertRaisesRegex(ValueError, "four"):
            f5_adapter._examples(self.speaker, args.clip, [self.reference])

    def test_adapter_rejects_wrong_revision_and_corrupt_weights(self):
        layout = f5_adapter._layout()
        meta = self.reference / "adapter.json"
        meta.write_text(json.dumps(dict(layout, revision="other")), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            f5_adapter.apply_adapter(None, self.reference)
        meta.write_text(json.dumps(dict(layout, weights_sha256="bad")), encoding="utf-8")
        (self.reference / "adapter.safetensors").write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "checksum"):
            f5_adapter.apply_adapter(None, self.reference)
