import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voice_lab import distill_student as student


class CorpusTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        self.manifest = self.corpus / "manifest.jsonl"
        self.train = {"text": "The first line.", "audio": "train.wav", "split": "train"}
        self.valid = {"text": "A new line.", "audio": "validation.wav", "split": "validation"}
        for row in (self.train, self.valid):
            (self.corpus / row["audio"]).write_bytes(row["audio"].encode())

    def write(self, *rows):
        self.manifest.write_text("\n".join(json.dumps(r) for r in rows))

    def test_relative_corpus_is_portable(self):
        self.write(self.train, self.valid)
        splits = student.read_manifest(self.manifest)
        self.assertEqual(splits["train"][0].path, self.corpus / "train.wav")
        self.assertEqual(splits["validation"][0].text, "A new line.")

    def test_validation_text_does_not_leak_into_training(self):
        self.write(self.train, {**self.valid, "text": "The  FIRST line!"})
        with self.assertRaisesRegex(ValueError, "between splits"):
            student.read_manifest(self.manifest)

    def test_audio_cannot_be_reused_across_splits(self):
        self.write(self.train, {**self.valid, "audio": "train.wav"})
        with self.assertRaisesRegex(ValueError, "Repeated audio"):
            student.read_manifest(self.manifest)

    def test_copied_audio_cannot_leak_between_splits(self):
        (self.corpus / "validation.wav").write_bytes((self.corpus / "train.wav").read_bytes())
        self.write(self.train, self.valid)
        with self.assertRaisesRegex(ValueError, "between splits"):
            student.read_manifest(self.manifest)

    def test_audio_cannot_escape_through_symlink(self):
        (self.root / "other.wav").touch()
        (self.corpus / "outside.wav").symlink_to(self.root / "other.wav")
        self.write(self.train, {**self.valid, "audio": "outside.wav"})
        with self.assertRaisesRegex(ValueError, "inside the teacher corpus"):
            student.read_manifest(self.manifest)

    def test_requires_a_distinct_validation_split(self):
        self.write(self.train)
        with self.assertRaisesRegex(ValueError, "validation pair"):
            student.read_manifest(self.manifest)

    def test_checkpoint_matches_recording_contents_text_and_recipe(self):
        self.write(self.train, self.valid)
        def key(steps=24):
            return student._checkpoint_key(student.read_manifest(self.manifest), steps)
        start = key()
        self.assertNotEqual(start, key(128))
        self.write({**self.train, "text": "Revised text."}, self.valid)
        self.assertNotEqual(start, key())
        self.write(self.train, self.valid)
        (self.corpus / self.train["audio"]).write_bytes(b"replacement audio")
        self.assertNotEqual(start, key())


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("scipy"),
                     "requires the optional training runtime")
class AlignmentTests(unittest.TestCase):
    def test_interrupted_checkpoint_write_preserves_previous_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fit-progress.pt"
            student._save_progress(path, {"key": "same", "step": 4})
            def interrupted(state, stream):
                stream.write(b"incomplete file")
                raise OSError("interrupted write")
            with patch("torch.save", side_effect=interrupted), self.assertRaises(OSError):
                student._save_progress(path, {"key": "same", "step": 8})
            self.assertEqual(student._load_progress(path, "same")["step"], 4)
            self.assertNotIn("step", student._load_progress(path, "different"))
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_monotone_alignment_handles_unequal_speaking_rates(self):
        import torch

        source = torch.tensor([[2., 3., 4., 0., -2., 1.], [-2., 1., 3., -3., 1., 2.]])
        target = source[:, [0, 1, 1, 1, 2, 3, 4, 5, 5]]
        mapped = student._alignment(source, target, torch.tensor([2, 3, 1]))
        self.assertEqual(len(mapped), target.shape[-1])
        self.assertEqual(mapped[0].item(), 0)
        self.assertEqual(mapped[-1].item(), 2)
        self.assertTrue((mapped[1:] >= mapped[:-1]).all())
        self.assertEqual(set(mapped.tolist()), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
