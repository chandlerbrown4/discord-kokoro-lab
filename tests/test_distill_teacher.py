import json
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from voice_lab import distill_teacher as teacher

try:
    import numpy as np
    import scipy
    import soundfile
except ImportError:
    np = None


class PromptTests(unittest.TestCase):
    def test_word_errors_and_distinct_custom_or_builtin_prompts(self):
        self.assertEqual(teacher.word_error_rate("I’ll go, now!", "I'll go now"), 0)
        self.assertEqual(teacher.word_error_rate("Where is the cat", "Where was the cat"), .25)
        self.assertEqual(list(teacher._prompts(["CAN I?", "Can I.", "Yes!"])), ["CAN I?", "Yes!"])
        lines = list(islice(teacher._prompts(None), 4000))
        self.assertEqual(len({tuple(teacher._words(line)) for line in lines}), 4000)


@unittest.skipIf(np is None, "requires NumPy, scipy and soundfile")
class CorpusTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.reference = self.root / "reference.wav"
        frames = np.arange(teacher.RATE * 4, dtype=np.float32) / teacher.RATE
        self.signal = .15 * np.sin(2 * np.pi * 200 * frames)
        teacher._wav(self.reference, self.signal)
        self.corpus = self.root / "corpus"
        self.text = "the original spoken reference"
        self.generations = []
        self.heard = self.text
        self.similarity = .85
        self.crash_on = None

        class Checker:
            def __init__(checker, owner):
                checker.owner = owner

            def transcribe(checker, path):
                return checker.owner.heard

            def embedding(checker, path):
                return np.array([1, 0]) if path == self.reference else np.array([self.similarity, .1])

        self.checker = Checker(self)
        self.prompts = ["Same held out line!", "the first entirely new line", "the second entirely new line",
                        "the third entirely new line", "the fourth entirely new line", "the fifth entirely new line"]

    def run_corpus(self, count, **opts):
        class FakeTeacher:
            def __init__(model, *args):
                pass

            def render(model, line, seed):
                if line == self.crash_on:
                    raise InterruptedError
                self.generations.append(line)
                self.heard = line
                return self.signal

        with patch.object(teacher, "_Qwen", FakeTeacher):
            return teacher.generate_corpus(self.corpus, self.reference, self.text, count=count, model_dir=self.root,
                                           prompts=self.prompts, excluded_texts=["same held-out line"],
                                           checker=self.checker, **opts)

    def test_adapted_teacher_has_its_own_corpus_and_immutable_manifest(self):
        first = self.run_corpus(3, freeze_manifest=True)
        evidence = first.read_bytes()
        adapter = self.root / "target.safetensors"
        adapter.write_bytes(b"test-adapter")
        with patch("voice_lab.qwen_adapt.apply_adapter") as apply:
            alternate = self.run_corpus(3, adapter=adapter, freeze_manifest=True)
            self.assertNotEqual(alternate, first)
            self.assertEqual(first.read_bytes(), evidence)
            apply.assert_called_once()
            self.assertEqual(self.run_corpus(3, adapter=adapter, freeze_manifest=True), alternate)
            apply.assert_called_once()

    def test_disjoint_text_resume_extend_and_repair_damaged_wav(self):
        self.crash_on = self.prompts[2]
        with self.assertRaises(InterruptedError):
            self.run_corpus(3)
        self.assertFalse((self.corpus / "manifest.jsonl").exists())
        self.crash_on = None
        manifest = self.run_corpus(3)
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual([r["text"] for r in rows], self.prompts[1:4])
        self.assertEqual([r["split"] for r in rows], ["train", "validation", "train"])
        self.assertEqual(self.generations, self.prompts[1:4])
        (manifest.parent / rows[0]["audio"]).write_bytes(b"corrupted")
        self.run_corpus(5)
        self.assertEqual(self.generations[3:], [self.prompts[1], *self.prompts[4:]])
        self.assertEqual(soundfile.info(manifest.parent / rows[0]["audio"]).samplerate, teacher.RATE)

    def test_failed_qa_is_auditable_and_cannot_enter_manifest(self):
        self.similarity = .2
        with self.assertRaisesRegex(ValueError, "Only 0"):
            self.run_corpus(3)
        self.assertFalse((self.corpus / "manifest.jsonl").is_file())
        rejected = [json.loads(p.read_text()) for p in self.corpus.rglob("*.json") if p.name != "recipe.json"]
        self.assertTrue(all(not r["accepted"] and len(r["attempts"]) == 2 for r in rejected))
        self.assertEqual(len(rejected), len(self.prompts) - 1)

    def test_reference_rejects_duplicated_pcm_and_reuses_verified_transcript(self):
        duplicate = self.root / "heldout.wav"
        soundfile.write(duplicate, soundfile.read(self.reference)[0], teacher.RATE, subtype="FLOAT")
        out = self.root / "reference"
        with self.assertRaisesRegex(ValueError, "different recordings"):
            teacher.prepare_reference([self.reference], out, holdouts=[duplicate], checker=self.checker)
        with patch.object(self.checker, "transcribe", wraps=self.checker.transcribe) as transcribe:
            wav, text = teacher.prepare_reference([self.reference], out, checker=self.checker)
            self.assertTrue(wav.is_file())
            self.assertEqual(text, self.text)
            teacher.prepare_reference([self.reference], out, checker=self.checker)
            transcribe.assert_called_once()
