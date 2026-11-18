import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from voice_lab import distill_teacher as teacher
from voice_lab import qwen_adapt as adapt


class TeacherSelectionTest(unittest.TestCase):
    @staticmethod
    def scores(scores):
        return [{"similarity": s, "wer": 0, "duration_error": .08, "pitch_range_error": 1.,
                 "original_sha256": str(i)} for i, s in enumerate(scores)]

    def test_promotion_requires_identity_words_and_delivery_on_same_lines(self):
        base = self.scores([.82, .83, .80])
        good = self.scores([.84, .84, .82])
        self.assertTrue(adapt._promote(base, good))
        for field, value in (("similarity", .64), ("similarity", float("nan")), ("pitch_range_error", float("nan")), ("wer", .4),
                             ("duration_error", .45), ("pitch_range_error", 4.)):
            bad = [*good[:-1], dict(good[-1], **{field: value})]
            self.assertFalse(adapt._promote(base, bad), field)
        self.assertFalse(adapt._promote(base, list(reversed(good))))
        self.assertFalse(adapt._promote(base, base))

    def test_too_few_training_clips_uses_original_teacher_without_loading_mlx(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "old.wav"
            new_ref, text, adapter, report = adapt.choose_teacher([], [], root, ref, "first line", None, root)
        self.assertEqual((new_ref, text, adapter), (ref, "first line", None))
        self.assertEqual(report["selected"], "base")

    def test_accepted_weights_are_reused_and_changed_training_excerpts_trigger_a_new_fit(self):
        try:
            import numpy as np
            import scipy.signal
            import soundfile as sf
        except ImportError:
            self.skipTest("The speech worker's audio dependencies are optional.")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = []
            time = np.arange(60000)/24000
            for i in range(10):
                p = root / f"{i}.wav"
                sf.write(p, .2*np.sin(2*np.pi*(100 + 8*i)*time), 24000)
                train.append(p)
            checker = Mock()
            checker.embedding_model = checker.asr_dir = None
            checker.transcribe.side_effect = lambda p: "The utterance number " + Path(p).stem
            selected = self.scores([.88, .84, .89])
            baseline = self.scores([.80, .82, .83])
            def fake_fit(model, pairs, dest, **kwargs):
                dest.write_bytes(b"trained")
                return {"fit_clips": len(pairs)}
            with (patch.object(teacher, "_Qwen"), patch.object(adapt, "_representatives", return_value=[]),
                  patch.object(adapt, "_screen_development", side_effect=[baseline, selected, baseline, selected]),
                  patch.object(adapt, "fit_adapter", side_effect=fake_fit) as fit,
                  patch.object(adapt, "apply_adapter")):
                args = ([], root/"attempts", train[0], "prior words", checker, root)
                ref, _, adapter, report = adapt.choose_teacher(train[:9], *args)
                self.assertEqual(report["selected"], "adapter")
                self.assertEqual(len(report["development_sha256"]), 3)
                self.assertFalse(set(report["fit_sha256"]) & set(report["development_sha256"]))
                self.assertTrue(ref.is_file() and adapter.is_file())
                again = adapt.choose_teacher(train[:9], *args)
                self.assertEqual(again[2], adapter)
                old_transcripts = {path: sum(call.args == (path,) for call in checker.transcribe.call_args_list)
                                   for path in train[:9]}
                newer = adapt.choose_teacher(train, *args)
                self.assertNotEqual(newer[2], adapter)
                for path in train[:9]:
                    self.assertEqual(sum(call.args == (path,) for call in checker.transcribe.call_args_list),
                                     old_transcripts[path])
                self.assertEqual(fit.call_count, 2)
                self.assertEqual(json.loads((adapter.parent/"decision.json").read_text())["selected"], "adapter")
