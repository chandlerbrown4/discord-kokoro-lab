import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from voice_lab import beta_enrollment as en
from voice_lab.beta_fit import _passes_quality, _promote

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "requires NumPy")
class EnrollmentTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)
        self.store = en.Enrollment(self.path / "session", self.path / "models")
        self.a = np.array([1, 0, 0], dtype="float32")
        self.b = np.array([0, 1, 0], dtype="float32")
        self.count = 0

    def ingest(self, groups):
        self.count += 1
        path = self.path / f"chunk-{self.count}.wav"
        path.write_bytes(str(self.count).encode())
        turns = {label: [(np.full(6 * en.RATE, value, dtype="<i2"), 5.1, emb)
                         for value, emb in values] for label, values in groups.items()}
        with patch.object(self.store, "_analyze", return_value=turns):
            return self.store.ingest(path)

    def test_chunk_labels_can_swap_and_retries_and_replayed_turns_do_not_add_evidence(self):
        self.ingest({0: [(1, self.a)], 1: [(2, self.b)]})
        self.ingest({0: [(3, self.b)], 1: [(4, self.a)]})
        state = self.ingest({0: [(4, self.a)], 1: [(5, self.b)]})
        self.assertEqual([r["clip_count"] for r in state["speakers"]], [2, 3])
        self.store = en.Enrollment(self.path / "session", self.path / "models")
        with patch.object(self.store, "_analyze", side_effect=AssertionError("reprocessed chunk")):
            self.assertEqual(state, self.store.ingest(self.path / "chunk-3.wav"))
        self.assertTrue(self.store.sample("1").is_file())
        with self.assertRaisesRegex(ValueError, "more clean"):
            self.store.reference("2")

    def test_ambiguous_identity_is_not_folded_into_an_existing_speaker(self):
        self.ingest({0: [(1, self.a)], 1: [(2, self.b)]})
        state = self.ingest({0: [(3, en._unit(self.a + self.b))]})
        self.assertEqual([r["clip_count"] for r in state["speakers"]], [1, 1])

    def test_inconsistent_turns_under_one_native_label_are_separated(self):
        state = self.ingest({0: [(1, self.a), (2, self.b), (3, self.a), (4, self.b)]})
        self.assertEqual([r["clip_count"] for r in state["speakers"]], [2, 2])

    def test_reference_and_validation_are_disjoint_and_fit_is_persisted_once(self):
        for i in range(6):
            self.ingest({0: [(10+i, self.a)]})
        speaker = self.store.snapshot()["speakers"][0]
        self.assertTrue(speaker["ready"])
        self.assertTrue(speaker["needs_fit"])
        self.assertEqual(speaker["model_status"], "ready_to_fit")
        vector = np.ones((1, 256, 1), dtype="float32")
        def fit(reference, training, held, embed, **kwargs):
            self.assertEqual(set(training), set(self.store.training_clips("1")))
            self.assertFalse(set(training) & set(held))
            self.assertEqual([len(training), len(held)], [4, 2])
            with wave.open(str(reference), "rb") as clip:
                values = set(np.frombuffer(clip.readframes(clip.getnframes()), "<i2").tolist())
            self.assertEqual(values, {0, 10, 12, 13, 14})
            return vector, {"selected": "fitted"}
        with patch("voice_lab.beta_fit.fit_target", side_effect=fit) as trainer:
            self.store.fit("1")
            state = self.store.fit("1")["speakers"][0]
            trainer.assert_called_once()
        self.assertEqual(state["model_status"], "optimized")
        self.assertFalse(state["needs_fit"])
        np.testing.assert_array_equal(np.load(self.store.adapter("1"), allow_pickle=False), vector)
        self.assertTrue(self.store.reference("1").is_file())

    def test_clipping_silence_and_short_turns_are_excluded(self):
        for audio in (np.zeros(en.RATE*3), np.ones(en.RATE*3), np.ones(en.RATE)*.08):
            self.assertIsNone(en._clean_turn(audio))
        rng = np.random.default_rng(27)
        pcm, voiced = en._clean_turn(np.concatenate((np.zeros(en.RATE), rng.normal(0, .1, 3*en.RATE))))
        self.assertAlmostEqual(voiced, 3., places=2)
        self.assertLess(len(pcm) / en.RATE, 3.1)

    def test_subsample_shifted_replay_with_different_edges_is_ignored(self):
        rng = np.random.default_rng(21)
        voice = rng.normal(0, 1400, 5*en.RATE).astype("<i2")
        cut = voice[1193:len(voice)-715] // 2
        self.assertTrue(en._same_excerpt(en._audio_signature(voice), en._audio_signature(cut)))
        other = rng.normal(0, 1400, len(cut)).astype("<i2")
        self.assertFalse(en._same_excerpt(en._audio_signature(voice), en._audio_signature(other)))

    def test_weak_profile_requests_more_evidence_and_can_retry(self):
        for i in range(6):
            self.ingest({0: [(20+i, self.a)]})
        with patch("voice_lab.beta_fit.fit_target", return_value=(None, {"passes_quality": False})):
            result = self.store.fit("1")["speakers"][0]
        self.assertEqual(result["model_status"], "needs_more_audio")
        self.assertFalse(result["ready"])
        self.assertFalse(result["needs_fit"])
        for i in range(6):
            result = self.ingest({0: [(40+i, self.a)]})["speakers"][0]
        self.assertTrue(result["needs_fit"])
        with patch("voice_lab.beta_fit.fit_target", return_value=(None, {"passes_quality": True})) as refit:
            result = self.store.fit("1")["speakers"][0]
        self.assertIsNotNone(refit.call_args.kwargs["previous"])
        self.assertEqual(result["model_status"], "reference_only")
        with patch.object(en.vc, "read_audio", return_value=np.full(32000, .1)), patch.object(
                self.store, "_embed", return_value=self.b):
            scores = self.store.score_preview("1", self.path / "converted.wav")
        self.assertEqual(scores["sample_count"], 3)
        self.assertAlmostEqual(scores["genuine"], 1.)
        self.assertAlmostEqual(scores["score"], 0.)

    def test_text_updates_extend_enrollment_and_count_only_new_training_utterances(self):
        for n in range(6):
            self.ingest({0: [(n + 1, self.a)]})
        first_inputs = self.store.text_inputs("1")
        with patch("voice_lab.beta_fit.fit_target", return_value=(None, {"passes_quality": True})):
            self.store.fit("1")
        self.store._signatures.clear()
        with patch.object(en, "MAX_CLIPS", 6), patch.object(en, "SIGNATURE_CACHE_SIZE", 2):
            self.ingest({0: [(16, self.a)]})
            self.assertEqual(self.store.snapshot()["speakers"][0]["clip_count"], 6)
            self.store.ensure_text_watch("1", first_inputs["training_clips"] + first_inputs["heldout_clips"])
            for n in range(8):
                self.ingest({0: [(20+n, self.a)]})
            state = self.store.snapshot()["speakers"][0]
            self.assertTrue(state["auto_update"])
            self.assertEqual(state["new_training_clips"], 6)
            self.assertLessEqual(len(self.store._signatures), 2)
            self.store.mark_text_attempt("1", state["training_count"])
            self.store.set_text_updates("1", False)
            self.ingest({0: [(40, self.a)]})
            state = self.store.snapshot()["speakers"][0]
            self.assertEqual(state["clip_count"], 14)
            self.assertEqual(state["new_training_clips"], 0)
            reopened = en.Enrollment(self.path / "session", self.path / "models")
            reopened.ensure_text_watch("1")
            self.assertFalse(reopened.snapshot()["speakers"][0]["auto_update"])
            reopened.set_text_updates("1", True)
            self.assertEqual(reopened.text_watches(), [("1", state["training_count"], state["training_count"])])

    def test_holdout_improvement_must_be_consistent(self):
        self.assertFalse(_promote([.8, .8], [.9, .75]))
        self.assertFalse(_promote([.8, .8], [.801, .803]))
        self.assertTrue(_promote([.8, .8], [.81, .82]))
        self.assertFalse(_passes_quality([.48, .60], [.79, .81])[0])
        self.assertFalse(_passes_quality([.95, .60], [.89, .91])[0])
        self.assertTrue(_passes_quality([.80, .81], [.89, .91])[0])
