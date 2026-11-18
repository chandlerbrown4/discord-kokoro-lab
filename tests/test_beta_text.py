import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from voice_lab.beta_text import TextVoices, _prefer_update
from voice_lab.distill_jobs import locate


class TextVoiceTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.call = Path(scratch.name) / "call"
        self.voices = TextVoices(Path(scratch.name))
        self.addCleanup(self.voices.suspend)
        self.inputs = {"training_clips": ["train"], "heldout_clips": ["held"]}
        self.runtime = Path(scratch.name) / "worker.py"
        self.runtime.write_text("""
import json, os, sys, time
from pathlib import Path
stage, directory = sys.argv[1], Path(sys.argv[2])
original = json.loads((directory / "inputs.json").read_text())["heldout_clips"][0]
if stage == "corpus":
    if (directory.parent / "wait").exists(): time.sleep(30)
    (directory / "manifest-path.json").write_text('"corpus"')
    (directory / "review-lines.json").write_text(json.dumps([dict(text="a", original=original)]))
    (directory / "corpus").write_text('{"audio":"pair.wav"}\\n' * int(sys.argv[sys.argv.index("--count")+1]))
    (directory / "pair.wav").touch()
elif stage == "student":
    (directory / "student").mkdir(exist_ok=True)
    (directory / "student/profile.pt").write_bytes(b"adapter")
    (directory / "review-0.wav").write_bytes(b"candidate")
elif stage == "evaluate":
    (directory / "review.wav").write_bytes(b"review")
    scorefile = directory.parent / "score.txt"
    score = float(scorefile.read_text()) if scorefile.exists() else .80
    (directory / "evaluation.json").write_text(json.dumps(dict(lines=[dict(text="a", original=original, similarity=score)],
        mean_cosine=score, mean_word_error_rate=0, passes_quality=True, genuine_cosine=.85)))
elif stage == "render":
    for line in sys.stdin:
        text = json.loads(line)["text"]
        (directory / "utterance.wav").write_bytes(text.encode())
        print("voice-render " + json.dumps(dict(seconds=.1, text=text)), flush=True)
""")
        def command(kind, stage, directory, model_dir):
            return [sys.executable, str(self.runtime), stage, str(directory)]
        mock = patch("voice_lab.beta_text._command", side_effect=command)
        self.addCleanup(mock.stop)
        mock.start()

    def wait_for(self, condition):
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(.01)
        self.fail("Text job did not reach the expected state.")

    def completed(self):
        self.wait_for(lambda: self.voices.job is None)

    def state(self, sid):
        return self.voices.snapshot(self.call, [{"id": sid}])[sid]

    def test_render_reuses_worker_and_loaded_library_can_render_again(self):
        self.voices.build(self.call, "7", self.inputs, "preview")
        self.completed()
        self.assertTrue(self.state("7")["passes_quality"])
        self.assertEqual(self.voices.audio(self.call, "7", "profile").read_bytes(), b"adapter")
        with self.assertRaises(ValueError):
            self.voices.audio(self.call, "2", "tts")
        self.voices.render(self.call, "7", "One")
        self.completed()
        first = self.voices.warm
        self.voices.render(self.call, "7", "Two")
        self.completed()
        self.assertIs(first, self.voices.warm)
        self.assertEqual(self.voices.audio(self.call, "7", "tts").read_bytes(), b"Two")
        self.voices.suspend()
        self.assertIsNotNone(first.poll())
        reopened = TextVoices(self.voices.model_dir)
        self.assertEqual(reopened.snapshot(self.call, [{"id": "7"}])["7"]["rendered"], "Two")
        self.assertNotIn("7", reopened.snapshot(self.call.parent / "another-call", [{"id": "7"}]))

    def test_removed_corpus_can_be_regenerated_and_a_complete_voice_survives_a_move(self):
        self.voices.build(self.call, "4", self.inputs, "preview")
        self.completed()
        folder = self.call / "text-voices/4" / self.state("4")["available"]
        for name in ("pair.wav", "student/profile.pt", "review-0.wav", "evaluation.json"):
            (folder / name).unlink()
        self.voices.build(self.call, "4", self.inputs, "preview")
        self.completed()
        self.assertTrue((folder / "pair.wav").is_file())
        self.assertTrue(self.state("4")["passes_quality"])

        moved = self.call.parent / "relocated"
        self.call.rename(moved)
        self.assertEqual(self.voices.audio(moved, "4", "profile").read_bytes(), b"adapter")
        old = Path("/old-library/calls/test-call/voices/sample.wav")
        call = moved / "calls/test-call"
        new_file = call / "voices/sample.wav"
        new_file.parent.mkdir(parents=True)
        new_file.touch()
        self.assertEqual(locate(call / "text-voices/9/preview-abc", str(old)), new_file)

    def test_auto_updates_freeze_genuine_controls_and_preserve_both_voices(self):
        self.voices.build(self.call, "7", self.inputs, "preview")
        self.completed()
        root = self.call / "text-voices/7"
        first = self.state("7")["available"]
        (root / "score.txt").write_text(".790")
        inputs = {"training_clips": ["train", "added"], "heldout_clips": ["held", "added-control"]}
        self.voices.build(self.call, "7", inputs, "full", automatic=True)
        self.completed()
        state = self.state("7")
        second = state["alternate"]
        self.assertEqual(state["last_decision"], "kept")
        self.assertEqual(state["available"], first)
        self.assertEqual(self.voices.audio(self.call, "7", "alternate-review"), root / second / "review.wav")
        controls = json.loads((root / second / "inputs.json").read_text())["heldout_clips"]
        self.assertEqual([locate(root / second, v) for v in controls], [(self.call / "held").resolve()])
        (root / "score.txt").write_text(".830")
        inputs["training_clips"].append("later")
        self.voices.build(self.call, "7", inputs, "full", automatic=True)
        self.completed()
        state = self.state("7")
        self.assertEqual(state["last_decision"], "promoted")
        self.assertNotEqual(state["available"], first)
        self.assertEqual(state["alternate"], first)
        self.assertAlmostEqual(state["similarity"], .830)

    def test_paired_promotion_requires_compatible_delivery_intelligibility_and_evidence(self):
        row = dict(text="A line", original="a.wav", original_sha256="a", similarity=.8,
                   word_error_rate=0, duration_log_error=.07, pitch_center_error_st=.5, pitch_span_error_st=.6)
        prior = dict(passes_quality=True, mean_cosine=.8, mean_word_error_rate=0, lines=[row]*3)
        updates = [({}, 0, .8, False), ({}, 0, .81, True),
                   ({"original_sha256": "different"}, 0, .90, False),
                   ({"duration_log_error": .31}, 0, .90, False),
                   ({"pitch_span_error_st": 4.5}, 0, .90, False),
                   ({"pitch_span_error_st": None}, 0, .90, False),
                   ({"duration_log_error": None}, 0, .90, False),
                   ({"word_error_rate": .3}, .10, .90, False),
                   ({"similarity": .74}, 0, .81, False)]
        for changed, wer, score, expected in updates:
            with self.subTest(changed=changed, score=score):
                rows = [dict(row, **changed), row, row]
                report = dict(prior, mean_cosine=score, mean_word_error_rate=wer, lines=rows)
                self.assertEqual(_prefer_update(prior, self.call, report, self.call)[0], expected)

    def test_failed_voice_recovery_requires_consistent_measured_improvement(self):
        # Rounded measurements from three separate same-line stock-voice comparisons.
        before = [(.498, .061, 10.074, 1.282), (.486, .091, 11.064, 2.108), (.515, .103, 9.788, .476)]
        after = [(.889, .052, 1.881, 1.202), (.920, .099, 2.450, 3.011), (.927, .095, 1.364, 3.563)]
        def report(values, passes):
            rows = [dict(text=str(i), original_sha256=str(i), similarity=cos, word_error_rate=0.,
                         duration_log_error=dur, pitch_center_error_st=center, pitch_span_error_st=span)
                    for i, (cos, dur, center, span) in enumerate(values)]
            return dict(passes_quality=passes, mean_cosine=sum(r["similarity"] for r in rows) / len(rows),
                        mean_word_error_rate=0., lines=rows)
        prior, candidate = report(before, False), report(after, True)
        def promotes(old, new):
            return _prefer_update(old, self.call, new, self.call)[0]
        self.assertTrue(promotes(prior, candidate))
        self.assertFalse(promotes(dict(prior, passes_quality=True), candidate))
        self.assertFalse(promotes(prior, dict(candidate, passes_quality=False)))
        for failure in [dict(similarity=.60), dict(pitch_center_error_st=7.),
                        dict(duration_log_error=.14), dict(word_error_rate=.01),
                        dict(pitch_span_error_st=4.1), dict(pitch_center_error_st=None)]:
            with self.subTest(failure=failure):
                rows = [*candidate["lines"][:2], dict(candidate["lines"][2], **failure)]
                self.assertFalse(promotes(prior, dict(candidate, lines=rows)))

    def test_background_resume_after_foreground_render_preserves_partial_work(self):
        self.voices.build(self.call, "7", self.inputs, "preview")
        self.completed()
        root = self.call / "text-voices/7"
        (root / "wait").touch()
        self.voices.build(self.call, "7", self.inputs, "full", automatic=True)
        self.wait_for(lambda: self.voices.process is not None)
        planned = self.state("7")["pending_update"]
        self.assertEqual(self.voices.job_info()["task"], "build")
        self.voices.suspend(keep_update=True, automatic_only=True)
        self.voices.render(self.call, "7", "Foreground")
        self.completed()
        self.assertEqual(self.state("7")["pending_update"], planned)
        self.assertEqual(self.voices.audio(self.call, "7", "tts").read_bytes(), b"Foreground")
        self.voices.suspend(keep_update=True)
        (root / "wait").unlink()
        reopened = TextVoices(self.call)
        self.addCleanup(reopened.suspend)
        reopened.build(self.call, "7", {}, "resume", automatic=True)
        self.wait_for(lambda: reopened.job is None)
        state = reopened.snapshot(self.call, [{"id": "7"}])["7"]
        self.assertIsNone(state["pending_update"])
        self.assertEqual(state["last_decision"], "kept")

    def test_transient_failures_back_off_and_eventually_stop_retries(self):
        self.runtime.write_text("raise RuntimeError('temporary worker failure')")
        self.voices.build(self.call, "7", self.inputs, "full", automatic=True)
        self.completed()
        for failures in (1, 2):
            pending = self.state("7")["pending_update"]
            self.assertEqual(pending["failures"], failures)
            self.assertGreater(pending["retry_at"], time.time())
            self.voices.build(self.call, "7", {}, "resume", automatic=True)
            self.completed()
        self.assertIsNone(self.state("7")["pending_update"])
        self.assertEqual(self.state("7")["state"], "error")

    def test_resuming_full_build_keeps_its_original_speaker_evidence(self):
        root = self.call / "text-voices" / "3"
        root.mkdir(parents=True)
        flag = root / "wait"
        flag.touch()
        self.voices.build(self.call, "3", self.inputs, "full")
        self.wait_for(lambda: self.voices.process is not None)
        with self.assertRaises(ValueError):
            self.voices.build(self.call, "9", self.inputs, "preview")
        self.voices.suspend()
        self.completed()
        paused = self.state("3")
        fixed_inputs = json.loads((root / paused["build"] / "inputs.json").read_text())
        self.assertEqual(paused["state"], "paused")
        flag.unlink()
        self.voices.build(self.call, "3", {"training_clips": ["later"]}, "resume")
        self.completed()
        self.assertEqual(self.state("3")["available"], paused["build"])
        self.assertEqual(json.loads((root / paused["build"] / "inputs.json").read_text()), fixed_inputs)
