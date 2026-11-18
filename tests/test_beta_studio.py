import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import Mock, patch

from voice_lab import beta, beta_driver, beta_server, vc


class StudioTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        args = argparse.Namespace(data_dir=root, model_dir=root / "models", capture_tool=root / "helper",
                                  local_files_only=True, device="cpu")
        with patch.object(beta.Studio, "prepare_models"):
            self.studio = beta.Studio(args)
        self.addCleanup(self.studio.close)

    def test_model_setup_child_errors_can_be_retried(self):
        popen = subprocess.Popen
        scripts = iter([
            """print('studio-setup {"error":"Voice model unavailable"}'); raise SystemExit(1)""",
            """print('studio-setup {"device":"cpu"}')""",
        ])
        def helper(args, **kwargs):
            self.assertEqual(args[1:3], ["-m", "voice_lab.beta_setup"])
            self.assertEqual(args[-1], "--local-files-only")
            return popen([sys.executable, "-c", next(scripts)], **kwargs)

        with patch.object(beta.subprocess, "Popen", side_effect=helper):
            self.studio.prepare_models()
            self.studio.setup_thread.join(5)
            self.assertEqual(self.studio.snapshot()["setup"], {"state": "error", "message": "Voice model unavailable"})
            self.studio.retry()
            self.studio.setup_thread.join(5)
            self.assertEqual(self.studio.snapshot()["setup"]["state"], "ready")
            self.assertIsNone(self.studio.setup_process)

    def test_quitting_stops_an_inflight_model_setup(self):
        popen = subprocess.Popen
        started = threading.Event()
        child = []
        def hanging_helper(*args, **kwargs):
            proc = popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
            child.append(proc)
            started.set()
            return proc

        with patch.object(beta.subprocess, "Popen", side_effect=hanging_helper):
            self.studio.prepare_models()
            self.assertTrue(started.wait(5))
            self.studio.close()
            self.studio.setup_thread.join(5)
        self.assertIsNotNone(child[0].poll())
        self.assertFalse(self.studio.setup_thread.is_alive())

    def test_finished_chunk_is_analyzed_and_session_can_be_reopened(self):
        studio = self.studio
        studio.select(None)
        call_id = studio.call_id
        studio.capture = Mock()
        studio.capture.status.return_value = {"state": "idle"}
        studio.attach()
        run = studio.capture.start.call_args.args[0]
        run.mkdir()
        chunk = run / "chunk-000001.wav"
        chunk.write_bytes(b"fixture")
        pending = {"id": "1", "name": "Speaker 1", "needs_fit": True, "model_status": "ready_to_fit"}
        ready = dict(pending, needs_fit=False, model_status="optimized")
        fitted = threading.Event()
        fake = Mock()
        fake.text_watches.return_value = []
        fake.snapshot.side_effect = [{"speakers": [pending]}, {"speakers": [ready]}]
        fake.fit.side_effect = lambda *a, **kw: fitted.set()
        with studio.lock:
            studio.enroll = fake
            studio.setup["state"] = "ready"
        studio.wake.set()
        self.assertTrue(fitted.wait(5))
        # Join the worker before asserting the committed state or changing libraries.
        studio.closing.set()
        studio.wake.set()
        studio.worker.join(4)
        fake.ingest.assert_called_once_with(chunk)
        fake.fit.assert_called_once_with("1", device="cpu", threads=2)
        self.assertEqual(studio.snapshot()["voices"]["speakers"][0]["model_status"], "optimized")
        self.assertEqual(studio.snapshot()["backlog"], 0)
        studio.select(None)
        self.assertNotEqual(call_id, studio.call_id)
        studio.select(call_id)
        self.assertEqual(studio.snapshot()["call"], call_id)

    def test_saved_call_scheduler_defers_to_live_and_skips_unusable_speakers(self):
        studio = self.studio
        studio.select(None)
        selected = studio.call_id
        queued = "20260101T000000Z-1234abcd"
        (studio.calls / queued / "text-voices").mkdir(parents=True)
        waiting = {"available": "preview-a", "pending_update": {"retry_at": 200, "failures": 1}}
        saved = Mock()
        saved.text_watches.return_value = [("1", 10, 10), ("2", 18, 12), ("3", 20, 14)]
        def inputs(sid):
            if sid == "2":
                raise FileNotFoundError("missing training audio")
            return {"training_clips": ["third"], "heldout_clips": ["control"]}
        saved.text_inputs.side_effect = inputs

        with studio.lock, patch.object(studio, "enroll_factory", return_value=saved), patch.object(
                studio.text, "build") as build, patch.object(studio.text, "snapshot") as snapshot, patch.object(
                studio.route, "status", return_value={"state": "running"}) as route, patch.object(
                studio.capture, "status", return_value={"state": "running"}), patch.object(beta.time, "time", return_value=100):
            snapshot.side_effect = lambda call, speakers: {s["id"]: waiting if s["id"] == "1" else
                                                            {"available": "preview-a"} for s in speakers}
            studio.setup["state"] = "ready"
            studio.text_checks = {queued: None}
            studio._schedule_text()
            build.assert_not_called()
            self.assertIn(queued, studio.text_checks)
            route.return_value = {"state": "idle"}
            studio.previewing = True
            studio._schedule_text()
            build.assert_not_called()
            studio.previewing = False
            studio._schedule_text()
            build.assert_called_once_with(studio.calls / queued, "3", inputs("3"), "full", automatic=True)
            saved.mark_text_attempt.assert_called_once_with("3", 20)
            self.assertIn("Speaker 2", studio.text_errors[queued])
            self.assertEqual(studio.call_id, selected)
            # A failed speaker cannot busy-loop. Only the pending due time remains.
            saved.text_watches.return_value = [("1", 10, 10), ("3", 20, 20)]
            studio._schedule_text()
            self.assertEqual(studio.text_checks, {queued: 200})
            before = saved.text_watches.call_count
            studio._schedule_text()
            self.assertEqual(saved.text_watches.call_count, before)
            studio.setup["state"] = "loading"

    def test_migration_skips_invalid_state_and_finds_existing_intent(self):
        studio = self.studio
        studio.select(None)
        directory = studio.directory / "text-voices"
        for sid, data in (("1", "broken JSON"), ("2", '{"available":"preview-two","training_count":8}')):
            voice = directory / sid
            voice.mkdir(parents=True)
            (voice / "state.json").write_text(data)
        enroll = Mock()
        studio.text_migrated.clear()
        studio._migrate_text(studio.call_id, enroll)
        enroll.ensure_text_watch.assert_called_once_with("2", [])
        enroll.mark_text_attempt.assert_called_once_with("2", 8)
        self.assertIn("Speaker 1", studio.text_errors[studio.call_id])

    def test_requires_valid_voice_and_sample_to_route(self):
        studio = self.studio
        studio.route = Mock()
        studio.enroll = Mock()
        studio.voices = {"speakers": [{"id": "1", "model_status": "needs_more_audio"}]}
        studio.calibration.write_bytes(b"mic")
        with self.assertRaises(ValueError):
            studio.route_voice("1", 1, 2)
        studio.voices["speakers"][0]["model_status"] = "optimized"
        studio.enroll.selected_files.return_value = (Path("reference.wav"), Path("adapter.npy"))
        studio.route_voice("1", 1, 2)
        studio.route.start.assert_called_once_with("1", Path("reference.wav"), studio.calibration,
                                                   1, 2, check=False, target_adapter=Path("adapter.npy"))

    def test_controls_require_local_origin_and_current_call(self):
        studio = self.studio
        handler = beta_server.Handler.__new__(beta_server.Handler)
        handler.server = argparse.Namespace(host_port="127.0.0.1:12345", studio=studio)
        def post(origin, body):
            from email.message import Message
            payload = json.dumps(body).encode()
            handler.headers = Message()
            for name, value in (("Host",handler.server.host_port),("Origin",origin),
                                ("Content-Type","application/json"),("Content-Length",str(len(payload)))):
                handler.headers[name] = value
            handler.rfile = BytesIO(payload)
            handler.reply = Mock()
            handler.path = "/api/new"
            handler.do_POST()
            return handler.reply.call_args.args[0]
        self.assertEqual(post("https://outside.invalid", {"call": None}), 403)
        self.assertEqual(post("http://" + handler.server.host_port, {"call": "stale"}), 400)
        self.assertIsNone(studio.call_id)
        self.assertEqual(post("http://" + handler.server.host_port, {"call": None}), 200)
        self.assertTrue(beta.CALL_ID.fullmatch(studio.call_id))

    def test_bad_driver_download_never_opens_installer(self):
        with patch.object(beta_driver, "urlopen", return_value=BytesIO(b"corrupt")), patch.object(
                beta_driver.subprocess, "run") as run, patch.object(beta_driver.sys, "platform", "darwin"):
            with self.assertRaisesRegex(RuntimeError, "published"):
                beta_driver.open_installer(self.studio.root)
        run.assert_not_called()


class AdapterTest(unittest.TestCase):
    def test_checked_data_only_adapter(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional NumPy dependency")
        with TemporaryDirectory() as name:
            path = Path(name) / "adapter.npy"
            shape = (1, 256, 1)
            correct = np.ones(shape, dtype=np.float32)
            np.save(path, correct)
            np.testing.assert_array_equal(vc.read_target_adapter(path), correct)
            for bad in [np.zeros(shape, dtype=np.float32), np.ones(3), np.full(shape, np.nan, dtype=np.float32),
                        np.array([object()], dtype=object)]:
                np.save(path, bad)
                with self.assertRaises(ValueError):
                    vc.read_target_adapter(path)
