import os
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from voice_lab import beta_audio


class BetaAudioTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def await_state(self, child, state):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            last = child.status()
            if last["state"] == state:
                return last
            time.sleep(0.01)
        self.fail(f"Expected {state}, got {last}")

    def script(self, body):
        path = self.root / "helper"
        path.write_text("#!" + sys.executable + "\n" + textwrap.dedent(body), encoding="utf-8")
        path.chmod(0o700)
        return path

    def test_capture_graceful_stop_and_restart(self):
        command = self.script("""
            import os, sys
            from pathlib import Path
            directory = Path(sys.argv[2])
            directory.mkdir()
            print("Recording Discord audio.", flush=True)
            while os.read(0, 1): pass
            (directory / "chunk-000001.wav").write_bytes(b"finalized")
        """)
        capture = beta_audio.CaptureProcess(command)
        self.addCleanup(capture.stop)
        first = self.root / "one"
        capture.start(first)
        self.await_state(capture, "running")
        with self.assertRaisesRegex(ValueError, "already running"):
            capture.start(self.root / "two")
        self.assertEqual(capture.stop()["state"], "stopped")
        self.assertEqual((first / "chunk-000001.wav").read_bytes(), b"finalized")
        with self.assertRaisesRegex(ValueError, "new directory"):
            capture.start(first)
        capture.start(self.root / "two")
        self.await_state(capture, "running")
        self.assertEqual(capture.stop(), capture.stop())

    def test_capture_reports_native_permission_diagnostic(self):
        command = self.script("""
            import sys
            print("capture-discord: Grant the requesting app Screen & System Audio Recording access.", file=sys.stderr)
            sys.exit(1)
        """)
        capture = beta_audio.CaptureProcess(command)
        self.addCleanup(capture.stop)
        capture.start(self.root / "one")
        status = self.await_state(capture, "error")
        self.assertIn("Screen & System Audio", status["message"])
        self.assertEqual(status, capture.stop())

    def test_named_cable_devices_and_switching(self):
        names = [
            {"name": "Built-in Microphone", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Mac Speakers", "max_input_channels": 0, "max_output_channels": 2},
            {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2},
        ]
        sd = Mock()
        sd.query_devices.return_value = names
        route = beta_audio.RouteProcess("mps")
        target = self.root / "reference.wav"
        target.touch()
        adapter = self.root / "adapter.npz"
        adapter.touch()
        with patch.object(beta_audio, "_sounddevice", return_value=sd):
            self.assertEqual([row["virtual"] for row in beta_audio.audio_devices()], [False, False, True])
            with self.assertRaisesRegex(ValueError, "virtual cable"):
                route.start(1, target, target, 0, 1)
            with self.assertRaisesRegex(ValueError, "distinct"):
                route.start(1, target, target, 2, 2)
            with patch.object(route, "_launch", return_value={}) as launch:
                route.start(1, target, target, 0, 2, check=True, target_adapter=adapter)
                args = launch.call_args.args[0]
                self.assertEqual(args[args.index("--target-adapter") + 1], str(adapter.resolve()))
                self.assertIn("--check", args)
                self.assertEqual(launch.call_args.kwargs["output_name"], "BlackHole 2ch")
        with patch.object(beta_audio, "_sounddevice", return_value=sd):
            for label in ("one", "two"):
                # Stand in for the converter only. Exercise actual child switching and its lifetime pipe.
                native_launch = beta_audio._AudioChild._launch
                def launch(command, description, ready, **details):
                    snippet = ("import os;print('Live timing: 160 ms blocks · 443 ms estimated software delay');"
                               "print('Monitor active.',flush=True);os.read(0,1)")
                    return native_launch(route, [sys.executable, "-uc", snippet], description, ready, **details)
                with patch.object(route, "_launch", side_effect=launch):
                    route.start(label, target, target, 0, 2)
                self.addCleanup(route.stop)
                status = self.await_state(route, "running")
                self.assertEqual(status["speaker_id"], label)
                self.assertEqual(status["timing"], "160 ms blocks · 443 ms estimated software delay")
                if label == "one":
                    previous = route._process
            self.assertIsNotNone(previous.poll())
            self.assertEqual(route.stop()["state"], "stopped")

    def test_switch_back_to_unaltered_mic_on_same_cable(self):
        # Both device selection in the parent and actual RawStream in the child use fixtures.
        device_module = self.root / "sounddevice.py"
        device_module.write_text(textwrap.dedent("""
            from pathlib import Path
            root = Path(__file__).parent
            class PortAudioError(Exception): pass
            def query_devices(index, kind):
                return dict(name=("Fixture mic", "BlackHole fixture")[index], hostapi=0)
            def check_input_settings(device, **settings): assert device == 0
            def check_output_settings(device, **settings): assert device == 1
            class RawStream:
                def __init__(self, device, callback, channels, dtype, samplerate, **kwargs):
                    assert device == (0, 1) and (channels, dtype, samplerate) == (1, "int16", 48000)
                    self.active = True
                    self.callback = callback
                def __enter__(self):
                    original = bytes(range(64))
                    played = bytearray(len(original))
                    self.callback(original, played, len(original) // 2, None, False)
                    (root / "cable.pcm").write_bytes(played)
                    return self
                def __exit__(self, *args):
                    self.active = False
                    (root / "cable.closed").touch()
        """), encoding="utf-8")
        names = [dict(id=0, inputs=1, outputs=0, name="Fixture mic", virtual=False),
                 dict(id=1, inputs=2, outputs=2, name="BlackHole fixture", virtual=True)]
        route = beta_audio.RouteProcess()
        self.addCleanup(route.stop)
        source = Path(__file__).resolve().parents[1] / "src"
        with patch.dict(os.environ, {"PYTHONPATH": os.pathsep.join((str(self.root), str(source)))}), \
             patch.object(beta_audio, "audio_devices", return_value=names):
            with route._control:
                route._launch([sys.executable, "-uc", "import os; print('Monitor active.'); os.read(0,1)"],
                              "voice route", "Monitor active.", speaker_id="2", mode="voice", mic=0, output=1)
            self.await_state(route, "running")
            previous = route._process
            with self.assertRaisesRegex(ValueError, "distinct"):
                route.use_microphone(0, 0)
            self.assertIsNone(previous.poll())
            route.use_microphone(0, 1)
            status = self.await_state(route, "running")
            self.assertIsNotNone(previous.poll())
            self.assertEqual((status["mode"], status["speaker_id"], status["mic"], status["output"]),
                             ("microphone", None, 0, 1))
            self.assertEqual(status["output_name"], "BlackHole fixture")
            self.assertEqual((self.root / "cable.pcm").read_bytes(), bytes(range(64)))
            self.assertEqual(route.stop()["state"], "stopped")
            self.assertTrue((self.root / "cable.closed").exists())

    @unittest.skipUnless(sys.platform == "darwin" and shutil.which("xcrun"), "requires the macOS Swift toolchain")
    def test_native_rotation_preserves_samples_boundaries_gaps_and_tail(self):
        writer = Path(__file__).resolve().parents[1] / "desktop/WAVWriter.swift"
        main = self.root / "main.swift"
        main.write_text(textwrap.dedent("""
            import Foundation
            let dest = URL(fileURLWithPath: CommandLine.arguments[1])
            let writer = try ChunkedWAVWriter(directory: dest, framesPerChunk: 4)
            try [Float(0.25), 0.5, -1].withUnsafeBufferPointer { try writer.append($0, at: 0) }
            precondition(!FileManager.default.fileExists(atPath: dest.appendingPathComponent("chunk-000001.wav").path))
            try [Float(0.75), -0.5, 1].withUnsafeBufferPointer { try writer.append($0, at: 6) }
            precondition(FileManager.default.fileExists(atPath: dest.appendingPathComponent("chunk-000002.wav").path))
            try [Float(1), 1, -0.75].withUnsafeBufferPointer { try writer.append($0, at: 7) }
            try writer.publish()
            try writer.publish()
            do {
                _ = try ChunkedWAVWriter(directory: dest, framesPerChunk: 4)
                fatalError("opened existing directory")
            } catch {}
        """), encoding="utf-8")
        binary = self.root / "writer-test"
        subprocess.run(["xcrun", "swiftc", "-module-cache-path", str(self.root / "module-cache"),
                        str(writer), str(main), "-o", str(binary)], capture_output=True, check=True)
        directory = self.root / "chunks"
        subprocess.run([str(binary), str(directory)], check=True)
        files = sorted(directory.iterdir())
        self.assertEqual([f.name for f in files], [f"chunk-{n:06d}.wav" for n in range(1, 4)])
        actual = []
        for file in files:
            with wave.open(str(file)) as wav:
                self.assertEqual(wav.getparams()[:3], (1, 2, 48000))
                actual.append(list(struct.unpack(f"<{wav.getnframes()}h", wav.readframes(wav.getnframes()))))
        self.assertEqual(actual, [[8192, 16384, -32767, 0], [0, 0, 24575, -16384], [32767, -24575]])
