import argparse
import io
import shutil
import stat
import struct
import subprocess
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from voice_lab import local_preview


def request(server, body, *, host="127.0.0.1:1234", origin="http://127.0.0.1:1234",
            mime="audio/wav", length=None):
    handler = local_preview.PreviewHandler.__new__(local_preview.PreviewHandler)
    content = (f"POST /take HTTP/1.1\r\nHost: {host}\r\nOrigin: {origin}\r\nContent-Type: {mime}\r\n"
               f"Content-Length: {len(body) if length is None else length}\r\n\r\n").encode()
    handler.rfile, handler.wfile = io.BytesIO(content + body), io.BytesIO()
    handler.server, handler.client_address = server, ("127.0.0.1", 50203)
    handler.log_message = Mock()
    handler.handle_one_request()
    head, payload = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
    return int(head.split()[1]), head.decode(), payload


def wav_bytes(seconds, magnitude=3000):
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(struct.pack("<hh", magnitude, -magnitude) * round(8000 * seconds))
    return out.getvalue()


class LocalPreviewTests(unittest.TestCase):
    def setUp(self):
        self.sandbox = tempfile.TemporaryDirectory()
        self.addCleanup(self.sandbox.cleanup)
        self.root = Path(self.sandbox.name)
        self.server = argparse.Namespace(host_port="127.0.0.1:1234", converting=threading.Lock(),
                                         settings=argparse.Namespace(output_dir=self.root, target=self.root / "voice.wav",
                                                                     weights=None, threads=4, device="mps",
                                                                     profile="fast", device_rate=48000,
                                                                     mic_peak_dbfs=-3, max_mic_gain_db=24))

    def test_defaults_to_cpu_and_accepts_mps(self):
        parser = argparse.ArgumentParser()
        local_preview.configure(parser.add_subparsers())
        args = ["vc-studio", "--target", "voice.wav", "--output-dir", "takes"]
        self.assertEqual(parser.parse_args(args).device, "cpu")
        self.assertEqual(parser.parse_args(args + ["--device", "mps"]).device, "mps")

    def test_rejects_other_origins_hosts_types_and_bodies_before_conversion(self):
        with patch.object(local_preview, "make_take") as conversion:
            for options, code in [({"host": "other:1234"}, 403), ({"origin": "http://other:1234"}, 403),
                                  ({"origin": "null"}, 403), ({"mime": "text/plain"}, 415),
                                  ({"length": "16777217"}, 413), ({"length": "-1"}, 411)]:
                with self.subTest(options=options):
                    self.assertEqual(request(self.server, b"test", **options)[0], code)
            self.server.converting.acquire()
            self.assertEqual(request(self.server, b"test")[0], 409)
            conversion.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_bad_takes_leave_no_files(self):
        for body in (b"broken", wav_bytes(2, 0), wav_bytes(30.1)):
            with self.subTest(size=len(body)):
                code, _, _ = request(self.server, body)
                self.assertEqual(code, 422)
                self.assertEqual(list(self.root.iterdir()), [])

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_decodes_uploaded_m4a_with_a_trailing_index(self):
        uploaded = self.root / "phone.m4a"
        subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi",
                        "-i", "sine=duration=11", "-c:a", "aac", "-b:a", "160k", str(uploaded)], check=True)
        data = uploaded.read_bytes()
        self.assertGreater(data.find(b"moov"), 65536)
        dry = self.root / "dry.wav"
        local_preview.decode_take(data, dry)
        with wave.open(str(dry)) as wav:
            self.assertAlmostEqual(wav.getnframes() / wav.getframerate(), 11, delta=.05)

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_good_takes_run_replay_and_return_comparison(self):
        def simulated_replay(args):
            self.assertEqual(args.target, self.root / "voice.wav")
            self.assertEqual(args.device, "mps")
            self.assertIsNone(args.source_calibration)
            self.assertEqual(args.device_rate, 48000)
            self.assertEqual((args.profile, args.mic_peak_dbfs, args.max_mic_gain_db), ("fast", -3, 24))
            with wave.open(str(args.output), "wb") as out:
                out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                out.writeframes(struct.pack("<h", 1000) * 32000)

        with patch.object(local_preview.vc_stream, "replay", side_effect=simulated_replay) as replay:
            status, headers, body = request(self.server, wav_bytes(2))
            replay.assert_called_once()
        self.assertEqual(status, 200)
        dirname = next(line.split(": ")[1] for line in headers.splitlines() if line.startswith("X-Preview-Take:"))
        saved = self.root / dirname
        self.assertTrue(dirname.startswith("take-"))
        self.assertEqual(stat.S_IMODE(saved.stat().st_mode), 0o700)
        self.assertEqual({p.name for p in saved.iterdir()}, {"dry.wav", "converted.wav", "side-by-side.wav"})
        self.assertEqual((saved / "side-by-side.wav").read_bytes(), body)
        with wave.open(io.BytesIO(body)) as check:
            self.assertEqual(check.getnframes(), 114000)
