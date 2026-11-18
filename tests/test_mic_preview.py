import math
import os
import shutil
import stat
import struct
import tempfile
import unittest
import wave
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

from voice_lab import mic_preview


def fake_input(raw, status="", chunk=739):
    sd = Mock()
    sd.PortAudioError = OSError
    sd.CallbackStop = type("Stop", (Exception,), {})
    sd.CallbackAbort = type("Abort", (Exception,), {})
    sd.query_devices.return_value = {"name": "Fixture mic"}

    @contextmanager
    def stream(**settings):
        try:
            for i in range(0, len(raw), chunk * 2):
                data = raw[i:i + chunk * 2]
                settings["callback"](data, len(data) // 2, None, status)
        except (sd.CallbackStop, sd.CallbackAbort):
            pass
        settings["finished_callback"]()
        yield

    sd.RawInputStream.side_effect = stream
    return sd


class MicCaptureTests(unittest.TestCase):
    def test_captures_exact_frames_on_requested_input(self):
        frames = 48000
        # Native PortAudio bytes, with an extra block beyond the requested cut.
        samples = [int(8000 * math.sin(2 * math.pi * i / 240)) for i in range(frames + 500)]
        sd = fake_input(struct.pack(f"={len(samples)}h", *samples))
        with tempfile.TemporaryDirectory() as tmp, patch.object(mic_preview, "_sounddevice", return_value=sd):
            out = Path(tmp, "dry.wav")
            mic_preview.record_mic(3, 1, out)
            sd.query_devices.assert_called_once_with(3, "input")
            sd.RawStream.assert_not_called()
            sd.RawOutputStream.assert_not_called()
            sd.RawInputStream.assert_called_once()
            self.assertEqual(sd.RawInputStream.call_args.kwargs["device"], 3)
            self.assertEqual(stat.S_IMODE(os.stat(out).st_mode), 0o600)
            with wave.open(str(out)) as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (48000, 1, 2))
                self.assertEqual(wav.getnframes(), frames)
                self.assertEqual(wav.readframes(frames), struct.pack(f"<{frames}h", *samples[:frames]))

    def test_partial_drop_and_silence_are_discarded(self):
        for raw, status, reason in [(bytes(1200), "input overflow", "dropped audio"),
                                    (bytes(1200), "", "before the take was complete"),
                                    (struct.pack("=h", 1) * 48000, "", "Too little microphone signal")]:
            sd = fake_input(raw, status=status)
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp, \
                    patch.object(mic_preview, "_sounddevice", return_value=sd):
                with self.assertRaisesRegex((RuntimeError, ValueError), reason):
                    mic_preview.record_mic(3, 1, Path(tmp, "dry.wav"))
                self.assertEqual(list(Path(tmp).iterdir()), [])

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_comparison_matches_level_and_keeps_dry_then_converted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, amplitude, sign in [("dry", 20000, 1), ("converted", 1000, -1)]:
                with wave.open(str(root / f"{name}.wav"), "wb") as wav:
                    wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                    wav.writeframes(struct.pack("<h", amplitude * sign) * 24000)
            out = root / "compare.wav"
            mic_preview.compare_takes(root / "dry.wav", root / "converted.wav", out)
            with wave.open(str(out)) as wav:
                self.assertEqual(wav.getnframes(), 66000)
                samples = struct.unpack("<66000h", wav.readframes(66000))
            self.assertEqual(samples[:24000], (3277,) * 24000)
            self.assertEqual(samples[24000:42000], (0,) * 18000)
            self.assertEqual(samples[42000:], (-3277,) * 24000)
