import argparse
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from voice_lab import live


def run(*args):
    parser = argparse.ArgumentParser()
    live.configure(parser.add_subparsers(required=True))
    parsed = parser.parse_args(["live", *map(str, args)])
    return parsed.handler(parsed)


class LiveTests(unittest.TestCase):
    def test_file_preview_matches_live_effect_across_block_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, "mic.wav")
            output = Path(tmp, "preview.wav")
            data = b"".join(struct.pack("<h", round(16000 * math.sin(i * 0.1))) for i in range(5301))
            with wave.open(str(source), "wb") as writer:
                writer.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                writer.writeframes(data)
            with patch.object(live, "_sounddevice", side_effect=AssertionError):
                run("preview", source, output, "--effect", "radio")
                with self.assertRaises(FileExistsError):
                    run("preview", source, output)
            with wave.open(str(output), "rb") as reader:
                result = reader.readframes(reader.getnframes())
                self.assertEqual(reader.getframerate(), 48000)
            self.assertEqual(len(result), len(data))
            self.assertNotEqual(result, data)
            self.assertEqual(result, live.MicEffect("radio", 48000).process(data, little_endian=True))

    def test_check_probes_only_requested_devices_without_opening_stream(self):
        sd = Mock()
        sd.PortAudioError = OSError
        sd.query_devices.side_effect = [{"name": "Mic", "hostapi": 2}, {"name": "Cable", "hostapi": 2}]
        with patch.object(live, "_sounddevice", return_value=sd):
            run("route", "--mic", "0", "--virtual-output", "7", "--check")
        sd.check_input_settings.assert_called_once_with(device=0, channels=1, dtype="int16", samplerate=48000)
        sd.check_output_settings.assert_called_once_with(device=7, channels=1, dtype="int16", samplerate=48000)
        sd.RawStream.assert_not_called()

    def test_route_passes_each_frame_to_only_selected_output(self):
        sd = Mock()
        sd.PortAudioError = OSError
        sd.query_devices.side_effect = [{"name": "Mic", "hostapi": 2}, {"name": "Cable", "hostapi": 2}]
        stream = Mock(active=True)
        stream.__enter__ = Mock(return_value=stream)
        stream.__exit__ = Mock(return_value=False)
        sd.RawStream.return_value = stream
        original = b"\x01\x00\xff\xff" * 20
        out = bytearray(len(original))

        def simulate_callback_and_stop(_):
            callback = sd.RawStream.call_args.kwargs["callback"]
            callback(original, out, len(original) // 2, None, False)
            raise KeyboardInterrupt

        with patch.object(live, "_sounddevice", return_value=sd), patch.object(
            live.time, "sleep", side_effect=simulate_callback_and_stop
        ):
            run("route", "--mic", "0", "--virtual-output", "7")
        self.assertEqual(sd.RawStream.call_args.kwargs["device"], (0, 7))
        self.assertEqual(out, original)
        stream.__exit__.assert_called_once()

    def test_rejects_same_or_implicit_device_ids(self):
        with patch.object(live, "_sounddevice", return_value=Mock()) as backend:
            for mic, cable in [(1, 1), (1, -1), (-1, 1)]:
                with self.assertRaisesRegex(ValueError, "distinct device IDs"):
                    run("route", "--mic", mic, "--virtual-output", cable)
            backend.return_value.RawStream.assert_not_called()

    def test_giant_lowers_pitch_and_handles_partial_final_block(self):
        try:
            engine = live.processor("giant", 48000)
        except RuntimeError as exc:
            self.skipTest(str(exc))
        with engine as effect:
            block = effect.block_frames
            delay = effect.delay_frames
        self.assertGreater(delay, 0)
        frames = block * 90 + 123
        rate = 48000
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp, "source.wav"), Path(tmp, "giant.wav")
            data = b"".join(struct.pack("<h", round(12000 * math.sin(2 * math.pi * 250 * i / rate)))
                            for i in range(frames))
            with wave.open(str(source), "wb") as writer:
                writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                writer.writeframes(data)
            run("preview", source, output, "--effect", "giant")
            with wave.open(str(output), "rb") as reader:
                self.assertEqual(reader.getnframes(), frames)
                signal = [sample for sample, in struct.iter_unpack("<h", reader.readframes(frames))]
            # Ignore both boundary transients when measuring fundamental pitch.
            middle = signal[frames // 4:3 * frames // 4]
            crossings = sum(a <= 0 < b for a, b in zip(middle, middle[1:]))
            self.assertAlmostEqual(crossings * rate / len(middle), 250 * 2 ** (-4 / 12), delta=3)
