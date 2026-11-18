import argparse
import hashlib
import importlib.util
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from voice_lab import vc

try:
    import numpy as np
except ImportError:
    np = None


class VCFetchTest(unittest.TestCase):
    def test_local_model_resolution_pins_both_revisions_and_checks_every_digest(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            digests = {}
            for name in vc.DIGESTS:
                path = root / name
                path.write_bytes(name.encode())
                digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            hub = Mock()
            hub.hf_hub_download.side_effect = lambda repo, name, **kwargs: str(root / Path(name).name)
            with patch.dict("sys.modules", {"huggingface_hub": hub}), patch.dict(vc.DIGESTS, digests):
                paths = vc._assets(None, local_only=True)
                self.assertEqual(set(paths), set(digests))
                hub.hf_hub_download.assert_any_call("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth",
                                                    revision=vc.REVISION, local_files_only=True)
                hub.hf_hub_download.assert_any_call("M4869/WavMark", vc.MARK_FILE,
                                                    revision=vc.MARK_REVISION, local_files_only=True)
                paths["checkpoint.pth"].write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    vc._assets(root, local_only=True)


@unittest.skipIf(np is None, "optional numpy dependency")
class VCPreviewTest(unittest.TestCase):
    def test_calibration_raises_quiet_inputs_and_retains_headroom(self):
        self.assertAlmostEqual(vc.calibration_gain(np.array([0.05, -0.1])), 10 ** (-12 / 20) / 0.1)
        self.assertAlmostEqual(vc.calibration_gain(np.array([0.0002])), 10 ** (18 / 20))
        self.assertEqual(vc.calibration_gain(np.array([-0.7, 0.0, 0.2])), 1)
        self.assertAlmostEqual(vc.calibration_gain(np.array([0.05]), -3, 24), 10 ** (-3 / 20) / 0.05)
        self.assertAlmostEqual(vc.calibration_gain(np.array([0.001]), -3, 24), 10 ** (24 / 20))
        for peak, cap in ((float("nan"), 18), (1, 12), (-12, -1), (-3, float("inf"))):
            with self.assertRaisesRegex(ValueError, "gain cap"):
                vc.calibration_gain(np.array([0.05]), peak, cap)

    @unittest.skipUnless(importlib.util.find_spec("openvoice") and importlib.util.find_spec("torch"),
                         "requires the packaged OpenVoice runtime")
    def test_live_tail_matches_full_vocoder_with_the_same_latents(self):
        import torch
        from openvoice.models import Generator

        torch.manual_seed(7)
        decoder = Generator(8, "1", [3, 7, 11], [[1, 3, 5]] * 3,
                            [8, 8, 2, 2], 32, [16, 16, 4, 4], gin_channels=4).eval()
        engine = vc.ConversionEngine.__new__(vc.ConversionEngine)
        engine.torch = torch
        engine.model = SimpleNamespace(dec=decoder)
        with torch.inference_mode():
            condition = torch.randn(1, 4, 1)
            latent = torch.randn(1, 8, 68)
            full = decoder(latent, g=condition)[0, 0]
            for start in (4096, 8192, 8321):
                tail = engine._decode_tail(latent, condition, start)
                torch.testing.assert_close(tail, full[start:], rtol=1e-5, atol=1e-6)

    def test_preview_writes_only_complete_converted_take(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parsed = argparse.ArgumentParser()
            vc.configure(parsed.add_subparsers())
            args = parsed.parse_args(["vc-preview", "recording.wav", str(root / "converted.wav"),
                                      "--target", "reference.wav"])
            samples = np.linspace(-0.1, 0.1, vc.RATE, dtype="float32")
            engine = Mock()
            engine.convert.return_value = samples * -0.5
            engine.apply_watermark.side_effect = lambda converted: converted
            with patch.object(vc, "read_audio", return_value=samples), patch.object(
                vc, "ConversionEngine", return_value=engine
            ) as load:
                vc.preview(args)
                self.assertIs(load.call_args.kwargs["source_calibration"], samples)
                with wave.open(str(args.output), "rb") as w:
                    self.assertEqual((w.getframerate(), w.getnchannels(), w.getnframes()), (vc.RATE, 1, vc.RATE))
                    original = w.readframes(w.getnframes())
                    result = np.frombuffer(original, dtype="<i2")
                self.assertGreater(result[0], 0)
                self.assertLess(result[-1], 0)
                with self.assertRaises(FileExistsError):
                    vc.preview(args)
                args.output = root / "failed.wav"
                engine.convert.side_effect = RuntimeError("model failure")
                with self.assertRaisesRegex(RuntimeError, "model failure"):
                    vc.preview(args)
                self.assertEqual(list(root.iterdir()), [root / "converted.wav"])


if __name__ == "__main__":
    unittest.main()
