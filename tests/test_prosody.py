import math
import tempfile
from pathlib import Path
import unittest
import wave

from voice_lab import prosody


class DeliveryTest(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
            import scipy, soundfile
        except ImportError:
            self.skipTest("requires speech analysis dependencies")
        self.np = np
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def tone(self, name, hz, seconds=1, sweep=0, pad=0, gain=.3):
        np = self.np
        rate = 24000
        t = np.arange(round(rate*seconds)) / rate
        frequency = hz * np.exp2(sweep*np.sin(2*np.pi*t*2)/12)
        samples = gain*np.sin(2*np.pi*np.cumsum(frequency)/rate)
        samples = np.pad(samples, round(pad*rate))
        dest = self.root / name
        with wave.open(str(dest), "wb") as f:
            f.setparams((1, 2, rate, 0, "NONE", "not compressed"))
            f.writeframes(np.rint(samples*32767).astype("<i2").tobytes())
        return dest

    def test_estimates_center_span_and_duration_on_separate_readings(self):
        original = self.tone("original.wav", 150)
        varied = self.tone("varied.wav", 150, sweep=4)
        slower = self.tone("slower.wav", 300, seconds=1.4)
        paired = prosody.errors(original, slower)
        self.assertAlmostEqual(paired["pitch_center_error_st"], 12., delta=.4)
        self.assertLess(paired["pitch_span_error_st"], .4)
        self.assertAlmostEqual(paired["duration_log_error"], math.log(1.4), delta=.03)
        self.assertGreater(prosody.errors(original, varied)["pitch_span_error_st"], 5)
        quiet = self.tone("padded.wav", 150, pad=.3, gain=.03)
        self.assertLess(prosody.errors(original, quiet)["duration_log_error"], .025)

    def test_silence_and_unvoiced_noise_cannot_imply_matching_pitch(self):
        original = self.tone("original.wav", 150)
        silent = self.tone("silent.wav", 100, gain=0)
        self.assertTrue(all(v is None for v in prosody.errors(original, silent).values()))
        rng = self.np.random.default_rng(2026)
        with wave.open(str(silent), "wb") as f:
            f.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            f.writeframes(rng.normal(0, 2200, 24000).astype("<i2").tobytes())
        result = prosody.errors(original, silent)
        self.assertIsNone(result["pitch_center_error_st"])
        self.assertIsNotNone(result["duration_log_error"])
        self.assertIsNone(prosody.summary([result])["pitch_span_error_st"])
