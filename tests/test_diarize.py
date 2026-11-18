from types import SimpleNamespace
import unittest

from voice_lab.diarize import OUTPUT_RATE, Turn, _exclusive_turns


class DiarizationWindowTest(unittest.TestCase):
    def test_discards_cross_talk_and_merges_duplicate_labels(self):
        def segment(start, end, speaker):
            return SimpleNamespace(start=start, end=end, speaker=speaker)

        turns, overlap = _exclusive_turns([
            segment(-0.1, 4, 8),
            segment(3, 5, 8),
            segment(4, 7, 3),
            segment(7, 9, 8),
        ], 8 * OUTPUT_RATE)

        self.assertEqual(overlap, OUTPUT_RATE)
        self.assertEqual(turns, [
            Turn(0, 4 * OUTPUT_RATE, 8),
            Turn(5 * OUTPUT_RATE, 7 * OUTPUT_RATE, 3),
            Turn(7 * OUTPUT_RATE, 8 * OUTPUT_RATE, 8),
        ])
