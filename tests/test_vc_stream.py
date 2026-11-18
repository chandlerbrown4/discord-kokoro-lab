import importlib.util
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from voice_lab import vc_stream as stream

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "VC streaming requires NumPy")
class StreamTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "device-rate replay requires SciPy")
    def test_replay_matches_the_warmed_live_worker_at_each_device_rate(self):
        class CountingEngine:
            device = "cpu"

            def __init__(self):
                self.reset_sampling()

            def reset_sampling(self):
                self.position = 0

            def convert(self, samples, *, output_start):
                self.position += 1
                return samples[output_start:] + self.position / 200

            def apply_watermark(self, samples):
                return samples

        profile = stream.BALANCED
        for rate in (44100, 48000):
            with self.subTest(rate=rate):
                block = round(profile.block * rate / stream.RATE)
                lookahead = round(profile.lookahead * rate / stream.RATE)
                dry = np.linspace(-0.4, 0.4, 2 * block + 33, dtype="float32")
                replay, _ = stream.render_blocks(CountingEngine(), dry, profile, device_rate=rate)
                bridge = stream._Exchange(rate, profile)
                converter = stream.StreamConverter(CountingEngine(), profile)
                probe = np.ones(profile.block, dtype="float32") * 0.1
                worker = threading.Thread(target=bridge.convert, args=(converter, rate, probe), daemon=True)
                worker.start()
                try:
                    self.assertTrue(bridge.ready.wait(10))
                    self.assertIsNone(bridge.error)
                    padded = np.pad(dry, (0, lookahead + (-(len(dry) + lookahead) % block)))
                    rendered = []
                    for index, chunk in enumerate(padded.reshape(-1, block)):
                        bridge.inputs.put((index, chunk), timeout=2)
                        actual_index, output = bridge.outputs.get(timeout=2)
                        self.assertEqual(actual_index, index)
                        rendered.append(output)
                    live = np.concatenate(rendered)[lookahead:lookahead + len(dry)]
                    np.testing.assert_array_equal(live, replay)
                finally:
                    bridge.shutdown.set()
                    worker.join(2)
                    self.assertFalse(worker.is_alive())

    def test_partial_block_and_lookahead_recovery(self):
        class Identity:
            @staticmethod
            def convert(value, *, output_start):
                return value[output_start:]

            @staticmethod
            def apply_watermark(value):
                return value

        for profile in (stream.QUALITY, stream.BALANCED, stream.FAST):
            for frames in (1024, profile.block, profile.block * 2 + 37):
                data = np.linspace(-0.5, 0.6, frames, dtype="float32")
                wet, times = stream.render_blocks(Identity(), data, profile)
                self.assertEqual(len(wet), frames)
                self.assertGreaterEqual(len(times), 1)
                np.testing.assert_allclose(wet, data, atol=1e-6)

    def test_incremental_mark_tracks_original_800ms_payload_positions(self):
        class Marked:
            @staticmethod
            def convert(x, *, output_start):
                return x[output_start:]

            @staticmethod
            def apply_watermark(window):
                self.assertEqual(len(window), stream.MARK_SAMPLES)
                return window + 0.01

        for profile in (stream.FAST, stream.BALANCED):
            converter = stream.StreamConverter(Marked(), profile)
            sample = np.ones(profile.block, "float32")
            joined = np.concatenate([converter.process(sample) for _ in range(10)])
            start = stream.QUALITY.block
            np.testing.assert_allclose(joined[start:start + stream.MARK_SAMPLES], 1.01, atol=1e-5)
            np.testing.assert_allclose(joined[start + stream.MARK_SAMPLES:start + stream.QUALITY.block], 1)

    def test_late_output_resumes_at_current_time_without_replaying_old_samples(self):
        bridge = stream._Exchange(48000, stream.FAST)
        bridge.set_processing_budget(0.02)
        frames = bridge.callback_frames
        zero = np.zeros((frames, 1), dtype="float32")
        out = zero.copy()
        clock = SimpleNamespace(inputBufferAdcTime=0, outputBufferDacTime=.01)
        for _ in range(bridge.start_tick + 1):
            bridge.transfer(zero, out, frames, clock, False)
            self.assertFalse(out.any())
        self.assertEqual(bridge.late, 1)
        bridge.inputs.get_nowait()
        full = np.repeat(np.arange(1, bridge.hop_ticks + 1, dtype="float32"), frames)
        bridge.outputs.put((0, full))
        bridge.transfer(zero, out, frames, clock, False)
        self.assertTrue(np.all(out == 2))
        for _ in range(bridge.hop_ticks - 2):
            bridge.transfer(zero, out, frames, clock, False)
        bridge.outputs.put((0, full))
        bridge.transfer(zero, out, frames, clock, True)
        self.assertFalse(out.any())
        self.assertEqual(bridge.late, 2)
        self.assertEqual(bridge.audio_flags, 1)
        self.assertAlmostEqual(bridge.max_lag, .01 + bridge.start_tick * .01 + stream.FAST.lookahead / stream.RATE)

    def test_device_clock_gap_expires_queued_and_inflight_audio(self):
        bridge = stream._Exchange(48000, stream.FAST)
        bridge.set_processing_budget(.2)
        frames = bridge.callback_frames
        data = np.ones((frames, 1), dtype="float32")
        out = data.copy()
        for tick in range(bridge.start_tick):
            adc = tick * stream.CALLBACK_SECONDS
            clock = SimpleNamespace(inputBufferAdcTime=adc, outputBufferDacTime=adc + .01)
            bridge.transfer(data, out, frames, clock, False)
        old = np.full(bridge.hop_ticks * frames, .7, dtype="float32")
        bridge.pending[0] = old
        bridge.outputs.put_nowait((1, old))
        clock = SimpleNamespace(inputBufferAdcTime=10, outputBufferDacTime=10.01)
        bridge.transfer(data, out, frames, clock, False)
        self.assertFalse(out.any())
        self.assertFalse(bridge.pending)
        self.assertEqual(bridge.late, 0)
        self.assertGreater(bridge.play_from, 1)

    def test_input_backlog_replaces_oldest_unstarted_hop(self):
        bridge = stream._Exchange(48000, stream.FAST)
        bridge.set_processing_budget(.2)
        frames = bridge.callback_frames
        out = np.zeros((frames, 1), dtype="float32")
        clock = SimpleNamespace(inputBufferAdcTime=0, outputBufferDacTime=.01)
        for value in (1, 2, 3):
            data = np.full_like(out, value)
            for _ in range(bridge.hop_ticks):
                bridge.transfer(data, out, frames, clock, False)
        index, block = bridge.inputs.get_nowait()
        self.assertEqual((index, bridge.dropped), (2, 2))
        self.assertTrue(np.all(block == 3))

    def test_preflight_falls_back_and_only_keeps_the_selected_worker(self):
        seen = []
        def clocked(exchange, converter, rate, probe):
            seen.append(exchange)
            exchange.can_stream = exchange.profile == stream.BALANCED
            exchange.ready.set()
            exchange.shutdown.wait(3)

        calibration = np.ones(stream.RATE, dtype="float32")
        with patch.object(stream._Exchange, "convert", clocked):
            chosen, worker = stream._start_worker(object(), 48000, calibration)
            try:
                self.assertEqual([x.profile for x in seen], [stream.FAST, stream.BALANCED])
                self.assertTrue(seen[0].shutdown.is_set())
                self.assertFalse(chosen.shutdown.is_set())
            finally:
                chosen.shutdown.set()
                worker.join(2)
            with self.assertRaisesRegex(RuntimeError, "deadline"):
                stream._start_worker(object(), 48000, calibration, "fast")
            chosen, worker = stream._start_worker(object(), 48000, calibration, "quality", offline=True)
            self.assertFalse(chosen.can_stream)
            chosen.shutdown.set()
            worker.join(2)
            self.assertTrue(all(x.shutdown.is_set() for x in seen))

    def test_auto_only_restarts_on_sustained_audible_overruns(self):
        exchange = stream._Exchange(48000, stream.FAST)
        audio = SimpleNamespace(active=True)
        observations = iter((5, 5, 9, 14))
        def advance(_):
            exchange.sequence += 100
            exchange.late = next(observations)
            return False

        with patch.object(exchange.shutdown, "wait", side_effect=advance):
            self.assertTrue(stream._watch_stream(audio, exchange, fallback=True))
        self.assertEqual(exchange.sequence, 400)

    def test_reset_forgets_old_speech(self):
        class Identity:
            def convert(self, x, *, output_start):
                return x[output_start:]

            def apply_watermark(self, x):
                return x

        converter = stream.StreamConverter(Identity())
        converter.process(np.ones(stream.BLOCK, dtype='float32'))
        converter.reset()
        self.assertFalse(converter.process(np.zeros(stream.BLOCK, dtype='float32')).any())
