"""Local, bounded-window OpenVoice conversion for microphone and file playback."""

import math
import queue
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from .models import _new_file
from .vc import ConversionEngine, RATE, add_calibration_options, calibration_options, read_audio


@dataclass(frozen=True)
class StreamProfile:
    block: int
    left: int
    lookahead: int


# All profiles resample to an integer number of frames at 44.1 and 48 kHz.
QUALITY = StreamProfile(17640, 4096, 2048)
BALANCED = StreamProfile(5292, 8192, 2048)
FAST = StreamProfile(3528, 8192, 2048)
BLOCK = QUALITY.block
LOOKAHEAD = QUALITY.lookahead
FADE = 512
MARK_SAMPLES = 16000
CALLBACK_SECONDS = 0.01


PROFILES = {"fast": FAST, "balanced": BALANCED, "quality": QUALITY}


class StreamConverter:
    def __init__(self, engine, profile=QUALITY):
        import numpy as np

        self.engine = engine
        self.profile = profile
        self.fade = np.linspace(0, 1, FADE, dtype='float32')
        self.reset()

    def reset(self):
        import numpy as np

        self.history = np.zeros(self.profile.left + self.profile.lookahead, 'float32')
        self.overlap = np.zeros(FADE, 'float32')
        self.mark_window = np.zeros(MARK_SAMPLES, 'float32')
        self.mark_position = 0

    def _mark(self, out):
        # Place a payload in the first 16k samples of each 800 ms interval,
        # as in the quality profile. Only emit the prefix observed so far.
        offset = 0
        while offset < len(out):
            position = self.mark_position
            count = min(len(out) - offset, QUALITY.block - position)
            marked_count = min(count, MARK_SAMPLES - position)
            if marked_count > 0:
                part = slice(offset, offset + marked_count)
                segment = slice(position, position + marked_count)
                self.mark_window[segment] = out[part]
                marked = self.engine.apply_watermark(self.mark_window)
                out[part] = marked[segment]
            self.mark_position += count
            offset += count
            if self.mark_position == QUALITY.block:
                self.mark_position = 0
                self.mark_window.fill(0)
        return out

    def process_device_block(self, block, device_rate):
        from scipy.signal import resample_poly
        import numpy as np

        divisor = math.gcd(device_rate, RATE)
        device_ratio, model_ratio = device_rate // divisor, RATE // divisor
        converted = self.process(resample_poly(block, model_ratio, device_ratio))
        out = resample_poly(converted, device_ratio, model_ratio)
        if out.shape != block.shape or not np.isfinite(out).all():
            raise RuntimeError("Conversion produced incomplete or invalid audio.")
        return np.clip(out, -1, 1, out=out)

    def process(self, block):
        import numpy as np

        p = self.profile
        if block.shape != (p.block,):
            raise ValueError(f"Expected {p.block} mono samples.")
        window = np.concatenate((self.history, block))
        self.history = window[-(p.left + p.lookahead):].copy()
        if np.any(window):
            rendered = self.engine.convert(window, output_start=p.left)
            out = rendered[:p.block].copy()
            out[:FADE] = self.overlap * (1 - self.fade) + out[:FADE] * self.fade
            self.overlap = rendered[p.block:p.block + FADE].copy()
        else:
            self.overlap.fill(0)
            out = np.zeros(p.block, 'float32')
        if p == QUALITY:
            return self.engine.apply_watermark(out)
        return self._mark(out)


def render_blocks(engine, samples, profile=QUALITY, device_rate=RATE):
    """Replay the live processor, trim its lookahead, and collect per-block times."""
    import numpy as np

    converter = StreamConverter(engine, profile)
    block_frames = round(profile.block * device_rate / RATE)
    lookahead = round(profile.lookahead * device_rate / RATE)
    end_pad = lookahead + (-(len(samples) + lookahead) % block_frames)
    padded = np.pad(samples, (0, end_pad))
    audio, times = [], []
    for i in range(0, len(padded), block_frames):
        block = padded[i:i + block_frames]
        started = time.perf_counter()
        if device_rate == RATE:
            audio.append(converter.process(block))
        else:
            audio.append(converter.process_device_block(block, device_rate))
        times.append(time.perf_counter() - started)
    return np.concatenate(audio)[lookahead:lookahead + len(samples)], times


def replay(args):
    import numpy as np

    if args.output.suffix.lower() != '.wav':
        raise ValueError('Choose a new WAV output for replay.')
    with _new_file(args.output) as temp:
        rate = getattr(args, "device_rate", None) or RATE
        samples = read_audio(args.input, rate=rate)
        calibration = args.source_calibration
        if calibration is None:
            calibration = samples if rate == RATE else args.input
        engine = ConversionEngine(target=args.target,
                                  source_calibration=calibration,
                                  weights=args.weights, threads=args.threads, device=args.device,
                                  target_adapter=getattr(args, "target_adapter", None),
                                  **calibration_options(args))
        profile_name = getattr(args, "profile", None)
        if profile_name in PROFILES:
            profile = PROFILES[profile_name]
        elif profile_name == "auto" or rate != RATE:
            probe = read_audio(args.source_calibration or args.input)
            exchange, worker = _start_worker(engine, rate if rate != RATE else 48000, probe, offline=True)
            exchange.shutdown.set()
            worker.join()
            profile = exchange.profile
        else:
            profile = BALANCED if args.device == "mps" else QUALITY
        result, elapsed = render_blocks(engine, samples, profile, rate)
        pcm = np.rint(np.clip(result, -1, 1) * 32767).astype('<i2')
        with wave.open(temp, 'wb') as out:
            out.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
            out.writeframes(pcm.tobytes())
    late = sum(t > profile.block / RATE for t in elapsed)
    print(f"Saved streamed conversion: {args.output}. {len(elapsed)} blocks, {late} over "
          f"{profile.block / RATE:.3f}s; median {np.median(elapsed):.3f}s, max {max(elapsed):.3f}s.")


class _Exchange:
    """The callback exchanges only owned arrays. Conversion happens on one worker."""
    def __init__(self, rate, profile):
        import numpy as np

        self.profile = profile
        self.callback_frames = round(rate * CALLBACK_SECONDS)
        self.hop_ticks = round(profile.block / RATE / CALLBACK_SECONDS)
        self.input_block = np.empty(self.hop_ticks * self.callback_frames, 'float32')
        self.inputs = queue.Queue(maxsize=1)
        self.outputs = queue.Queue(maxsize=2)
        self.pending = {}
        self.sequence = 0
        self.last_adc = None
        self.play_from = 0
        self.audio_flags = self.dropped = self.late = 0
        self.max_lag = None
        self.shutdown = threading.Event()
        self.ready = threading.Event()
        self.warm_seconds = None
        self.can_stream = False
        self.max_work_seconds = 0.0
        self.error = None

    def set_processing_budget(self, seconds):
        # A complete input block is available at the *last* callback of its
        # hop. Reserve whole callbacks for the measured worker time and jitter.
        self.start_tick = self.hop_ticks - 1 + math.ceil(seconds / CALLBACK_SECONDS)

    def transfer(self, data, out, frames, clock, status):
        import numpy as np

        out.fill(0)
        if frames != self.callback_frames:
            self.error = RuntimeError("The audio driver changed its callback size.")
            self.shutdown.set()
            return
        adc = clock.inputBufferAdcTime
        if self.last_adc is not None and adc - self.last_adc > 0.1:
            # A suspended device can resume hundreds of ms later without
            # delivering the intervening callbacks. Start a fresh full hop;
            # even an in-flight result from before the gap is now expired.
            self.sequence = (self.sequence // self.hop_ticks + 2) * self.hop_ticks
            self.play_from = self.sequence // self.hop_ticks
            self.pending.clear()
        self.last_adc = adc
        tick = self.sequence
        self.sequence += 1
        self.audio_flags += bool(status)
        play_index, part = divmod(tick - self.start_tick, self.hop_ticks)
        while True:
            try:
                index, block = self.outputs.get_nowait()
            except queue.Empty:
                break
            if index >= max(play_index, self.play_from):
                self.pending[index] = block
        if play_index >= self.play_from:
            block = self.pending.get(play_index)
            if block is None:
                self.late += 1
            else:
                start = part * frames
                out[:, 0] = block[start:start + frames]
                lag = (clock.outputBufferDacTime - clock.inputBufferAdcTime
                       + self.start_tick * CALLBACK_SECONDS + self.profile.lookahead / RATE)
                if math.isfinite(lag):
                    self.max_lag = max(self.max_lag or 0, lag)
            if part == self.hop_ticks - 1:
                self.pending.pop(play_index, None)

        input_index, part = divmod(tick, self.hop_ticks)
        start = part * frames
        self.input_block[start:start + frames] = data[:, 0]
        if part == self.hop_ticks - 1:
            # Keep the newest audio when the worker falls behind. A missing
            # sequence resets its causal history before rendering resumes.
            if self.inputs.full():
                try:
                    self.inputs.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
            self.inputs.put_nowait((input_index, self.input_block))
            self.input_block = np.empty_like(self.input_block)

    def convert(self, converter, device_rate, probe):
        try:
            from scipy.signal import resample_poly

            if sys.platform == "darwin":
                from ctypes import CDLL
                # QOS_CLASS_USER_INTERACTIVE on the live worker only.
                CDLL(None).pthread_set_qos_class_self_np(0x21, 0)

            gcd = math.gcd(device_rate, RATE)
            device_ratio, model_ratio = device_rate // gcd, RATE // gcd

            # PyTorch and driver initialization can be thread-specific. Warm
            # the actual worker and both resamplers before opening the mic.
            device_probe = resample_poly(probe, device_ratio, model_ratio)
            device_frames = self.hop_ticks * self.callback_frames
            if len(device_probe) < device_frames:
                raise ValueError("Voice calibration is shorter than one model block.")
            converter.process_device_block(device_probe[:device_frames], device_rate)
            times = []
            hop = self.profile.block / RATE
            # Clock the probe at live cadence, including idle time between
            # kernels. Back-to-back GPU timings understate wake-up cost.
            due = time.monotonic()
            peak_response = 0
            count = max(4, math.ceil(1.6 / hop))
            for index in range(count):
                wait = due - time.monotonic()
                if self.shutdown.wait(max(0.0, wait)):
                    return
                position = index % (len(device_probe) // device_frames) * device_frames
                mark = time.monotonic()
                converter.process_device_block(device_probe[position:position + device_frames], device_rate)
                finished = time.monotonic()
                times.append(finished - mark)
                peak_response = max(peak_response, finished - due)
                due += hop
            self.warm_seconds = max(times)
            self.can_stream = sum(times) / len(times) < 0.85 * hop and peak_response < 1.5 * hop
            # Short live Metal jobs can briefly take twice their warm time,
            # especially when resuming after a pause. Leave room for that
            # response without accumulating queued audio across the call.
            reserve = max(peak_response * 1.35 + 0.025, 0.06)
            if converter.engine.device == "mps":
                # A warm response already near an entire hop leaves too
                # little capacity for sustained playback on a shared GPU.
                self.can_stream &= peak_response < 0.85 * hop
                reserve = max(reserve, min(hop + 0.04, 0.3), peak_response * 2 + 0.025)
            self.set_processing_budget(reserve)
            converter.reset()
            # The calibration probe must not consume the first live phrase's synthesis noise.
            converter.engine.reset_sampling()
            self.ready.set()
            previous = None
            while not self.shutdown.is_set():
                try:
                    index, block = self.inputs.get(timeout=.1)
                except queue.Empty:
                    continue
                if previous is not None and index != previous + 1:
                    converter.reset()
                previous = index
                start = time.perf_counter()
                out = converter.process_device_block(block, device_rate)
                self.max_work_seconds = max(self.max_work_seconds, time.perf_counter() - start)
                try:
                    self.outputs.put_nowait((index, out))
                except queue.Full:
                    # The callback rejects expired results after device stalls.
                    pass
        except Exception as error:
            self.error = error
            self.shutdown.set()
        finally:
            self.ready.set()


def _start_worker(engine, rate, calibration, requested=None, *, offline=False, after=None):
    """Select the smallest tested hop that this device sustains, on the worker that will run it."""
    import numpy as np

    candidates = (PROFILES[requested],) if requested in PROFILES else (FAST, BALANCED, QUALITY)
    if after is not None:
        candidates = candidates[candidates.index(after) + 1:]
    for profile in candidates:
        # Include a full watermark cycle and changing speech around the peak.
        length = min(len(calibration), math.ceil(1.6 * RATE / profile.block) * profile.block)
        start = max(0, min(int(np.argmax(np.abs(calibration))) - length // 2, len(calibration) - length))
        probe = calibration[start:start + length]
        probe = np.pad(probe, (0, -len(probe) % profile.block))
        converter = StreamConverter(engine, profile)
        exchange = _Exchange(rate, profile)
        worker = threading.Thread(target=exchange.convert, args=(converter, rate, probe), daemon=True)
        worker.start()
        selected = False
        try:
            exchange.ready.wait()
            if exchange.error:
                raise RuntimeError(f"Could not start the converter: {exchange.error}") from exchange.error
            if exchange.can_stream or (offline and profile == QUALITY):
                selected = True
                return exchange, worker
        finally:
            if not selected:
                exchange.shutdown.set()
                worker.join()
    raise RuntimeError("Converter could not sustain the audio block deadline. Try --device mps on "
                       "Apple Silicon or adjust --threads.")


def monitor(args):
    from .live import _sounddevice

    sd = _sounddevice()
    if min(args.mic, args.output_device) < 0:
        raise ValueError("Choose explicit microphone and headphone device IDs.")
    try:
        mic = sd.query_devices(args.mic, 'input')
        output = sd.query_devices(args.output_device, 'output')
        if mic['hostapi'] != output['hostapi']:
            raise ValueError("Use devices from the same audio host API.")
        settings = dict(channels=1, dtype='float32', samplerate=args.rate)
        sd.check_input_settings(device=args.mic, **settings)
        sd.check_output_settings(device=args.output_device, **settings)
        print(f"{mic['name']} -> OpenVoice V2 -> {output['name']} at {args.rate} Hz")
        engine = ConversionEngine(target=args.target, source_calibration=args.source_calibration,
                                  weights=args.weights, threads=args.threads, device=args.device,
                                  target_adapter=getattr(args, "target_adapter", None),
                                  **calibration_options(args))
        calibration = read_audio(args.source_calibration)
        requested = getattr(args, "profile", None)
        previous = None
        while True:
            exchange, worker = _start_worker(engine, args.rate, calibration, requested, after=previous)
            try:
                profile = exchange.profile
                delay = exchange.start_tick * CALLBACK_SECONDS + profile.lookahead / RATE
                print(f"Live timing: {round(profile.block / RATE * 1000)} ms blocks · "
                      f"{round(delay * 1000)} ms estimated software delay", flush=True)
                if args.check:
                    return
                fallback = requested in (None, "auto") and profile != QUALITY
                with sd.Stream(device=(args.mic, args.output_device), latency='low', **settings,
                               blocksize=exchange.callback_frames, callback=exchange.transfer) as audio:
                    print("Monitor active. Press Ctrl+C to stop.", flush=True)
                    if not _watch_stream(audio, exchange, fallback):
                        return
            finally:
                exchange.shutdown.set()
                worker.join()
                if exchange.sequence:
                    print(f"{exchange.dropped} input hops dropped; {exchange.late} late 10 ms outputs; "
                          f"{exchange.audio_flags} device warnings.")
                if exchange.max_lag is not None:
                    print(f"Maximum estimated input-to-output delay: {exchange.max_lag:.3f}s.")
            previous = profile
            print("Live timing: Recalibrating after repeated missed audio deadlines.", flush=True)
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Audio hardware rejected the selected route: {exc}") from exc


def _watch_stream(audio, exchange, fallback):
    """Switch to the next profile only after consecutive seconds with audible losses."""
    previous_tick = previous_loss = strikes = 0
    try:
        while audio.active and not exchange.shutdown.wait(.1):
            if fallback and exchange.sequence - previous_tick >= round(1 / CALLBACK_SECONDS):
                previous_tick = exchange.sequence
                loss = exchange.late
                strikes = strikes + 1 if loss - previous_loss >= 4 else 0
                previous_loss = loss
                if strikes >= 2:
                    return True
    except KeyboardInterrupt:
        return False
    if exchange.error:
        raise RuntimeError(f"Conversion stopped: {exchange.error}") from exchange.error
    raise RuntimeError("Audio stream stopped unexpectedly.")


def add_stream_options(parser):
    parser.add_argument('--profile', choices=('auto', 'fast', 'balanced', 'quality'),
                        help='160, 240, or 800 ms model blocks (default: automatically measure the fastest sustainable profile)')
    add_calibration_options(parser)


def add_device_replay_option(parser):
    parser.add_argument('--device-rate', type=int, choices=(44100, 48000),
                        help='Replay the live input and output resamplers at this device rate')


def configure(commands):
    replay_parser = commands.add_parser('vc-replay', help='Render the microphone converter block by block to a WAV')
    replay_parser.add_argument('input', type=Path, help='A local microphone take, up to 30 s')
    replay_parser.add_argument('output', type=Path, help='New converted WAV')
    add_device_replay_option(replay_parser)
    replay_parser.set_defaults(handler=replay)
    monitor_parser = commands.add_parser('vc-monitor', help='Monitor a converted microphone on an explicit output')
    monitor_parser.add_argument('--mic', required=True, type=int)
    monitor_parser.add_argument('--output-device', required=True, type=int, help='Headphone output for local monitoring')
    monitor_parser.add_argument('--rate', choices=(44100, 48000), default=48000, type=int)
    monitor_parser.add_argument('--check', action='store_true', help='Test model speed and device formats without opening a stream')
    monitor_parser.set_defaults(handler=monitor)
    for parser in (replay_parser, monitor_parser):
        add_stream_options(parser)
        parser.add_argument('--target-adapter', type=Path, help='Optional fitted OpenVoice V2 target vector')
        parser.add_argument('--target', required=True, type=Path, help='Approved target voice reference')
        parser.add_argument('--source-calibration', type=Path, required=parser is monitor_parser,
                            help='Reference speech from the mic speaker; replay defaults to the input take')
        parser.add_argument('--weights', type=Path, help='Local pinned converter weights directory')
        parser.add_argument('--threads', type=int, default=4)
        parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu', help='OpenVoice compute device')
