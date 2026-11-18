"""Record a finite local mic take and audition a conversion of the same performance."""

import math
import shutil
import sys
import time
import wave
from array import array
from pathlib import Path
from threading import Event

from .audition import _sample
from .live import _sounddevice
from .models import _new_file
from .reference import RATE as PLAYBACK_RATE


MIC_RATE = 48000
GAP_SECONDS = 0.75


def _dbfs(peak):
    return f"{20 * math.log10(peak / 32768):.1f} dBFS" if peak else "silence"


def require_signal(samples):
    """Return the peak of a 48 kHz input with sustained signal above -50 dBFS."""
    block = MIC_RATE // 50
    active_frames = 0
    threshold = (32768 * 10 ** (-50 / 20)) ** 2
    for start in range(0, len(samples), block):
        part = samples[start:start + block]
        if sum(x * x for x in part) >= len(part) * threshold:
            active_frames += len(part)
    peak = max((abs(x) for x in samples), default=0)
    if active_frames < MIC_RATE * 0.15:
        raise ValueError(f"Too little microphone signal (peak {_dbfs(peak)}). "
                         "Check the mic ID, input level and microphone access, then record a new take.")
    return peak


def record_mic(mic, seconds, output):
    output = Path(output)
    if output.suffix.lower() != ".wav" or mic < 0 or not 1 <= seconds <= 30:
        raise ValueError("Choose a new WAV, an explicit mic input ID and a duration of 1–30 seconds.")
    sd = _sounddevice()
    target = round(seconds * MIC_RATE) * 2
    raw = bytearray()
    done = Event()
    failure = ""

    def receive(indata, frames, stream_time, status):
        nonlocal failure
        if status or len(indata) != 2 * frames:
            failure = f"The microphone dropped audio ({status or 'incomplete input block'}). Record a new take."
            done.set()
            raise sd.CallbackAbort
        raw.extend(indata[:min(len(indata), target - len(raw))])
        if len(raw) == target:
            done.set()
            raise sd.CallbackStop

    with _new_file(output) as staging:
        try:
            device = sd.query_devices(mic, "input")
            settings = dict(device=mic, samplerate=MIC_RATE, channels=1, dtype="int16")
            sd.check_input_settings(**settings)
            print(f"Recording {seconds:g}s from {device['name']}. Speak now.", flush=True)
            with sd.RawInputStream(**settings, callback=receive, finished_callback=done.set):
                deadline = time.monotonic() + seconds + 3
                previous = 0
                while not done.wait(timeout=1):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("The microphone did not deliver the requested audio in time.")
                    end = len(raw)
                    recent = array("h", raw[previous:end])
                    level = max((abs(x) for x in recent), default=0)
                    print(f"{end / (2 * MIC_RATE):.1f}s recorded, recent peak {_dbfs(level)}", flush=True)
                    previous = end
        except sd.PortAudioError as exc:
            raise RuntimeError(f"Could not capture from microphone {mic}. Check its device ID "
                               f"and microphone access: {exc}") from exc
        if failure or len(raw) != target:
            raise RuntimeError(failure or "The microphone closed before the take was complete.")
        samples = array("h", raw)
        peak = require_signal(samples)
        if sys.byteorder == "big":
            samples.byteswap()
        with wave.open(staging, "wb") as wav:
            wav.setparams((1, 2, MIC_RATE, 0, "NONE", "not compressed"))
            wav.writeframes(samples.tobytes())
    print(f"Saved {seconds:g}s of dry microphone audio to {output} (peak {_dbfs(peak)}).", flush=True)


def compare_takes(dry, converted, output):
    output = Path(output)
    if output.suffix.lower() != ".wav":
        raise ValueError("Choose a new .wav file for the comparison.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install FFmpeg to prepare the listening comparison.")
    takes = [_sample(Path(path), ffmpeg) for path in (dry, converted)]
    if takes[0][0] == takes[1][0]:
        raise ValueError("The dry and converted takes contain identical audio.")
    level = min(32768 * 0.1, *(30000 * rms / peak for _, rms, peak in takes))
    with _new_file(output) as staging, wave.open(staging, "wb") as wav:
        wav.setparams((1, 2, PLAYBACK_RATE, 0, "NONE", "not compressed"))
        for i, (samples, rms, _) in enumerate(takes):
            if i:
                wav.writeframesraw(bytes(round(PLAYBACK_RATE * GAP_SECONDS) * 2))
            gain = level / rms
            matched = array("h", (round(x * gain) for x in samples))
            if sys.byteorder == "big":
                matched.byteswap()
            wav.writeframesraw(matched.tobytes())
    boundary = len(takes[0][0]) / PLAYBACK_RATE + GAP_SECONDS
    print(f"Saved {output}. Dry microphone first, converted voice at {boundary:.2f}s.")


def configure(commands):
    capture = commands.add_parser("mic-record", help="Record a short local microphone take")
    capture.add_argument("output", type=Path, help="New dry microphone WAV at 48 kHz")
    capture.add_argument("--mic", type=int, required=True, help="Input ID from voice-lab live devices")
    capture.add_argument("--seconds", type=float, default=6, help="Duration in seconds, 1–30 (default 6)")
    capture.set_defaults(handler=lambda args: record_mic(args.mic, args.seconds, args.output))
    compare = commands.add_parser("mic-compare", help="Hear a dry take followed by its converted version")
    compare.add_argument("dry", type=Path)
    compare.add_argument("converted", type=Path)
    compare.add_argument("output", type=Path, help="New level-matched comparison WAV")
    compare.set_defaults(handler=lambda args: compare_takes(args.dry, args.converted, args.output))
