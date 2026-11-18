"""Route a physical microphone to a virtual cable or preview the same effect on a WAV."""

import math
import sys
import time
import wave
from array import array
from contextlib import closing, nullcontext
from pathlib import Path

from .models import _new_file


class MicEffect:
    """Stateful mono PCM16 filter shared by the hardware callback and file preview."""

    block_frames = 0
    delay_frames = 0

    def __init__(self, name, rate):
        self.name = name
        self.low_cut = math.exp(-2 * math.pi * 350 / rate)
        self.high_cut = 1 - math.exp(-2 * math.pi * 2900 / rate)
        self.bass = self.band = 0.0

    def process(self, data, *, little_endian=False):
        if self.name == "clean":
            return data
        samples = array("h")
        samples.frombytes(data)
        swap = little_endian and sys.byteorder == "big"
        if swap:
            samples.byteswap()
        bass, band = self.bass, self.band
        for i, sample in enumerate(samples):
            bass = self.low_cut * bass + (1 - self.low_cut) * sample
            band += self.high_cut * (sample - bass - band)
            samples[i] = max(-32768, min(32767, round(band)))
        self.bass, self.band = bass, band
        if swap:
            samples.byteswap()
        return samples.tobytes()


def processor(name, rate):
    if name == "giant":
        from .live_character import GiantVoice
        return closing(GiantVoice(rate))
    return nullcontext(MicEffect(name, rate))


def preview(args):
    if args.output.suffix.lower() != ".wav":
        raise ValueError("Choose a new .wav file for the preview.")
    with wave.open(str(args.input), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or not source.getnframes():
            raise ValueError("The preview needs a nonempty mono PCM16 WAV.")
        rate = source.getframerate()
        with processor(args.effect, rate) as effect, _new_file(args.output) as staging, wave.open(staging, "wb") as out:
            out.setparams(source.getparams())
            length = source.getnframes()
            # Drain the shifter's final samples and compensate for its input-to-output delay.
            block = effect.block_frames or 1024
            for offset in range(0, length + effect.delay_frames, block):
                count = max(0, min(block, length - offset))
                data = source.readframes(count)
                if len(data) != count * 2:
                    raise ValueError("Truncated input WAV.")
                result = effect.process(data.ljust(block * 2, b"\x00"), little_endian=True)
                first = max(0, effect.delay_frames - offset)
                last = min(block, length + effect.delay_frames - offset)
                out.writeframesraw(result[2 * first:2 * last])
    print(f"Saved {args.effect} preview to {args.output}")


def _sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install the live extra with python -m pip install -e '.[live]' "
                           "and make PortAudio available to Python.") from exc
    return sounddevice


def devices(args):
    sd = _sounddevice()
    try:
        devices = sd.query_devices()
        apis = sd.query_hostapis()
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Could not list audio devices: {exc}") from exc
    if not devices:
        print("No audio devices are visible to PortAudio.")
        return
    for index, device in enumerate(devices):
        print(f"{index:3d}  {device['name']}  ({device['max_input_channels']} in, "
              f"{device['max_output_channels']} out, {apis[device['hostapi']]['name']})")


def route(args):
    sd = _sounddevice()
    if args.mic < 0 or args.virtual_output < 0 or args.mic == args.virtual_output:
        raise ValueError("Choose distinct device IDs for the physical mic and the virtual output.")
    status_count = 0

    def callback(indata, outdata, frames, stream_time, status):
        nonlocal status_count
        if status:
            status_count += 1
        outdata[:] = effect.process(indata)

    try:
        mic = sd.query_devices(args.mic, "input")
        cable = sd.query_devices(args.virtual_output, "output")
        if mic["hostapi"] != cable["hostapi"]:
            raise ValueError("Select both devices from the same audio host API.")
        settings = {"channels": 1, "dtype": "int16", "samplerate": args.rate}
        sd.check_input_settings(device=args.mic, **settings)
        sd.check_output_settings(device=args.virtual_output, **settings)
        print(f"{mic['name']} -> {args.effect} -> {cable['name']} at {args.rate} Hz", flush=True)
        with processor(args.effect, args.rate) as effect:
            if effect.delay_frames:
                print(f"Effect delay {1000 * effect.delay_frames / args.rate:.1f} ms, "
                      f"excluding devices and Discord.")
            if args.check:
                print("Both devices advertise this mono PCM format. No stream started.")
                return
            with sd.RawStream(device=(args.mic, args.virtual_output), latency="low",
                              blocksize=effect.block_frames, callback=callback, **settings) as stream:
                print("Mic route active. Press Ctrl+C to stop.", flush=True)
                try:
                    while stream.active:
                        time.sleep(0.2)
                    raise RuntimeError("Audio stream stopped unexpectedly. Check both devices.")
                except KeyboardInterrupt:
                    pass
        print(f"Stopped. PortAudio flagged {status_count} callback(s) with timing or buffer issues.")
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Audio device error: {exc}. Check the device IDs, sample rate "
                           "and microphone access.") from exc


def configure(commands):
    live = commands.add_parser("live", help="Preview or route the user's microphone with a stock effect")
    actions = live.add_subparsers(dest="live_action", required=True)
    listing = actions.add_parser("devices", help="List PortAudio input and output IDs")
    listing.set_defaults(handler=devices)
    hardware = actions.add_parser("route", help="Send a mic to a virtual audio device")
    hardware.add_argument("--mic", required=True, type=int, help="Physical microphone input ID")
    hardware.add_argument("--virtual-output", required=True, type=int, help="Virtual cable playback device ID")
    hardware.add_argument("--rate", choices=(44100, 48000), type=int, default=48000)
    hardware.add_argument("--check", action="store_true", help="Probe device formats without starting the stream")
    hardware.set_defaults(handler=route)
    offline = actions.add_parser("preview", help="Apply the mic effect to a file with no hardware")
    offline.add_argument("input", type=Path, help="Mono PCM16 WAV")
    offline.add_argument("output", type=Path, help="New WAV for local playback")
    offline.set_defaults(handler=preview)
    for action in (hardware, offline):
        action.add_argument("--effect", choices=("clean", "radio", "giant"), default="clean",
                            help="Direct mic, band-limited radio, or a deeper fictional giant voice")
