"""Fixed-block fictional voice effect using the Rubber Band 4 live C API."""

import ctypes as ct
import sys
from array import array
from ctypes.util import find_library
from pathlib import Path


def _load():
    # macOS find_library does not search the Homebrew prefix on all Python builds.
    paths = [find_library("rubberband")]
    if sys.platform == "darwin":
        paths += [str(prefix / "lib/librubberband.dylib") for prefix in
                  (Path("/opt/homebrew"), Path("/usr/local"))]
    for path in filter(None, paths):
        try:
            lib = ct.CDLL(path)
            state, size = ct.c_void_p, ct.c_uint
            channels = ct.POINTER(ct.POINTER(ct.c_float))
            for name, args, result in (
                ("new", [size, size, ct.c_int], state),
                ("delete", [state], None),
                ("set_pitch_scale", [state, ct.c_double], None),
                ("get_block_size", [state], size),
                ("get_start_delay", [state], size),
                ("shift", [state, channels, channels], None),
            ):
                fn = getattr(lib, "rubberband_live_" + name)
                fn.argtypes, fn.restype = args, result
            return lib
        except (OSError, AttributeError):
            continue
    raise RuntimeError("The giant effect requires the Rubber Band 4 shared library. "
                       "On macOS install it with brew install rubberband.")


class GiantVoice:
    """Lower pitch and vowel resonances together, preserving spoken tempo."""

    def __init__(self, rate):
        self.lib = _load()
        self.state = self.lib.rubberband_live_new(rate, 1, 0)  # short window, formants follow pitch
        if not self.state:
            raise RuntimeError("Could not initialize the Rubber Band live shifter.")
        self.lib.rubberband_live_set_pitch_scale(self.state, 2 ** (-4 / 12))
        self.block_frames = self.lib.rubberband_live_get_block_size(self.state)
        self.delay_frames = self.lib.rubberband_live_get_start_delay(self.state)
        block = ct.c_float * self.block_frames
        self.input, self.output = block(), block()
        mono = ct.POINTER(ct.c_float) * 1
        self.inputs, self.outputs = mono(self.input), mono(self.output)

    def close(self):
        self.lib.rubberband_live_delete(self.state)

    def process(self, data, *, little_endian=False):
        if len(data) != 2 * self.block_frames:
            raise ValueError(f"The live shifter requires {self.block_frames} frames per block.")
        samples = array("h")
        samples.frombytes(data)
        swap = little_endian and sys.byteorder == "big"
        if swap:
            samples.byteswap()
        for index, sample in enumerate(samples):
            self.input[index] = sample / 32768
        self.lib.rubberband_live_shift(self.state, self.inputs, self.outputs)
        for index, sample in enumerate(self.output):
            samples[index] = max(-32768, min(32767, round(sample * 32768)))
        if swap:
            samples.byteswap()
        return samples.tobytes()
