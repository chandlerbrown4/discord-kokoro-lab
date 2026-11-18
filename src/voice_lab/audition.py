"""Build a level-matched, blind comparison for one listening trial."""

import json
import math
import random
import secrets
import shutil
import sys
import wave
from array import array
from pathlib import Path

from .diarize import _ffmpeg
from .reference import RATE
from .tracks import new_directory


def _sample(path, ffmpeg):
    if not path.is_file():
        raise ValueError(f"Audio file does not exist: {path}")
    raw = _ffmpeg(ffmpeg, path.resolve(), "-t", "30.01", "-ar", str(RATE),
                  "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1")
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder == "big":
        samples.byteswap()
    if not 0 < len(samples) <= 30 * RATE:
        raise ValueError(f"Select a clip of up to 30 seconds: {path}")
    # Gate by frame energy so interword pauses do not make a take sound louder.
    window = RATE // 50
    energies = []
    for start in range(0, len(samples), window):
        frame = samples[start:start + window]
        energies.append((sum(x * x for x in frame) / len(frame), len(frame)))
    strongest = max(energy for energy, _ in energies)
    if strongest < 1:
        raise ValueError(f"Clip has no audible signal: {path}")
    active = [(energy, size) for energy, size in energies if energy >= strongest * 0.001]
    rms = math.sqrt(sum(energy * size for energy, size in active) / sum(size for _, size in active))
    return samples, rms, max(abs(x) for x in samples)


def run(args):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install FFmpeg to prepare the listening comparison.")
    paths = {"reference": args.reference, "baseline": args.baseline, "candidate": args.candidate}
    audio = {role: _sample(path, ffmpeg) for role, path in paths.items()}
    if audio["baseline"][0] == audio["candidate"][0]:
        raise ValueError("The two generated takes contain identical audio.")
    # Use one shared RMS target low enough to leave every peak intact.
    level = min(32768 * 0.10, *(30000 * rms / peak for _, rms, peak in audio.values()))
    seed = args.seed if args.seed is not None else secrets.randbits(64)
    order = ["baseline", "candidate"]
    random.Random(seed).shuffle(order)
    with new_directory(args.output) as stage:
        with wave.open(str(stage / "listen.wav"), "wb") as wav:
            wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            for position, role in enumerate(["reference", *order]):
                if position:
                    wav.writeframesraw(bytes(2 * RATE * 3 // 4))
                samples, rms, _ = audio[role]
                gain = level / rms
                frames = array("h", (round(x * gain) for x in samples))
                if sys.byteorder == "big":
                    frames.byteswap()
                wav.writeframesraw(frames.tobytes())
        context = ("The reference demonstrates the speaker on different words."
                   if args.different_text else "All three takes should say the same line.")
        (stage / "scorecard.txt").write_text(
            f"Listen in order: reference, A, B. {context}\n"
            "Which sounds more like the speaker? A / B / tie / neither: ____\n"
            "Which sounds more natural? A / B / tie / neither: ____\n"
            "Wrong words, odd timing, or other audible differences: ____\n"
            "Record your answers before opening key.json.\n", encoding="utf-8",
        )
        key = {"reference_text": "different" if args.different_text else "same",
               "seed": seed, "A": order[0], "B": order[1],
               "sources": {role: path.name for role, path in paths.items()}}
        (stage / "key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    print(f"Listen to {args.output / 'listen.wav'} and fill in scorecard.txt before opening key.json.")


def configure(commands):
    parser = commands.add_parser("audition", help="Blindly compare two takes against a reference voice")
    for role, description in (("reference", "A separate real recording of the speaker"),
                              ("baseline", "The previous synthesized take"),
                              ("candidate", "The new synthesized take of the same text")):
        parser.add_argument(f"--{role}", type=Path, required=True, help=description)
    parser.add_argument("--output", type=Path, required=True, help="New directory for the trial")
    parser.add_argument("--different-text", action="store_true", help="Reference speaks different words")
    parser.add_argument("--seed", type=int, help="Repeat a previous trial order")
    parser.set_defaults(handler=run)
