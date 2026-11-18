"""Local inference with stock-compatible Kokoro v1 voice packs."""

import os
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path

from .reference import RATE, pcm_frames

MODEL = "hexgrad/Kokoro-82M"
PACK_SHAPE = (510, 1, 256)


@contextmanager
def _new_file(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(prefix=".voice-lab-", dir=output.parent)
    os.close(fd)
    try:
        yield staging
        os.link(staging, output)
    finally:
        os.unlink(staging)


def _validate_pack(pack, torch):
    if (not isinstance(pack, torch.Tensor) or pack.layout != torch.strided or tuple(pack.shape) != PACK_SHAPE
            or not torch.is_floating_point(pack) or not torch.isfinite(pack).all().item()):
        raise ValueError("Expected a finite Kokoro v1 voice tensor with shape [510, 1, 256].")
    return pack.detach().to(device="cpu", dtype=torch.float32)


def enroll_voice(args):
    if args.output.suffix.lower() != ".pt":
        raise ValueError("A Kokoro pack should have a .pt extension.")
    if len(pcm_frames(args.reference)) < 3 * RATE * 2:
        raise ValueError("Use at least 3 seconds of clear audio from one consenting speaker.")
    try:
        import torch
        from inno_kokoro.enroll import Tuner, enroll, read
    except ImportError as exc:
        raise RuntimeError("Install the voice dependencies with: python -m pip install -e '.[kokoro]'") from exc
    with _new_file(args.output) as staging:
        tuner = Tuner(backend="torch", check_update=False)
        pack, _ = enroll(*read(str(args.reference)), tuner, return_pt=True)
        with open(staging, "wb") as file:
            torch.save(_validate_pack(pack, torch), file)
    print(f"Saved Kokoro voice pack: {args.output}")


def speak(args):
    if args.output.suffix.lower() != ".wav":
        raise ValueError("Speech output should have a .wav extension.")
    if args.voice.suffix.lower() != ".pt" or not args.voice.is_file():
        raise ValueError("Select a local Kokoro .pt voice pack created by enroll or fit.")
    if not 0.5 <= args.speed <= 2.0:
        raise ValueError("--speed must be between 0.5 and 2.0.")
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Enter text to render.")
    try:
        import torch
        from kokoro import KPipeline
    except ImportError as exc:
        raise RuntimeError("Install the voice dependencies with: python -m pip install -e '.[kokoro]'") from exc
    with _new_file(args.output) as staging:
        pack = _validate_pack(torch.load(args.voice, map_location="cpu", weights_only=True), torch)
        pipeline = KPipeline(lang_code=args.lang, repo_id=MODEL, device=args.device)
        total = 0
        with wave.open(staging, "wb") as wav:
            wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            for result in pipeline(text, voice=pack, speed=args.speed):
                if result.audio is not None and result.audio.numel():
                    audio = result.audio.detach().cpu().clamp(-1, 1).mul(32767).round()
                    if total:
                        wav.writeframesraw(b"\x00\x00" * (RATE // 8))
                    wav.writeframesraw(audio.numpy().astype("<i2").tobytes())
                    total += audio.numel()
        if not total:
            raise ValueError("Kokoro produced no speech. Check the supplied text.")
    print(f"Saved generated speech: {args.output}")


def configure(commands):
    enroll = commands.add_parser("enroll", help="Create a Kokoro voice pack from a short reference")
    enroll.add_argument("reference", type=Path, help="Reviewed mono 24 kHz PCM16 WAV (3–30 seconds)")
    enroll.add_argument("output", type=Path, help="New Kokoro .pt voice pack")
    enroll.set_defaults(handler=enroll_voice)

    render = commands.add_parser("speak", help="Render local speech with a fitted or enrolled voice")
    text = render.add_mutually_exclusive_group(required=True)
    text.add_argument("--text")
    text.add_argument("--text-file", type=Path)
    render.add_argument("--voice", type=Path, required=True, help="Kokoro .pt voice pack")
    render.add_argument("--output", type=Path, required=True, help="New generated .wav")
    render.add_argument("--lang", choices=("a", "b"), default="a", help="a: US English; b: UK English")
    render.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    render.add_argument("--speed", type=float, default=1.0, help="Speech rate multiplier (0.5–2.0; default: 1.0)")
    render.set_defaults(handler=speak)
