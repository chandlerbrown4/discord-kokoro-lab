"""Fit a stock-compatible Kokoro voice pack with KVoiceWalk."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
import zipfile
from pathlib import Path


TRAINER_REVISION = "3a38c6030cc4657df073c67ded37cdf7627c4969"
_RESULT_NAME = re.compile(r"fitted_(\d+)_[^_]+_[^_]+_reference\.pt")

# Run in the trainer's environment so the main CLI does not need PyTorch.
_EXPORT_PACK = """
import sys
import torch

pack = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
if not isinstance(pack, torch.Tensor) or pack.shape != (510, 1, 256):
    raise SystemExit('Expected a Kokoro v1.0 voice tensor of shape (510, 1, 256).')
if not pack.is_floating_point() or pack.layout != torch.strided:
    raise SystemExit('Expected a dense floating-point voice tensor.')
pack = pack.detach().to(dtype=torch.float32).contiguous().clone()
if not torch.isfinite(pack).all():
    raise SystemExit('The fitted voice contained non-finite values.')
torch.save(pack, sys.argv[2])
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reference", required=True, type=Path,
        help="Reviewed mono 24 kHz PCM WAV from one speaker (20–30 seconds recommended)",
    )
    parser.add_argument(
        "--transcript", required=True, type=Path,
        help="Reviewed UTF-8 text that matches the spoken English in the reference WAV",
    )
    parser.add_argument(
        "--trainer", required=True, type=Path,
        help=f"Local KVoiceWalk clone containing {TRAINER_REVISION[:12]}",
    )
    parser.add_argument("--output", required=True, type=Path, help="New Kokoro v1.0 .pt voice pack")
    parser.add_argument(
        "--trainer-python", type=Path,
        help="Python interpreter with KVoiceWalk dependencies (defaults to the trainer's .venv)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=1_000, help="Random walk steps (default: %(default)s)")


def _check_input(reference: Path, transcript: Path) -> str:
    try:
        with wave.open(str(reference), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getframerate() != 24_000 or audio.getnframes() < 1:
                raise ValueError("Reference must be a nonempty mono 24 kHz PCM WAV.")
    except (EOFError, wave.Error) as exc:
        raise ValueError("Reference must be a nonempty mono 24 kHz PCM WAV.") from exc

    try:
        text = transcript.read_text(encoding="utf-8-sig").strip()
    except UnicodeError as exc:
        raise ValueError("Transcript must be UTF-8 text.") from exc
    if not text:
        raise ValueError("Transcript is empty. Use the exact English spoken in the reference WAV.")
    return text


def _stage_trainer(trainer: Path, work_dir: Path) -> None:
    archive = work_dir / "trainer.zip"
    result = subprocess.run(
        [
            "git", "-C", str(trainer), "archive", "--format=zip", "--output", str(archive),
            TRAINER_REVISION, "main.py", "utilities", "voices",
        ],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise ValueError(
            f"Could not read pinned KVoiceWalk revision {TRAINER_REVISION} from {trainer}. "
            "Clone the upstream repo and fetch that revision."
        )
    with zipfile.ZipFile(archive) as source:
        source.extractall(work_dir)
    # A reference can have zero beat variance, so relative error for that
    # feature is undefined. Librosa can also return tempo as a one-item array.
    scorer = work_dir / "utilities" / "fitness_scorer.py"
    source_text = scorer.read_text(encoding="utf-8")
    patches = {
        "            diff = abs((value - self.target_features[key])/self.target_features[key])":
            "            target = self.target_features[key]\n"
            "            if target == 0 or not np.isfinite(target) or not np.isfinite(value):\n"
            "                continue\n"
            "            diff = abs((value - target) / target)",
        'features["tempo"] = float(tempo)': 'features["tempo"] = float(np.asarray(tempo).item())',
    }
    for original, replacement in patches.items():
        if source_text.count(original) != 1:
            raise ValueError("Pinned KVoiceWalk scorer does not match the expected source.")
        source_text = source_text.replace(original, replacement)
    scorer.write_text(source_text, encoding="utf-8")


def _best_pack(work_dir: Path) -> Path:
    candidates = []
    for path in work_dir.glob("out/fitted_reference_*/fitted_*.pt"):
        match = _RESULT_NAME.fullmatch(path.name)
        if match and path.is_file():
            candidates.append((int(match[1]), path))
    if not candidates:
        raise ValueError("KVoiceWalk did not beat its starting voice. Try a cleaner reference or increase --steps.")
    # KVoiceWalk writes a checkpoint only on strict improvement. Pick by step
    # number because scores in filenames are rounded.
    return max(candidates)[1]


def run(args: argparse.Namespace) -> int:
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    output = args.output.expanduser().absolute()
    if output.suffix != ".pt":
        raise ValueError("--output must end in .pt.")
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists: {output}")
    reference = args.reference.expanduser().resolve(strict=True)
    text = _check_input(reference, args.transcript.expanduser())
    trainer = args.trainer.expanduser().resolve(strict=True)
    default_python = trainer / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    python = args.trainer_python or (default_python if default_python.is_file() else Path(sys.executable))
    # Keep the virtualenv's path rather than following its python symlink.
    python = python.expanduser().absolute()
    if not python.is_file():
        raise FileNotFoundError(f"Trainer Python interpreter does not exist: {python}")

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    environment["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    venv = python.parent.parent
    if (venv / "pyvenv.cfg").is_file():
        environment["VIRTUAL_ENV"] = str(venv)
        environment["PATH"] = os.pathsep.join((str(python.parent), *os.get_exec_path(environment)))
    with tempfile.TemporaryDirectory(prefix="voice-lab-fit-") as work:
        work_dir = Path(work)
        _stage_trainer(trainer, work_dir)
        shutil.copyfile(reference, work_dir / "reference.wav")
        (work_dir / "transcript.txt").write_text(text, encoding="utf-8")
        command = [
            str(python), "-u", "main.py",
            "--target_audio", "reference.wav",
            "--target_text", "transcript.txt",
            "--device", args.device,
            "--output_name", "fitted",
            "--step_limit", str(args.steps),
        ]
        try:
            subprocess.run(command, cwd=work_dir, env=environment, stdin=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"KVoiceWalk exited with code {exc.returncode}. "
                "Check its log and the trainer Python environment."
            ) from exc

        best = _best_pack(work_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".voice-lab-export-", dir=output.parent) as staging:
            staged = Path(staging) / "voice.pt"
            try:
                subprocess.run(
                    [str(python), "-c", _EXPORT_PACK, str(best), str(staged)],
                    cwd=work_dir, env=environment, check=True,
                )
            except subprocess.CalledProcessError as exc:
                raise ValueError("KVoiceWalk produced an invalid Kokoro voice pack. See the error above.") from exc
            staged.chmod(0o600)
            # Publish only a complete file, without overwriting competing output.
            os.link(staged, output)

    print(f"Fitted Kokoro voice pack: {output}")
    return 0
