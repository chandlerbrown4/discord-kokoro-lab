"""Check the beta's model assets and runtime in a short-lived process."""

import json
from pathlib import Path
import shutil
import sys

from . import diarize, vc


def _report(**event):
    print("studio-setup " + json.dumps(event), flush=True)


def prepare(model_dir, requested_device, local_only):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is missing. Relaunch the app to repair its runtime.")
    _report(message="Preparing the speaker separation models…")
    diarize._models(model_dir, local_only)
    _report(message="Preparing the voice conversion models…")
    # An established installation should still launch offline.
    try:
        vc._assets(None, local_only=True)
    except RuntimeError:
        if local_only:
            raise
        vc._assets(None, local_only=False)

    _report(message="Checking the speech runtime…")
    import sherpa_onnx  # noqa: F401
    import torch
    import wavmark  # noqa: F401
    from openvoice.models import SynthesizerTrn  # noqa: F401

    if requested_device == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple Metal is unavailable in this process.")
    return requested_device


if __name__ == "__main__":
    try:
        device = prepare(Path(sys.argv[1]), sys.argv[2], "--local-files-only" in sys.argv[3:])
    except Exception as exc:
        _report(error=str(exc))
        sys.exit(1)
    _report(device=device)
