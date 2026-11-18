"""Check imports and the bundled-wheel FFmpeg before declaring setup complete."""
import os
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import sherpa_onnx
import sounddevice
import torch
import wavmark
from scipy.signal import resample_poly
from openvoice.models import SynthesizerTrn

assert all((sherpa_onnx.OfflineSpeakerDiarization, sounddevice.Stream, torch.inference_mode,
            wavmark.load_model, resample_poly, SynthesizerTrn))
env_dir = Path(sys.argv[1])
# Pick the wheel's actual bundled executable rather than a PATH fallback.
binaries = list((Path(imageio_ffmpeg.__file__).parent / "binaries").glob("ffmpeg-macos-*"))
if len(binaries) != 1:
    raise RuntimeError("The FFmpeg runtime wheel is missing its Mac executable.")
frontend = env_dir / "bin" / "ffmpeg"
if "--verify-only" not in sys.argv:
    temporary = frontend.with_name(".voice-lab-ffmpeg")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(binaries[0], frontend.parent))
    temporary.replace(frontend)
subprocess.run([str(frontend), "-version"], stdout=subprocess.DEVNULL, check=True)
print("Verified speech models, PortAudio bindings, and FFmpeg.", flush=True)
