"""Offline OpenVoice V2 speech-to-speech conversion and its shared streaming kernel."""

import hashlib
import json
import shutil
import time
import wave
from pathlib import Path

from .diarize import _ffmpeg
from .models import _new_file

RATE = 22050
HOP = 256
REVISION = "fd981100305a0e4291f93a9ad169c6d9f7bed54a"
MARK_REVISION = "0ad3c7b74f641bddb61f6b85cdf2de0d93a5bfef"
MARK_FILE = "step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl"
DIGESTS = {
    "config.json": "9dfff60350b8c63f2c664efd92a61b2516efb22671466960f0e5dfebd881fa47",
    "checkpoint.pth": "9652c27e92b6b2a91632590ac9962ef7ae2b712e5c5b7f4c34ec55ee2b37ab9e",
    MARK_FILE: "7a3d873d115c2c644685fb4cf109bdb299718296b88c871d3cf1c8e4cbc13144",
}


def _assets(weights, *, local_only):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install the OpenVoice dependencies described in the README.") from exc
    files = {}
    for name, expected in DIGESTS.items():
        if weights is not None and name != MARK_FILE:
            path = Path(weights) / name
        else:
            if name == MARK_FILE:
                repo, filename, revision = "M4869/WavMark", name, MARK_REVISION
            else:
                repo, filename, revision = "myshell-ai/OpenVoiceV2", "converter/" + name, REVISION
            try:
                path = Path(hf_hub_download(repo, filename, revision=revision, local_files_only=local_only))
            except OSError as exc:
                if local_only:
                    raise RuntimeError("Missing OpenVoice or WavMark assets. Run voice-lab vc-fetch first.") from exc
                raise
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {name}.")
        files[name] = path
    return files


def read_audio(path, max_seconds=30, *, rate=RATE):
    """Decode a bounded local clip to mono float32 at the requested rate."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install the OpenVoice dependencies described in the README.") from exc

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install FFmpeg to decode local voice samples.")
    path = Path(path).expanduser().resolve(strict=True)
    raw = _ffmpeg(ffmpeg, path, "-t", str(max_seconds + 0.05), "-ar", str(rate),
                  "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1")
    audio = np.frombuffer(raw, dtype="<f4").copy()
    if not 0 < len(audio) <= max_seconds * rate or not np.all(np.isfinite(audio)):
        raise ValueError(f"Select finite audio up to {max_seconds}s long: {path}")
    return audio


class ConversionEngine:
    """Convert one source speaker into one target from local mono audio at RATE Hz."""

    def __init__(self, *, target, source_calibration, weights=None, threads=4, device="cpu",
                 mic_peak_dbfs=-12.0, max_mic_gain_db=18.0, target_adapter=None):
        try:
            import numpy as np
            import torch
            import wavmark
            from openvoice.models import SynthesizerTrn
            from openvoice.mel_processing import spectrogram_torch
        except ImportError as exc:
            raise RuntimeError("Install the pinned OpenVoice source and VC dependencies from the README.") from exc
        if not 1 <= threads <= 32:
            raise ValueError("Expected 1–32 CPU threads.")
        if device not in ("cpu", "mps"):
            raise ValueError("Choose the cpu or mps converter device.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple Metal is unavailable in this process. Select cpu or run on a Metal-capable Mac.")
        torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.spectrogram = spectrogram_torch
        files = _assets(weights, local_only=True)
        config = json.loads(files["config.json"].read_text(encoding="utf-8"))
        self.model = SynthesizerTrn(0, 513, n_speakers=0, **config["model"]).eval()
        checkpoint = torch.load(files["checkpoint.pth"], map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        # Bake the pretrained evaluation weights once, avoiding their recomputation per block.
        for layer in self.model.modules():
            if hasattr(layer, "weight_g") and hasattr(layer, "weight_v"):
                torch.nn.utils.remove_weight_norm(layer)
        self.marker = wavmark.load_model(str(files[MARK_FILE])).to(device)
        tag = np.unpackbits(np.frombuffer(b"OVC2", dtype="uint8")).astype("float32")
        self.mark_bits = torch.from_numpy(tag)[None].to(device)
        source = self._reference_audio(source_calibration)
        # One gain for the whole take keeps pauses and emphasis in proportion.
        self.source_gain = calibration_gain(source, mic_peak_dbfs, max_mic_gain_db)
        with torch.inference_mode():
            if target_adapter is None:
                self.target = self._embedding(self._reference_audio(target)).to(device)
            else:
                self.target = torch.from_numpy(read_target_adapter(target_adapter)).to(device)
            self.source = self._embedding(source * self.source_gain).to(device)
        self.model.to(device)
        self.reset_sampling()

    def reset_sampling(self):
        """Start a new session at the same point in the synthesis noise sequence."""
        self.torch.manual_seed(1701)

    def _spec(self, samples):
        y = self.torch.from_numpy(samples).float()[None]
        return self.spectrogram(y, 1024, RATE, HOP, 1024, center=False)

    @staticmethod
    def _reference_audio(reference):
        import numpy as np

        audio = read_audio(reference) if isinstance(reference, (str, Path)) else np.asarray(reference, dtype="float32")
        if (audio.ndim != 1 or not RATE <= len(audio) <= 30 * RATE
                or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-4):
            raise ValueError("Each voice reference needs 1–30 seconds of audible mono speech.")
        return audio

    def _embedding(self, audio):
        return self.model.ref_enc(self._spec(audio).transpose(1, 2)).unsqueeze(-1)

    def _decode_tail(self, latent, condition, start):
        """Decode only the live tail with the receptive field of the pinned V2 generator."""
        from torch.nn.functional import leaky_relu

        decoder = self.model.dec
        # Its 7-tap pre/post convolutions and four residual stages need
        # fewer than 16 input frames of left context in total.
        origin = max(0, start // HOP - 16)
        x = decoder.conv_pre(latent[:, :, origin:]) + decoder.cond(condition)
        scale = HOP
        for stage, upsample in enumerate(decoder.ups):
            stride = upsample.stride[0]
            x = upsample(leaky_relu(x, 0.1))
            scale //= stride
            origin *= stride
            # At this resolution the current residual bank sees at most
            # 60 samples on either side. All later stages combined fit
            # within another 68 (strides 8/8/2/2). Keep that full halo.
            discard = max(0, start // scale - 128 - origin)
            x = x[:, :, discard:]
            origin += discard
            first = stage * decoder.num_kernels
            residual = decoder.resblocks[first](x)
            for index in range(1, decoder.num_kernels):
                residual = residual + decoder.resblocks[first + index](x)
            x = residual / decoder.num_kernels
        wave = self.torch.tanh(decoder.conv_post(leaky_relu(x)))
        return wave[0, 0, start - origin:]

    def convert(self, samples, *, output_start=0):
        """Run the encoder on all context and return waveform samples from output_start."""
        import numpy as np

        source = np.asarray(samples, dtype="float32")
        if source.ndim != 1 or not 1024 <= len(source) <= 30 * RATE or not np.isfinite(source).all():
            raise ValueError("Conversion input must contain 1024 frames to 30 seconds of finite mono audio.")
        if not 0 <= output_start < len(source):
            raise ValueError("Requested audio starts outside the conversion window.")
        if not source.any():
            return np.zeros_like(source[output_start:])
        if self.source_gain > 1:
            source = np.clip(source * self.source_gain, -1, 1)
        padded = np.pad(source, (0, -len(source) % HOP))
        torch = self.torch
        with torch.inference_mode():
            spec = self._spec(padded).to(self.device)
            length = torch.tensor([spec.shape[-1]], device=self.device)
            if output_start:
                # Preserve full-context encoder/flows and synthesis noise.
                # Only the vocoder can skip already-played audio.
                model = self.model
                zero = torch.zeros_like(self.source)
                latent, _, _, mask = model.enc_q(spec, length, g=zero if model.zero_g else self.source, tau=0.3)
                latent = model.flow(latent, mask, g=self.source)
                latent = model.flow(latent, mask, g=self.target, reverse=True)
                wave_out = self._decode_tail(latent * mask, zero if model.zero_g else self.target, output_start)
            else:
                wave_out = self.model.voice_conversion(spec, length, self.source, self.target, tau=0.3)[0][0, 0]
            out = wave_out[:len(source) - output_start].cpu().numpy().copy()
        if len(out) != len(source) - output_start or not np.isfinite(out).all():
            raise RuntimeError("OpenVoice returned invalid or incomplete output.")
        return out

    def apply_watermark(self, samples):
        """Use the model's 16,000-sample WavMark windows on the output timeline."""
        import numpy as np

        out = np.asarray(samples, dtype="float32").copy()
        with self.torch.inference_mode():
            for start in range(0, len(out) - 15999, 32000):
                window = out[start:start + 16000]
                if not window.any():
                    continue
                signal = self.torch.from_numpy(window)[None].to(self.device)
                marked = self.marker.encode(signal, self.mark_bits)
                window[:] = marked.cpu().numpy().reshape(-1)
        return out


def calibration_gain(samples, mic_peak_dbfs=-12.0, max_mic_gain_db=18.0):
    """Calculate one bounded gain from the mic calibration take."""
    import numpy as np

    if not -60 <= mic_peak_dbfs <= 0 or not 0 <= max_mic_gain_db <= 36:
        raise ValueError("Choose a mic peak from -60 to 0 dBFS and a gain cap from 0 to 36 dB.")
    peak = float(np.max(np.abs(samples)))
    return min(10 ** (max_mic_gain_db / 20), max(1.0, 10 ** (mic_peak_dbfs / 20) / max(peak, 1e-8)))


def read_target_adapter(path):
    """Load one OpenVoice V2 target vector, never an executable checkpoint."""
    import numpy as np

    path = Path(path)
    if path.stat().st_size > 4096:
        raise ValueError("Unexpected target adapter size.")
    with path.open("rb") as src:
        if src.read(6) != b"\x93NUMPY":
            raise ValueError("Choose a NumPy target adapter produced by Voice Lab.")
        src.seek(0)
        target = np.load(src, allow_pickle=False)
    if (target.shape != (1, 256, 1) or target.dtype != np.float32
            or not np.isfinite(target).all() or not target.any()):
        raise ValueError("Invalid OpenVoice V2 target adapter.")
    return target


def add_calibration_options(parser):
    parser.add_argument("--mic-peak-dbfs", type=float, default=-12.0,
                        help="Peak level for raising a quiet calibration take, in dBFS (default: -12)")
    parser.add_argument("--max-mic-gain-db", type=float, default=18.0,
                        help="Maximum fixed mic gain in dB (default: 18)")


def calibration_options(args):
    return dict(mic_peak_dbfs=getattr(args, "mic_peak_dbfs", -12.0),
                max_mic_gain_db=getattr(args, "max_mic_gain_db", 18.0))


def fetch(args):
    for name, path in _assets(None, local_only=False).items():
        print(f"{name}  {path}")


def preview(args):
    if args.output.suffix.lower() != ".wav":
        raise ValueError("Use a new .wav output.")
    with _new_file(args.output) as staging:
        samples = read_audio(args.input)
        calibration = args.source_calibration if args.source_calibration else samples
        engine = ConversionEngine(target=args.target, source_calibration=calibration,
                                  weights=args.weights, threads=args.threads, device=args.device,
                                  target_adapter=getattr(args, "target_adapter", None),
                                  **calibration_options(args))
        started = time.monotonic()
        result = engine.apply_watermark(engine.convert(samples))
        elapsed = time.monotonic() - started
        pcm = (result.clip(-1, 1) * 32767).round().astype("<i2")
        with wave.open(staging, "wb") as output:
            output.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            output.writeframes(pcm.tobytes())
    print(f"Converted {len(samples) / RATE:.2f}s of speech in {elapsed:.2f}s after loading: {args.output}")


def configure(commands):
    assets = commands.add_parser("vc-fetch", help="Cache and verify pinned OpenVoice V2 and WavMark weights")
    assets.set_defaults(handler=fetch)
    command = commands.add_parser("vc-preview", help="Convert a local spoken take to a reviewed target voice")
    command.add_argument("input", type=Path, help="One spoken take, up to 30 seconds")
    command.add_argument("output", type=Path, help="New mono PCM16 WAV")
    command.add_argument("--target", required=True, type=Path, help="Approved target speaker reference WAV")
    command.add_argument("--target-adapter", type=Path, help="Optional fitted OpenVoice V2 target vector")
    command.add_argument("--source-calibration", type=Path, help="Separate reference of the mic speaker")
    command.add_argument("--weights", type=Path, help="Local directory containing the pinned converter weights")
    command.add_argument("--threads", type=int, default=4)
    add_calibration_options(command)
    command.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="CPU or Apple Metal")
    command.set_defaults(handler=preview)
