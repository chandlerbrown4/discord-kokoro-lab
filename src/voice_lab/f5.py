"""Reference-conditioned local speech synthesis with pinned F5 and Vocos weights."""

import platform
import wave
from pathlib import Path

from .models import _new_file
from .reference import RATE, pcm_frames

MODEL = "lucasnewman/f5-tts-mlx"
VOCODER = "lucasnewman/vocos-mel-24khz"
WEIGHTS = {
    MODEL: ("2d719cadec8fd3887c8599475a5f65924d523654",
            ("model_v1_8b.safetensors", "duration_v2.safetensors", "vocab.txt")),
    VOCODER: ("ee89d00fac94b61e1235e0869f3e8d49711b7484", ("config.yaml", "model.safetensors")),
}
HOP = 256
MAX_HOPS = 1920
PACE = 0.7803748693834901


def _snapshots(local_only):
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise RuntimeError("The F5 MLX backend requires an Apple Silicon Mac.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install the F5 dependencies with python -m pip install -e '.[f5]'") from exc
    snapshots = {}
    for repo, (revision, filenames) in WEIGHTS.items():
        for name in filenames:
            try:
                path = hf_hub_download(repo, name, revision=revision, local_files_only=local_only)
            except OSError as exc:
                if local_only:
                    raise RuntimeError(f"Missing cached {repo}/{name}. Run voice-lab f5-fetch first.") from exc
                raise
            snapshots[repo] = Path(path).parent
    return snapshots


def fetch(args):
    for repo, path in _snapshots(local_only=False).items():
        print(f"{repo} {WEIGHTS[repo][0]}  {path}")


def _load_model(snapshots):
    # Upstream from_pretrained always resolves both repos from the network or
    # the cached main branches. Instantiate its v1 8-bit architecture from
    # explicit local snapshot paths so both revisions stay reproducible.
    import mlx.core as mx
    import mlx.nn as nn
    from f5_tts_mlx import F5TTS
    from f5_tts_mlx.dit import DiT
    from f5_tts_mlx.duration import DurationPredictor, DurationTransformer
    from vocos_mlx import Vocos

    checkpoint = snapshots[MODEL]
    vocab = {symbol: index for index, symbol in
             enumerate((checkpoint / "vocab.txt").read_text(encoding="utf-8").split("\n"))}
    n_tokens = len(vocab) - 1
    predictor = DurationPredictor(
        transformer=DurationTransformer(dim=512, depth=8, heads=8, text_dim=512,
                                        ff_mult=2, conv_layers=2, text_num_embeds=n_tokens),
        vocab_char_map=vocab,
    )
    predictor.load_weights(str(checkpoint / "duration_v2.safetensors"))
    vocoder = Vocos.from_pretrained(str(snapshots[VOCODER]))
    model = F5TTS(
        transformer=DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
                        conv_layers=4, text_num_embeds=n_tokens, text_mask_padding=True),
        duration_predictor=predictor, vocoder=vocoder.decode, vocab_char_map=vocab,
    )
    nn.quantize(model, bits=8, class_predicate=lambda _, layer:
                isinstance(layer, nn.Linear) and layer.weight.shape[1] % 64 == 0)
    model.load_weights(str(checkpoint / "model_v1_8b.safetensors"))
    model.eval()
    mx.eval(model.parameters())
    return model


def _sentence(text, name):
    value = " ".join(text.split())
    if not value:
        raise ValueError(f"{name} is empty.")
    return value if value[-1] in ".!?" else value + "."


def _waveform(raw, *, pad_to_hop):
    import mlx.core as mx
    import numpy as np

    audio = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms == 0:
        raise ValueError("The F5 input is silent.")
    if pad_to_hop:
        audio = np.pad(audio, (0, -len(audio) % HOP))
    return mx.array(audio * max(1, 0.1 / rms))[None, :]


def speak(args):
    if args.output.suffix.lower() != ".wav":
        raise ValueError("--output must be a new .wav file.")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be an unsigned 32-bit integer.")
    if args.duration is not None and not HOP / RATE <= args.duration < MAX_HOPS * HOP / RATE:
        raise ValueError("--duration must be at least one model frame and below 20.48 seconds.")
    raw = pcm_frames(args.reference, max_seconds=12)
    if len(raw) < 3 * RATE * 2:
        raise ValueError("Choose 3–12 seconds of clean speech for the F5 reference.")
    caption = _sentence(args.transcript.read_text(encoding="utf-8-sig"), "Reference transcript")
    line = _sentence(args.text if args.text is not None else
                     args.text_file.read_text(encoding="utf-8-sig"), "Text to synthesize")
    if len(caption) + len(line) + 1 >= MAX_HOPS:
        raise ValueError("The text is too long. Render one short sentence at a time.")
    with _new_file(args.output) as staging:
        snapshots = _snapshots(local_only=True)
        try:
            import numpy as np
            from f5_tts_mlx.utils import convert_char_to_pinyin, list_str_to_idx
        except ImportError as exc:
            raise RuntimeError("Install the F5 dependencies with python -m pip install -e '.[f5]'") from exc
        prompt = _waveform(raw, pad_to_hop=True)
        model = _load_model(snapshots)
        if args.adapter is not None:
            from .f5_adapter import apply_adapter
            apply_adapter(model, args.adapter)
        tokens = list_str_to_idx(convert_char_to_pinyin([caption + " " + line]), model._vocab_char_map)
        ref_hops = prompt.shape[1] // HOP
        if args.duration is None:
            predicted = int(model.predict_duration(model._mel_spec(prompt[0]), tokens).item())
            # Scale only the new phrase, leaving the reference duration fixed.
            new_hops = round(max(round(RATE / HOP), predicted - ref_hops) / PACE)
        else:
            new_hops = round(args.duration * RATE / HOP)
            if ref_hops + new_hops <= tokens.shape[1]:
                raise ValueError("The requested duration is too short for this text.")
        duration = max(ref_hops + new_hops, tokens.shape[1] + 1)
        if duration > MAX_HOPS:
            raise ValueError("The reference and line exceed the model context. Use a shorter line or reference.")
        wave_out, _ = model.sample(prompt, text=tokens, duration=duration, steps=8, method="rk4",
                                   cfg_strength=2.0, sway_sampling_coef=-1.0, seed=args.seed,
                                   max_duration=MAX_HOPS)
        generated = np.asarray(wave_out[prompt.shape[1]:], dtype="float32")
        if generated.ndim != 1 or not generated.size or not np.isfinite(generated).all():
            raise ValueError("F5 produced invalid audio.")
        peak = float(np.max(np.abs(generated)))
        if peak < 1e-6:
            raise ValueError("F5 produced a silent sample.")
        pcm = np.rint(generated * (32767 * min(1, 0.99 / peak))).astype("<i2")
        with wave.open(staging, "wb") as output:
            output.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            output.writeframes(pcm.tobytes())
    print(f"Saved {len(generated) / RATE:.2f}s of synthetic speech to {args.output} (F5 seed {args.seed}).")


def configure(commands):
    download = commands.add_parser("f5-fetch", help="Download pinned public F5 and Vocos weights")
    download.set_defaults(handler=fetch)
    render = commands.add_parser("f5-speak", help="Generate local F5 speech from an approved voice reference")
    render.add_argument("--reference", required=True, type=Path, help="3–12 seconds of mono 24 kHz PCM16 speech")
    render.add_argument("--transcript", required=True, type=Path, help="Reviewed text spoken in the reference")
    words = render.add_mutually_exclusive_group(required=True)
    words.add_argument("--text", help="One short line to synthesize")
    words.add_argument("--text-file", type=Path, help="UTF-8 file containing the line")
    render.add_argument("--output", required=True, type=Path, help="New 24 kHz mono PCM16 WAV")
    render.add_argument("--adapter", type=Path, help="Optional local directory produced by f5-fit")
    render.add_argument("--seed", type=int, default=391, help="Sampling seed (default: %(default)s)")
    render.add_argument("--duration", type=float, help="Set the new speech duration in seconds at generation time")
    render.set_defaults(handler=speak)
