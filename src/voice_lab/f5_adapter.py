"""Fit and restore the small attention adapter for the pinned MLX F5 model."""

import hashlib
import json
import math
from pathlib import Path

from . import f5
from .reference import _entries, pcm_frames
from .tracks import new_directory


RANK = 8
BLOCKS = 6
PROJECTIONS = ("to_q", "to_k", "to_v")
LEARNING_RATE = 5e-5


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _examples(speaker_dir, clips, references):
    if len(clips) < 4 or len(set(clips)) != len(clips):
        raise ValueError("Select at least four different reviewed training clips.")
    directory = speaker_dir.expanduser().resolve(strict=True)
    manifest = _entries(directory)
    prompts = []
    for reference in references:
        profile = reference.expanduser()
        pcm = pcm_frames(profile / "reference.wav", max_seconds=12)
        if len(pcm) < 3 * f5.RATE * 2:
            raise ValueError("Each reference must contain 3–12 seconds of reviewed speech.")
        text = f5._sentence((profile / "transcript.txt").read_text(encoding="utf-8-sig"), "Reference transcript")
        if any(pcm == other for other, _ in prompts):
            raise ValueError("Choose each conditioning reference only once.")
        prompts.append((pcm, text))

    examples, seen = [], set()
    for name in clips:
        if name not in manifest:
            raise ValueError(f"Clip is absent from the selected speaker manifest: {name}")
        path = (directory / name).resolve(strict=True)
        if not path.is_relative_to(directory):
            raise ValueError(f"Clip is outside the selected speaker directory: {name}")
        text = manifest[name].get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Review and enter the transcript in manifest.jsonl for {name}.")
        pcm = pcm_frames(path)
        digest = _digest(pcm)
        if digest in seen or any(pcm in prompt or prompt in pcm for prompt, _ in prompts):
            raise ValueError(f"Use distinct audio for conditioning and each training take: {name}")
        seen.add(digest)
        examples.append((name, pcm, f5._sentence(text, name)))
    return prompts, examples


def _attach(model):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    class LowRankLinear(nn.Module):
        def __init__(self, linear):
            super().__init__()
            self.linear = linear
            self.linear.freeze()
            outputs, inputs = linear.weight.shape
            if isinstance(linear, nn.QuantizedLinear):
                inputs = inputs * 32 // linear.bits
            self.down = mx.random.normal((inputs, RANK)) * 0.02
            self.up = mx.zeros((RANK, outputs))

        def __call__(self, x):
            delta = (x @ self.down.astype(x.dtype)) @ self.up.astype(x.dtype)
            return self.linear(x) + delta

    blocks = model.transformer.transformer_blocks[-BLOCKS:]
    if len(blocks) != BLOCKS or any(not isinstance(block.attn[name], (nn.QuantizedLinear, nn.Linear))
                                    for block in blocks for name in PROJECTIONS):
        raise ValueError("Expected the unmodified attention layers of the pinned F5 model.")
    model.freeze()
    for block in blocks:
        for name in PROJECTIONS:
            block.attn[name] = LowRankLinear(block.attn[name])
    weights = dict(tree_flatten(model.trainable_parameters()))
    if len(weights) != BLOCKS * len(PROJECTIONS) * 2:
        raise ValueError("Unexpected trainable parameter layout in the F5 model.")
    return weights


def _layout():
    return {"format_version": 1, "model": f5.MODEL, "revision": f5.WEIGHTS[f5.MODEL][0],
            "quantization_bits": 8, "rank": RANK, "last_blocks": BLOCKS, "projections": list(PROJECTIONS)}


def apply_adapter(model, path):
    folder = Path(path).expanduser()
    metadata = json.loads((folder / "adapter.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in _layout().items()):
        raise ValueError("Adapter bundle is incompatible with the pinned F5 model or adapter format.")
    checkpoint = folder / "adapter.safetensors"
    if _digest(checkpoint.read_bytes()) != metadata.get("weights_sha256"):
        raise ValueError("Adapter weights failed their recorded SHA-256 checksum.")
    import mlx.core as mx

    loaded = mx.load(str(checkpoint))
    expected = _attach(model)
    if loaded.keys() != expected.keys() or any(loaded[k].shape != v.shape or loaded[k].dtype != v.dtype
                                                for k, v in expected.items()):
        raise ValueError("Adapter tensor names, shapes or dtypes do not match the F5 model.")
    # strict=True would require every frozen base tensor in the checkpoint.
    model.load_weights(list(loaded.items()), strict=False)


def _fit(model, prompts, examples, steps, seed):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    import numpy as np
    from f5_tts_mlx.utils import convert_char_to_pinyin, list_str_to_idx

    mx.random.seed(seed)
    _attach(model)
    conditions = [(f5._waveform(raw, pad_to_hop=True), caption) for raw, caption in prompts]
    batch = []
    for index, (name, raw, text) in enumerate(examples):
        prompt, caption = conditions[index % len(conditions)]
        target = f5._waveform(raw, pad_to_hop=False)
        prefix = prompt.shape[1] // f5.HOP
        samples = mx.concatenate((prompt, target), axis=1)
        mel = model._mel_spec(samples[0])
        tokens = list_str_to_idx(convert_char_to_pinyin([caption + " " + text]), model._vocab_char_map)
        if not prefix < mel.shape[1] <= f5.MAX_HOPS or tokens.shape[1] >= mel.shape[1]:
            raise ValueError(f"Reference and clip text/audio do not fit the F5 context: {name}")
        cond = mx.concatenate((mel[:, :prefix], mx.zeros_like(mel[:, prefix:])), axis=1)
        mx.eval(mel, cond, tokens)
        batch.append((mel, cond, tokens, prefix))

    def objective(model, mel, cond, tokens, prefix, noise, time):
        flow = mel - noise
        mixed = noise + time[:, None, None] * flow
        pred = model.transformer(mixed, cond, tokens, time, drop_audio_cond=False, drop_text=False)
        return nn.losses.mse_loss(pred[:, prefix:], flow[:, prefix:])

    rng = np.random.default_rng(seed)
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)
    loss_grad = nn.value_and_grad(model, objective)
    losses = []
    for step in range(steps):
        offset = step % len(batch)
        if offset == 0:
            order = rng.permutation(len(batch))
        item = batch[order[offset]]
        loss, grads = loss_grad(model, *item, mx.random.normal(item[0].shape), mx.random.uniform(.04, .96, (1,)))
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state, loss)
        value = float(loss.item())
        if not math.isfinite(value):
            raise ValueError(f"Non-finite training loss at step {step + 1}.")
        losses.append(value)
        if step == 0 or (step + 1) % 6 == 0 or step + 1 == steps:
            print(f"F5 fit {step + 1}/{steps}  loss {value:.4f}", flush=True)
    return dict(tree_flatten(model.trainable_parameters())), losses


def run(args):
    if args.steps < 1 or not 0 <= args.seed < 2**32:
        raise ValueError("Use a positive step count and an unsigned 32-bit seed.")
    prompts, examples = _examples(args.speaker_dir, args.clip, args.reference)
    with new_directory(args.output.expanduser()) as staging:
        snapshots = f5._snapshots(local_only=True)
        try:
            import mlx.core as mx
            model = f5._load_model(snapshots)
        except ImportError as exc:
            raise RuntimeError("Install F5 dependencies with python -m pip install -e '.[f5]'") from exc
        mx.set_cache_limit(512 * 1024 * 1024)
        weights, losses = _fit(model, prompts, examples, args.steps, args.seed)
        checkpoint = staging / "adapter.safetensors"
        mx.save_safetensors(str(checkpoint), weights)

        def inputs(raw, text):
            return {"pcm_sha256": _digest(raw), "normalized_text_sha256": _digest(text.encode("utf-8"))}

        metadata = dict(_layout(), weights_sha256=_digest(checkpoint.read_bytes()), seed=args.seed,
                        steps=args.steps, learning_rate=LEARNING_RATE, loss=losses,
                        references=[inputs(raw, text) for raw, text in prompts],
                        training=[dict(clip=name, **inputs(raw, text)) for name, raw, text in examples])
        (staging / "adapter.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved the fitted F5 adapter to {args.output}.")


def configure(commands):
    fit = commands.add_parser("f5-fit", help="Fit a local F5 adapter from approved, reviewed speaker takes")
    fit.add_argument("speaker_dir", type=Path, help="Prepared numeric speaker directory with reviewed manifest.jsonl")
    fit.add_argument("output", type=Path, help="New adapter bundle directory")
    fit.add_argument("--reference", type=Path, action="append", required=True,
                     help="Reference bundle directory with separate 3–12 s speech and transcript; repeat to alternate")
    fit.add_argument("--clip", required=True, action="append", metavar="clips/NAME.wav",
                     help="Reviewed clip in the speaker manifest; select at least four distinct takes")
    fit.add_argument("--steps", type=int, default=24)
    fit.add_argument("--seed", type=int, default=1701)
    fit.set_defaults(handler=run)
