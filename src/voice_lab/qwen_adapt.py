"""Fit local low-rank updates to Qwen's residual acoustic-code predictor.

The shared 8-bit teacher and its primary speech-code generator stay frozen.
MLX is imported only when an optional teacher training worker needs it.
"""
from functools import lru_cache
from pathlib import Path
import math
import os
from tempfile import TemporaryDirectory

from .distill_teacher import QWEN_REVISION, RATE, _read

FORMAT = "voice-lab-qwen-detail-v1"
RANK = 8
STEPS = 12
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


@lru_cache(maxsize=1)
def _adapter_class():
    import mlx.core as mx
    import mlx.nn as nn

    class AdaptedLinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            base.freeze()
            width = base.weight.shape[1]
            if isinstance(base, nn.QuantizedLinear):
                width *= 32 // base.bits
            self.down = .015 * mx.random.normal((width, RANK))
            self.up = mx.zeros((RANK, base.weight.shape[0]))

        def __call__(self, x):
            return self.base(x) + ((x @ self.down.astype(x.dtype)) @ self.up.astype(x.dtype))

    return AdaptedLinear


def _layers(teacher):
    for block in teacher.model.talker.code_predictor.model.layers:
        for key in PROJECTIONS:
            yield block.self_attn, key


def remove_adapter(teacher):
    cls = _adapter_class()
    for attn, key in _layers(teacher):
        if isinstance(attn[key], cls):
            attn[key] = attn[key].base


def _install(teacher):
    cls = _adapter_class()
    layers = list(_layers(teacher))
    if any(isinstance(attn[key], cls) for attn, key in layers):
        raise ValueError("A Qwen adapter is already installed.")
    replacements = [cls(attn[key]) for attn, key in layers]
    teacher.model.freeze()
    for (attn, key), layer in zip(layers, replacements):
        attn[key] = layer


def _parameters(teacher):
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(teacher.model.talker.code_predictor.trainable_parameters()))
    if len(params) != sum(2 for _ in _layers(teacher)) or any(
            not k.endswith((".up", ".down")) for k in params):
        raise ValueError("Unexpected trainable parameters in Qwen's acoustic predictor.")
    return params


def apply_adapter(teacher, path):
    """Apply a complete, validated local adapter to the caller's teacher."""
    import mlx.core as mx

    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Unexpected Qwen adapter size.")
    weights, metadata = mx.load(str(path), return_metadata=True)
    if metadata.get("format") != FORMAT or metadata.get("base_revision") != QWEN_REVISION:
        raise ValueError("This adapter needs a different Qwen checkpoint or format.")
    _install(teacher)
    try:
        expected = _parameters(teacher)
        if expected.keys() != weights.keys() or any(
                tensor.shape != expected[k].shape or tensor.dtype != expected[k].dtype
                for k, tensor in weights.items()):
            raise ValueError("Incomplete or incompatible Qwen acoustic adapter.")
        if not bool(mx.stack([mx.all(mx.isfinite(v)) for v in weights.values()]).all().item()):
            raise ValueError("Invalid Qwen adapter weights.")
        teacher.model.talker.code_predictor.load_weights(list(weights.items()), strict=False)
    except BaseException:
        remove_adapter(teacher)
        raise


def _example(teacher, path, text):
    import mlx.core as mx
    import soundfile as sf

    model, talker = teacher.model, teacher.model.talker
    if not 0.5 <= sf.info(path).duration <= 20 or not text.strip():
        raise ValueError("Choose transcribed speech excerpts between half a second and twenty seconds.")
    audio = mx.array(_read(path, RATE))
    codes = model.speech_tokenizer.encode(audio[None, None, :])
    prefix, trailing, pad, _ = model._prepare_icl_generation_inputs(text, teacher.audio, teacher.transcript, language="English")
    tokens = [codes[:, i, :] for i in range(codes.shape[1])]
    frames = codes.shape[2]
    # Reproduce the runtime's text/codec inputs at each forced target frame.
    trailing_length = min(trailing.shape[1], frames)
    conditioning = mx.concatenate((trailing[:, :trailing_length], mx.broadcast_to(
        pad, (1, frames - trailing_length, pad.shape[-1]))), axis=1)
    inputs = mx.concatenate((prefix, conditioning + model._codec_embeds_for_tokens(tokens)), axis=1)
    _, hidden = talker(inputs)
    context = hidden[0, prefix.shape[1] - 1:-1, None, :]
    ids = codes[0].T
    embeddings = [talker.get_input_embeddings()(ids[:, :1])]
    embeddings.extend(layer(ids[:, i:i+1]) for i, layer in enumerate(
        talker.code_predictor.codec_embedding[:-1], 1))
    features = mx.concatenate([context, *embeddings], axis=1)
    targets = ids[:, 1:]
    mx.eval(features, targets)
    return features, targets


def fit_adapter(teacher, pairs, output, *, steps=STEPS, progress=None):
    """Fit on genuine (WAV path, transcript) pairs and leave the teacher at its base weights."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np

    pairs = list(pairs)
    if not pairs or type(steps) is not int or steps < 1:
        raise ValueError("Qwen training needs genuine speech pairs and a positive step count.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.random.seed(1701)
    examples = [_example(teacher, Path(wav), text) for wav, text in pairs]

    def loss_fn(predictor, inputs, targets):
        if predictor.small_to_mtp_projection is not None:
            inputs = predictor.small_to_mtp_projection(inputs)
        hidden = predictor.model(inputs)
        logits = mx.stack([head(hidden[:, i]) for i, head in enumerate(predictor.lm_head, 1)], axis=1)
        return nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="mean")

    _install(teacher)
    try:
        predictor = teacher.model.talker.code_predictor
        _parameters(teacher)
        opt = optim.AdamW(learning_rate=.00015)
        loss_grad = nn.value_and_grad(predictor, loss_fn)
        rng = np.random.default_rng(1701)
        losses, order = [], []
        for step in range(1, steps+1):
            if not order:
                order = list(rng.permutation(len(examples)))
            loss, grads = loss_grad(predictor, *examples[order.pop()])
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(predictor, grads)
            mx.eval(predictor.trainable_parameters(), opt.state, loss)
            value = float(loss.item())
            if not np.isfinite(value):
                raise ValueError("Qwen training produced a non-finite acoustic loss.")
            losses.append(value)
            if progress and (step == 1 or step % 8 == 0 or step == steps):
                progress(f"Fitting the Qwen teacher on original speech, update {step}/{steps}")
        weights = _parameters(teacher)
        if not any(float(mx.max(mx.abs(w)).item()) > 0 for k, w in weights.items() if k.endswith(".up")):
            raise ValueError("The Qwen adapter has no learned acoustic update.")
        with TemporaryDirectory(dir=output.parent) as tmp:
            staged = Path(tmp) / "adapter.safetensors"
            mx.save_safetensors(str(staged), weights, metadata={"format": FORMAT, "base_revision": QWEN_REVISION})
            staged.replace(output)
        return {"steps": steps, "fit_clips": len(examples), "losses": losses,
                "trainable_parameters": sum(v.size for v in weights.values())}
    finally:
        remove_adapter(teacher)


def _pitch_span(audio):
    """Robust central pitch range in semitones, across periodic voiced frames."""
    import numpy as np
    from scipy.signal import correlate, resample_poly

    audio = resample_poly(audio, 1, 3)
    f0 = []
    for offset in range(0, len(audio) - 320, 160):
        frame = audio[offset:offset + 320]
        frame = frame - frame.mean()
        if np.mean(frame ** 2) < .00005:
            continue
        acf = correlate(frame, frame, mode="full", method="fft")[319:]
        # 60–400 Hz at 8 kHz. Only accept periodic frames (ignore fricatives).
        lag = 20 + int(np.argmax(acf[20:134]))
        if acf[lag] > .45 * acf[0]:
            f0.append(8000 / lag)
    if len(f0) < 5:
        return None
    low, high = np.percentile(f0, [15, 85])
    return float(12 * np.log2(high / low))


def _screen_development(teacher, records, checker, destination, progress):
    import numpy as np
    from .distill_teacher import _wav, word_error_rate

    destination.mkdir(exist_ok=True)
    rows = []
    for index, (path, text, digest) in enumerate(records):
        if progress:
            progress(f"Checking the Qwen teacher on original line {index+1}/{len(records)}")
        audio = teacher.render(text, int(digest[:8], 16) % (2**31))
        if len(audio) < RATE // 2 or not np.isfinite(audio).all() or np.max(abs(audio)) < .005:
            raise ValueError("Qwen produced an inaudible development line.")
        wav = destination / f"line-{index+1}.wav"
        _wav(wav, audio / max(1., np.max(abs(audio))/.98))
        original = _read(path, RATE)
        actual_pitch, new_pitch = _pitch_span(original), _pitch_span(audio)
        heard = checker.transcribe(wav)
        rows.append({"original_sha256": digest, "text": text, "heard": heard,
                     "similarity": float(checker.similarity(path, wav)), "wer": word_error_rate(text, heard),
                     "duration_error": abs(len(audio) / len(original) - 1),
                     "pitch_range_error": abs(actual_pitch - new_pitch) if None not in (new_pitch, actual_pitch) else None})
    return rows


def _promote(base, candidate):
    """A candidate must improve identity while keeping words, pitch range and pacing stable."""
    from statistics import mean
    fields = ("similarity", "wer", "duration_error")
    if any(not math.isfinite(r[k]) for r in [*base, *candidate] for k in fields):
        return False
    if any(r["pitch_range_error"] is not None and not math.isfinite(r["pitch_range_error"])
           for r in [*base, *candidate]):
        return False
    if len(base) < 3 or [r["original_sha256"] for r in base] != [r["original_sha256"] for r in candidate]:
        return False
    changes = [n["similarity"] - o["similarity"] for o, n in zip(base, candidate)]
    if mean(changes) < .005 or min(changes) < -.04 or sum(d > 0 for d in changes) < (len(changes)+1)//2:
        return False
    if mean(r["wer"] for r in candidate) > min(.20, mean(r["wer"] for r in base) + .01):
        return False
    if mean(r["duration_error"] for r in candidate) > mean(r["duration_error"] for r in base) + .08:
        return False
    pitch_delta = [c["pitch_range_error"] - b["pitch_range_error"] for b, c in zip(base, candidate)
                   if b["pitch_range_error"] is not None and c["pitch_range_error"] is not None]
    return not pitch_delta or mean(pitch_delta) <= .75


def _partition(records):
    """Hold out quiet and expressive deliveries before fitting either reference or adapter."""
    import numpy as np
    from .distill_teacher import _words

    unique, seen = [], set()
    for path, text, digest, span in sorted(records, key=lambda x: x[2]):
        words = tuple(_words(text))
        if len(words) < 2 or words in seen:
            continue
        seen.add(words)
        unique.append((span if span is not None else -1., path, text, digest))
    if len(unique) < 9:
        return [], []
    unique.sort(key=lambda r: (r[0], r[3]))
    count = 3 if len(unique) < 16 else 4
    dev_indices = set(np.rint(np.linspace(.15, .85, count) * (len(unique)-1)).astype(int))
    fit = [r[1:] for i, r in enumerate(unique) if i not in dev_indices]
    dev = [r[1:] for i, r in enumerate(unique) if i in dev_indices]
    if len(fit) > 32:
        # Bound cached codec features and training cost while retaining delivery coverage.
        fit = [fit[i] for i in np.rint(np.linspace(0, len(fit)-1, 32)).astype(int)]
    return fit, dev


def _representatives(fit, checker):
    """Use central identities and cover low, typical and wide pitch movement."""
    import numpy as np

    embeddings = np.stack([checker.embedding(p) for p, _, _ in fit])
    center = np.median(embeddings, axis=0)
    center /= max(np.linalg.norm(center), 1e-7)
    scores = embeddings @ center
    candidates = []
    for idx, (path, _, _) in enumerate(fit):
        if scores[idx] < np.quantile(scores, .3):
            continue
        audio = _read(path, RATE)
        span = _pitch_span(audio)
        if span is not None:
            candidates.append((idx, span, len(audio) / RATE))
    if len(candidates) < 3:
        return []
    quartiles = np.quantile([c[1] for c in candidates], [.5, .2, .8])
    chosen = []
    for desired in quartiles:
        i, _, _ = min((c for c in candidates if c[0] not in chosen),
                      key=lambda c: (.035 * abs(c[1]-desired) - scores[c[0]] - .01*min(c[2], 5), c[0]))
        chosen.append(i)
    return [fit[i][0] for i in chosen]


def _read_excerpts(sources, checker, cache):
    import json
    from .distill_teacher import _json

    for digest, path in sources.items():
        entry = cache / f"{digest}.json"
        if entry.is_file():
            try:
                metadata = json.loads(entry.read_text())
                text, span = metadata["text"], metadata["pitch_span"]
                if isinstance(text, str) and (span is None or math.isfinite(span)):
                    yield path, text, digest, span
                    continue
            except (ValueError, TypeError, KeyError):
                pass
        text = checker.transcribe(path).strip()
        span = _pitch_span(_read(path, RATE))
        _json(entry, {"text": text, "pitch_span": span})
        yield path, text, digest, span


def choose_teacher(train, holdouts, directory, base_reference, base_text, checker, model_dir,
                   progress=None, *, excerpt_cache=None):
    """Try one acoustic adapter; genuine release holdouts never enter teacher selection.

    Returns reference WAV, transcript, optional accepted adapter and a decision summary.
    A lower scoring challenger is persisted for audit and never used in generation.
    """
    import hashlib
    import json
    import soundfile as sf
    from .distill_teacher import (_audio_digest, _disjoint, _hash_file, _json, _locked,
                                 _model_key, _Qwen, ASR_REVISION, prepare_reference)

    paths = list(dict.fromkeys(map(Path, train)))
    _disjoint(paths, holdouts)
    directory = Path(directory)
    refused = (base_reference, base_text, None)
    if len(paths) < 9:
        return (*refused, {"selected": "base", "reason": "Fewer than nine training excerpts."})
    # Hash decoded samples so copies rewrapped in another WAV are one training example.
    sources = {}
    for path in paths:
        if .5 <= sf.info(path).duration <= 20:
            sources.setdefault(_audio_digest(path), path)
    if len(sources) < 9:
        return (*refused, {"selected": "base", "reason": "Fewer than nine distinct training excerpts."})
    embedding_path = checker.embedding_model
    asr_path = checker.asr_dir or os.environ.get("VOICE_LAB_WHISPER_MODEL")
    config = {"policy": "delivery-reference-v2", "format": FORMAT, "revision": QWEN_REVISION, "steps": STEPS, "rank": RANK,
              "training_audio": sorted(sources), "whisper_revision": ASR_REVISION,
              "asr_files": _model_key(Path(asr_path), ["model.bin"]) if asr_path else [],
              "model_files": _model_key(model_dir, ["model.safetensors", "speech_tokenizer/model.safetensors"]),
              "embedding": _hash_file(embedding_path) if embedding_path else None}
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    stage = directory / identity
    decision, adapter = stage / "decision.json", stage / "adapter.safetensors"
    with _locked(stage):
        if decision.exists():
            report = json.loads(decision.read_text())
            if report["selected"] == "base":
                return (*refused, report)
            if report.get("reference_dir") not in {"delivery-reference", "reference"}:
                raise ValueError("The Qwen teacher decision names an unknown reference.")
            ref_dir = stage / report["reference_dir"]
            ref, refmeta = ref_dir / "teacher-reference.wav", ref_dir / "teacher-reference.json"
            saved_adapter = adapter if report["selected"] == "adapter" else None
            if ref.is_file() and refmeta.is_file() and (saved_adapter is None or
                    adapter.is_file() and _hash_file(adapter) == report["adapter_sha256"]):
                metadata = json.loads(refmeta.read_text())
                if _hash_file(ref) == report["reference_sha256"] and metadata["text"] == report["reference_text"]:
                    return ref, metadata["text"], saved_adapter, report
        # A larger corpus only needs ASR and pitch analysis for newly arrived speech.
        asr_key = hashlib.sha256(json.dumps([ASR_REVISION, config["asr_files"], "pitch-v1"]).encode()).hexdigest()
        records = _read_excerpts(sources, checker, (excerpt_cache or directory / "excerpts") / asr_key)
        fit, dev = _partition(records)
        if not fit:
            report = {"selected": "base", "reason": "Fewer than nine distinct transcribable lines."}
            _json(decision, report)
            return (*refused, report)
        dev_files = [r[0] for r in dev]
        ref, text = prepare_reference([r[0] for r in fit], stage / "reference",
                                      holdouts=[*holdouts, *dev_files], checker=checker, progress=progress)
        teacher = _Qwen(model_dir, ref, text)
        before = _screen_development(teacher, dev, checker, stage / "baseline", progress)
        incumbent = before
        selected = "base"
        ref_dir = "reference"
        representatives = _representatives(fit, checker)
        reference_scores = []
        if representatives:
            ref_dir = "delivery-reference"
            ref, text = prepare_reference(representatives, stage / ref_dir,
                                          holdouts=[*holdouts, *dev_files], checker=checker, progress=progress)
            teacher.set_reference(ref, text)
            reference_scores = _screen_development(teacher, dev, checker, stage / "delivery-baseline", progress)
            if _promote(before, reference_scores):
                incumbent = reference_scores
                selected = "reference"
        training = fit_adapter(teacher, [(p, t) for p, t, _ in fit], adapter,
                               steps=max(STEPS, len(fit)), progress=progress)
        apply_adapter(teacher, adapter)
        after = _screen_development(teacher, dev, checker, stage / "candidate", progress)
        if _promote(before, after) and (selected == "base" or _promote(incumbent, after)):
            selected = "adapter"
        messages = {"adapter": "The fitted Qwen teacher improved on separate original speech.",
                    "reference": "The selected reference improved the Qwen teacher on separate original speech.",
                    "base": "Kept the reference-conditioned Qwen teacher after checking separate speech."}
        report = {"selected": selected, "reason": messages[selected], "reference_dir": ref_dir,
                  "reference_sha256": _hash_file(ref), "reference_text": text, "adapter_sha256": _hash_file(adapter),
                  "fit_sha256": [r[2] for r in fit], "development_sha256": [r[2] for r in dev],
                  "before": before, "revised_reference": reference_scores, "after": after, "training": training}
        _json(decision, report)
        if selected == "base":
            return (*refused, report)
        return ref, text, adapter if selected == "adapter" else None, report
