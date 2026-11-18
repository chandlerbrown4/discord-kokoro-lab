"""Text-conditioned Kokoro adaptation on paired teacher WAVs. Heavy imports stay in the worker."""

import argparse
import hashlib
import json
import math
import os
import pickle
import random
from contextlib import nullcontext
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path

from .models import MODEL, _new_file, _validate_pack
from .reference import RATE, pcm_frames

REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
ENCODER_REVISION = "1c8bc7bf43125cfb2dbe61ba70dd105f21f73192"
FORMAT = "voice-lab-kokoro-adapter-v1"
FRAME = 600  # Kokoro's duration predictor operates at 40 frames per second.
CROP_FRAMES = 96
DEFAULT_STEPS = 96


@dataclass(frozen=True)
class Pair:
    path: Path
    text: str
    signature: str = ""


def read_manifest(manifest):
    manifest = Path(manifest).resolve(strict=True)
    splits = {"train": [], "validation": []}
    paths, texts, recordings = set(), {}, {}
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("split") not in splits:
            raise ValueError(f"Missing train or validation split on manifest line {number}.")
        text, audio, split = row.get("text"), row.get("audio"), row["split"]
        if not isinstance(text, str) or not text.strip() or not isinstance(audio, str) or not audio:
            raise ValueError(f"Expected paired text and WAV on manifest line {number}.")
        path = (manifest.parent / audio).resolve(strict=True)
        if Path(audio).is_absolute() or not path.is_relative_to(manifest.parent) or not path.is_file():
            raise ValueError(f"Audio must be a file inside the teacher corpus on line {number}.")
        key = "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        signature = digest.digest()
        if path in paths or texts.get(key, split) != split or recordings.get(signature, split) != split:
            raise ValueError(f"Repeated audio or text shared between splits on line {number}.")
        paths.add(path)
        texts[key] = split
        recordings[signature] = split
        splits[split].append(Pair(path, text.strip(), signature.hex()))
    if not all(splits.values()):
        raise ValueError("Provide at least one training pair and one distinct validation pair.")
    return splits


def _base_assets():
    from huggingface_hub import hf_hub_download

    return tuple(hf_hub_download(MODEL, filename, revision=REVISION)
                 for filename in ("config.json", "kokoro-v1_0.pth"))


def _encoder_asset():
    from inno_kokoro.enroll import fetch_weights

    # fetch_weights verifies the published SHA256, including cache hits.
    return fetch_weights(backend="torch", revision=ENCODER_REVISION)


def _base():
    import torch
    from kokoro import KModel, KPipeline

    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    config, weights = _base_assets()
    model = KModel(repo_id=MODEL, config=config, model=weights)
    model.requires_grad_(False).eval()
    frontend = KPipeline(lang_code="a", repo_id=MODEL, model=False)
    return model, frontend


def _parameters(model):
    # Fit the pretrained acoustic style-modulation biases and the pitch/rate heads.
    # Convolution kernels, text encoders and the waveform generator remain shared.
    return {name: value for name, value in model.named_parameters() if (
        name.startswith(("decoder.encode.", "decoder.decode.")) and
        name.endswith((".norm1.fc.bias", ".norm2.fc.bias"))
    ) or name in {"predictor.F0_proj.weight", "predictor.F0_proj.bias",
                  "predictor.duration_proj.linear_layer.bias"}}


def _pcm(pair):
    import numpy as np
    import torch

    raw = pcm_frames(pair.path)
    audio = torch.from_numpy(np.frombuffer(raw, dtype="<i2").astype("float32")) / 32768
    if audio.numel() < RATE // 2 or audio.abs().max() < 0.003:
        raise ValueError(f"Teacher audio is silent or too short: {pair.path.name}")
    return audio


def _tokens(frontend, text):
    rows = [r.phonemes for r in frontend(text) if r.phonemes]
    if len(rows) != 1:
        raise ValueError("Use one short spoken line per training WAV.")
    return rows[0]


def _embed(encoder, audio):
    import torch.nn.functional as F

    # Use the same differentiable 24 -> 16 kHz frontend for both signals.
    signal = F.interpolate(audio.reshape(1, 1, 1, -1), size=(1, audio.numel() * 2 // 3),
                           mode="bilinear", align_corners=False, antialias=True)
    return encoder(signal.reshape(1, -1))


class Features:
    def __init__(self):
        import numpy as np
        import torch

        edges = 700 * np.expm1(np.linspace(np.log1p(40/700), np.log1p(11000/700), 66))
        bins = np.fft.rfftfreq(1024, 1 / RATE)[None, :]
        low, mid, high = (edges[i:i+64, None] for i in range(3))
        weights = np.maximum(0.0, np.minimum((bins-low)/(mid-low), (high-bins)/(high-mid)))
        self.bank = torch.from_numpy((weights / weights.sum(1, keepdims=True)).astype("float32"))
        self.window = torch.hann_window(1024)

    def mel(self, audio):
        import torch

        spectrum = torch.stft(audio, 1024, hop_length=FRAME, window=self.window, return_complex=True)
        return (self.bank @ spectrum.abs()).clamp_min(0.025).log()


def _alignment(source, target, durations):
    """Map each teacher frame to a phoneme using monotone DTW on spectral envelopes."""
    import numpy as np
    import torch
    from scipy.spatial.distance import cdist

    def normalize(mel):
        x = mel.numpy().T
        return (x - x.mean(0)) / np.maximum(x.std(0), 0.4)

    tokens = np.repeat(np.arange(len(durations)), durations.numpy())
    cost = cdist(normalize(source)[:len(tokens)], normalize(target), metric="sqeuclidean") / source.shape[0]
    n, m = cost.shape
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0
    move = np.zeros((n, m), np.int8)
    for i in range(n):
        for j in range(m):
            options = (dp[i, j], dp[i, j+1] + 0.15, dp[i+1, j] + 0.15)
            k = min(range(3), key=options.__getitem__)
            dp[i+1, j+1] = cost[i, j] + options[k]
            move[i, j] = k
    i, j = n-1, m-1
    mapped = np.zeros(m, np.int64)
    while i >= 0 and j >= 0:
        mapped[j] = tokens[i]
        direction = move[i, j]
        i -= direction != 2
        j -= direction != 1
    return torch.from_numpy(mapped)


@dataclass
class Example:
    phonemes: str
    audio: object
    tokens: object
    pitch: float
    # Token indices aligned to target audio, populated once for training examples.
    alignment: object = None


def _example(pair, model, frontend):
    import torch
    from inno_kokoro.prosody import hz, stats

    phonemes = _tokens(frontend, pair.text)
    ids = [model.vocab[c] for c in phonemes if c in model.vocab]
    if not ids or len(ids)+2 > model.context_length:
        raise ValueError(f"Text does not fit Kokoro context: {pair.path.name}")
    audio = _pcm(pair)
    semitones, _, voiced = stats(audio.numpy(), RATE)
    pitch = float(hz(semitones)) if voiced > 0 else 0.
    return Example(phonemes, audio, torch.tensor([[0, *ids, 0]]), pitch)


def _training_loss(example, pack, delta, model, encoder, features, rng):
    import torch
    import torch.nn.functional as F

    style = pack[len(example.phonemes)-1] + delta
    ids = example.tokens
    mask = torch.zeros_like(ids, dtype=torch.bool)
    lengths = torch.tensor([ids.shape[-1]])
    with torch.no_grad():
        language = model.bert_encoder(model.bert(ids, attention_mask=(~mask).int())).transpose(-1, -2)
        text = model.text_encoder(ids, lengths, mask)
    predictor_style = style[:, 128:]
    encoded = model.predictor.text_encoder(language, predictor_style, lengths, mask)
    lstm, _ = model.predictor.lstm(encoded)
    duration = model.predictor.duration_proj(lstm).sigmoid().sum(-1).squeeze(0)
    indices = example.alignment
    target_duration = indices.bincount(minlength=ids.shape[-1]).clamp_min(1)
    duration_loss = F.smooth_l1_loss(duration.log(), target_duration.float().log())
    start = rng.randint(0, max(0, len(indices)-CROP_FRAMES))
    indices = indices[start:start+CROP_FRAMES]
    f0, noise = model.predictor.F0Ntrain(encoded.transpose(-1, -2)[:, :, indices], predictor_style)
    predicted = model.decoder(text[:, :, indices], f0, noise, style[:, :128]).reshape(-1)
    target = example.audio[start*FRAME:(start+len(indices))*FRAME]
    n = min(predicted.numel(), target.numel())
    predicted, target = predicted[:n], target[:n]
    speaker = 1 - (_embed(encoder, predicted) * _embed(encoder, target).detach()).sum()
    actual_mel, target_mel = features.mel(predicted), features.mel(target)
    # Pool a few frames to tolerate small errors in teacher-to-student phoneme alignment.
    pooled = lambda x: F.avg_pool1d(x.unsqueeze(0), kernel_size=4, stride=3)
    acoustic = F.l1_loss(pooled(actual_mel), pooled(target_mel))
    timbre = F.l1_loss(actual_mel.mean(-1), target_mel.mean(-1))
    voiced = f0[f0 > 50]
    pitch_loss = f0.new_zeros(())
    if example.pitch and voiced.numel() >= 8:
        pitch_loss = F.smooth_l1_loss(voiced.log().mean(), f0.new_tensor(math.log(example.pitch)))
    return speaker + 0.2 * acoustic + 0.2 * timbre + 0.25 * duration_loss + pitch_loss


def _validate(examples, pack, model, encoder, features):
    import torch

    scores, timbres, rates = [], [], []
    # Use unconstrained text-to-audio inference. Fixed noise makes checkpoints comparable.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(745)
        for ex in examples:
            wav = model(ex.phonemes, pack[len(ex.phonemes)-1])
            scores.append(float((_embed(encoder, wav) * _embed(encoder, ex.audio)).sum()))
            timbres.append(float((features.mel(wav).mean(-1) - features.mel(ex.audio).mean(-1)).abs().mean()))
            rates.append(abs(math.log(wav.numel()/ex.audio.numel())))
    mean = lambda xs: sum(xs) / len(xs)
    return {"speaker_cosine": mean(scores), "log_mel_distance": mean(timbres),
            "duration_log_error": mean(rates),
            "selection_score": mean(scores)-0.2*mean(timbres)-0.15*mean(rates)}


def _checkpoint_key(splits, steps):
    rows = [(split, pair.text, pair.signature) for split, pairs in splits.items() for pair in pairs]
    recipe = ["fit-v1", REVISION, ENCODER_REVISION, steps, rows]
    return hashlib.sha256(json.dumps(recipe, ensure_ascii=False).encode()).hexdigest()


def _save_progress(path, state):
    import torch
    # The worker holds the per-build lock. Reuse this name after a hard stop.
    temp = path.with_name(path.name + ".next")
    try:
        with temp.open("wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _load_progress(path, key):
    import torch
    if path.is_file():
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and state.get("key") == key:
                return state
        except (EOFError, OSError, RuntimeError, pickle.UnpicklingError):
            pass
    return {"key": key, "alignments": []}


def train(manifest, output, *, steps=DEFAULT_STEPS, progress=None, resume=False):
    """Fit an English speaker profile. Returns the path to the new profile.pt."""
    import torch
    from inno_kokoro.enroll import Tuner, enroll

    if not isinstance(steps, int) or steps < 1:
        raise ValueError("Training requires a positive number of optimizer steps.")
    output = Path(output) / "profile.pt"
    splits = read_manifest(manifest)
    log = progress or (lambda message: None)
    started = time.monotonic()
    # This publication boundary prevents incomplete or competing runs replacing a valid profile.
    checkpoint = output.with_name("fit-progress.pt")
    guard = nullcontext()
    if resume:
        from filelock import FileLock
        output.parent.mkdir(parents=True, exist_ok=True)
        guard = FileLock(str(checkpoint) + ".lock", timeout=0)
    with guard, _new_file(output) as staging:
        state = _load_progress(checkpoint, _checkpoint_key(splits, steps)) if resume else {"alignments": []}
        previous_time = state.get("active_seconds", 0.)
        def save():
            if resume:
                state["active_seconds"] = previous_time + time.monotonic() - started
                _save_progress(checkpoint, state)
        log("Loading the shared Kokoro model and training-only speaker encoder.")
        # Some base-model buffers are initialized at load. Reopening a worker
        # must restore the same base before applying saved optimizer parameters.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(744)
            model, frontend = _base()
            tuner = Tuner(path=_encoder_asset(), backend="torch", check_update=False)
            features = Features()
        training = [_example(p, model, frontend) for p in splits["train"]]
        validation = [_example(p, model, frontend) for p in splits["validation"]]
        reference = torch.cat([ex.audio for ex in training])[:30*RATE]
        if reference.numel() < 3*RATE:
            raise ValueError("At least three seconds of paired teacher training audio are required.")
        if "pack" not in state:
            pack, _ = enroll(reference.numpy(), RATE, tuner, return_pt=True)
            state["pack"] = _validate_pack(pack, torch)
        pack = state["pack"]
        params = _parameters(model)
        original = {name: p.detach().clone() for name, p in params.items()}
        delta = torch.nn.Parameter(torch.zeros((1, 256)))
        log(f"Aligning {len(training)} teacher recordings to their transcripts.")
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(746)
            completed = state["alignments"]
            for ex, alignment in zip(training, completed):
                ex.alignment = alignment
            if completed:
                torch.set_rng_state(state["alignment_rng"])
            for index in range(len(completed), len(training)):
                ex = training[index]
                result = model(ex.phonemes, pack[len(ex.phonemes)-1], return_output=True)
                frames = ex.audio.numel() // FRAME
                if frames < 4:
                    raise ValueError("The teacher line is too short for alignment.")
                ex.alignment = _alignment(features.mel(result.audio), features.mel(ex.audio)[:, :frames],
                                          result.pred_dur)
                completed.append(ex.alignment)
                if resume and (index % 8 == 0 or index + 1 == len(training)):
                    state["alignment_rng"] = torch.get_rng_state()
                    save()
        if "baseline" not in state:
            state["baseline"] = _validate(validation, pack, model, tuner.encoder, features)
        baseline = state["baseline"]
        for parameter in params.values():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam([{"params": [delta], "lr": 0.008},
                                      {"params": list(params.values()), "lr": 0.0015}])
        history, best = state.get("history", []), state.get("best")
        best_score = state.get("best_score", -math.inf)
        rng = random.Random(7531)
        order = state.get("order", [])
        restored = state.get("step", 0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(747)
            if restored:
                with torch.no_grad():
                    delta.copy_(state["delta"])
                    for name, param in params.items():
                        param.copy_(state["params"][name])
                optimizer.load_state_dict(state["optimizer"])
                rng.setstate(state["random"])
                torch.set_rng_state(state["torch_rng"])
                log(f"Resuming Kokoro after step {restored}/{steps}.")
            for step in range(restored+1, steps+1):
                if not order:
                    order = list(range(len(training)))
                    rng.shuffle(order)
                loss = _training_loss(training[order.pop()], pack, delta, model, tuner.encoder, features, rng)
                if not torch.isfinite(loss).item():
                    raise ValueError("Kokoro training produced a non-finite loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([delta, *params.values()], 2.0, error_if_nonfinite=True)
                optimizer.step()
                if step == steps or step % max(1, math.ceil(steps/4)) == 0:
                    fitted = (pack + delta).detach()
                    metrics = _validate(validation, fitted, model, tuner.encoder, features)
                    history.append({"step": step, "training_loss": float(loss.detach()), **metrics})
                    if metrics["selection_score"] > best_score:
                        best_score = metrics["selection_score"]
                        best = {"style": fitted.clone(), "weights": {name: (p.detach()-original[name]).clone()
                                for name, p in params.items()}, "step": step}
                    message = f"Kokoro step {step}/{steps}, validation speaker cosine {metrics['speaker_cosine']:.3f}."
                elif step == 1 or step % 8 == 0:
                    message = f"Fitting Kokoro, step {step}/{steps}."
                else:
                    continue
                if resume:
                    state.update(step=step, params={name: param.detach() for name, param in params.items()},
                                 delta=delta.detach(), optimizer=optimizer.state_dict(), order=order,
                                 random=rng.getstate(), torch_rng=torch.get_rng_state(),
                                 history=history, best=best, best_score=best_score)
                    save()
                log(message)
        report = {"train_pairs": len(training), "validation_pairs": len(validation), "steps": steps,
                  "initial": baseline, "validation": history, "training_seconds": previous_time + time.monotonic()-started,
                  "updated_acoustic_parameters": sum(v.numel() for v in params.values())}
        with open(staging, "wb") as stream:
            torch.save({"format": FORMAT, "revision": REVISION, "report": report, **best}, stream)
    if resume:
        checkpoint.unlink(missing_ok=True)
    return output


class Synthesizer:
    """Warm text-to-speech runtime requiring only Kokoro and one local profile."""

    def __init__(self, profile_path):
        import torch

        profile_path = Path(profile_path)
        if profile_path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Kokoro adapter exceeds the supported profile size.")
        state = torch.load(profile_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("format") != FORMAT or state.get("revision") != REVISION:
            raise ValueError("Unsupported Kokoro adapter format or base revision.")
        self.pack = _validate_pack(state["style"], torch)
        self.model, self.frontend = _base()
        params = _parameters(self.model)
        weights = state.get("weights")
        if not isinstance(weights, dict) or weights.keys() != params.keys():
            raise ValueError("Kokoro adapter does not match the model's trainable layers.")
        for name, target in params.items():
            value = weights[name]
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                    or value.dtype != target.dtype or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid Kokoro adapter tensor: {name}")
            target.add_(value)

    def render(self, text, output_path):
        import torch

        if not text.strip():
            raise ValueError("Enter some text for Kokoro to read.")
        output_path = Path(output_path)
        with _new_file(output_path) as staging, torch.inference_mode(), wave.open(staging, "wb") as wav:
            wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            total = 0
            for result in self.frontend(text):
                if not result.phonemes:
                    continue
                audio = self.model(result.phonemes, self.pack[len(result.phonemes)-1])
                if not audio.numel() or not torch.isfinite(audio).all():
                    raise ValueError("Kokoro produced invalid audio.")
                if total:
                    wav.writeframesraw(b"\0\0" * (RATE//8))
                wav.writeframesraw(audio.clamp(-1, 1).mul(32767).round().numpy().astype("<i2").tobytes())
                total += audio.numel()
            if not total:
                raise ValueError("Kokoro could not pronounce the supplied text.")
        return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("manifest", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    speak = commands.add_parser("speak")
    speak.add_argument("profile", type=Path)
    speak.add_argument("output", type=Path)
    speak.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "train":
        print(train(args.manifest, args.output, steps=args.steps, progress=lambda m: print(m, flush=True)))
    else:
        print(Synthesizer(args.profile).render(args.text, args.output))


if __name__ == "__main__":
    main()
