"""Fit an OpenVoice speaker condition with frozen weights and check unseen turns."""

from . import vc


def _condition(engine, audio):
    spec = engine._spec(audio).transpose(1, 2).to(engine.device)
    return engine.model.ref_enc(spec).unsqueeze(-1)


def _fit_condition(engine, clips):
    """Distill per-utterance flow conditioning into one shared 256-parameter speaker vector."""
    import numpy as np
    import torch

    model = engine.model.requires_grad_(False)
    pairs, conditions = [], []
    with torch.no_grad():
        for path in clips:
            x = vc.read_audio(path)
            condition = _condition(engine, x)
            # Train on the loudest contiguous two seconds of each isolated turn.
            span = min((2 * vc.RATE // vc.HOP) * vc.HOP, len(x) // vc.HOP * vc.HOP)
            strides = np.arange(0, len(x) - span + 1, vc.HOP)
            energy = np.concatenate(([0.], np.cumsum(x.astype("float64") ** 2)))
            start = strides[np.argmax(energy[strides + span] - energy[strides])]
            spec = engine._spec(x[start:start + span]).to(engine.device)
            length = torch.tensor([spec.shape[-1]], device=engine.device)
            g = torch.zeros_like(condition) if model.zero_g else condition
            z, _, _, mask = model.enc_q(spec, length, g=g, tau=0)
            teacher = model.flow(z, mask, g=condition)
            conditions.append(condition)
            pairs.append((z.detach(), mask.detach(), teacher.detach()))

    center = torch.stack(conditions).mean(0).detach()
    learned = center.clone().requires_grad_()
    optimizer = torch.optim.Adam([learned], lr=.0015)
    # Two deterministic sweeps give equal weight to separate speech turns.
    for _ in range(2):
        for z, mask, teacher in pairs:
            restored = model.flow(z, mask, g=learned)
            loss = (torch.nn.functional.smooth_l1_loss(restored, teacher, beta=.5)
                    + .02 * (learned - center).square().mean())
            if not torch.isfinite(loss):
                raise RuntimeError("The speaker adaptation produced a non-finite training loss.")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([learned], 1.)
            optimizer.step()
    return learned.detach()


def _promote(baseline, fitted):
    import numpy as np

    delta = np.asarray(fitted) - np.asarray(baseline)
    return bool(len(delta) >= 2 and np.isfinite(delta).all() and delta.mean() >= .005 and delta.min() >= -.02)


def _passes_quality(scores, genuine):
    import numpy as np

    # A beta readiness heuristic for an individual speaker. Relative improvement
    # alone can select the better of two unusable targets.
    floor = max(.65, float(np.mean(genuine)) - .12)
    return bool(np.mean(scores) >= floor and min(scores) >= floor - .10), floor


def fit_target(reference, training, holdouts, embed, *, comparison_center, previous=None,
               device="cpu", weights=None, threads=2):
    import numpy as np
    import torch
    from scipy.signal import resample_poly

    if len(set(training)) < 4 or len(set(holdouts)) < 2 or set(training) & set(holdouts):
        raise ValueError("Speaker adaptation needs four training turns and two separate validation turns.")
    engine = vc.ConversionEngine(target=reference, source_calibration=reference, weights=weights,
                                 threads=threads, device=device)
    baseline = engine.target
    fitted = _fit_condition(engine, training)
    if tuple(fitted.shape) != (1, 256, 1) or not torch.isfinite(fitted).all():
        raise ValueError("OpenVoice returned an invalid fitted speaker condition.")

    targets = {"reference": baseline, "fitted": fitted}
    if previous:
        old_ref, old_adapter = previous
        with torch.no_grad():
            if old_adapter:
                old = np.load(old_adapter, allow_pickle=False)
                if old.shape != (1, 256, 1) or old.dtype != np.float32 or not np.isfinite(old).all():
                    raise ValueError("The previous speaker adapter is invalid.")
                targets["previous"] = torch.from_numpy(old).to(device)
            else:
                targets["previous"] = _condition(engine, vc.read_audio(old_ref))
    scores = {name: [] for name in targets}
    genuine = []
    for path in holdouts:
        x = vc.read_audio(path)
        original = embed(resample_poly(x, 320, 441))
        genuine.append(float(original @ comparison_center))
        # Condition the source on this turn, as the live converter does with
        # microphone calibration. Only the target changes between the renders.
        engine.source_gain = vc.calibration_gain(x)
        with torch.no_grad():
            engine.source = _condition(engine, x * engine.source_gain)
        for name, target in targets.items():
            engine.target = target
            engine.reset_sampling()
            out = engine.convert(x)
            actual = embed(resample_poly(out, 320, 441))
            scores[name].append(float(comparison_center @ actual))
    chosen = "fitted" if _promote(scores["reference"], scores["fitted"]) else "reference"
    if previous:
        old_passes, _ = _passes_quality(scores["previous"], genuine)
        new_passes, _ = _passes_quality(scores[chosen], genuine)
        if (old_passes and not new_passes) or (
                old_passes == new_passes and not _promote(scores["previous"], scores[chosen])):
            chosen = "previous"
    passes, floor = _passes_quality(scores[chosen], genuine)
    kind = chosen if chosen != "previous" else ("fitted" if previous[1] else "reference")
    report = {"metric": "WeSpeaker cosine to training centroid on separate same-speaker reconstructions",
              "training_clips": len(training), "validation_clips": len(holdouts), "steps": 2 * len(training),
              "reference": float(np.mean(scores["reference"])), "fitted": float(np.mean(scores["fitted"])),
              "per_clip": scores, "genuine": float(np.mean(genuine)), "quality_floor": floor,
              "passes_quality": passes, "selected_score": float(np.mean(scores[chosen])),
              "selected": kind, "retained_previous": chosen == "previous"}
    if previous:
        report["previous"] = float(np.mean(scores["previous"]))
    adapter = fitted.cpu().numpy().astype("<f4") if chosen == "fitted" else None
    return adapter, report
