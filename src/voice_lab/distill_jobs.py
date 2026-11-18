"""Isolated training stages. The studio process never imports the teacher or student."""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import threading
import sys
import time

from .beta import _atomic_text


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    _atomic_text(path, json.dumps(value, allow_nan=False, indent=2) + "\n")


def locate(directory, value):
    """Resolve library-relative evidence, including absolute paths saved by early betas."""
    path = Path(value)
    if not path.is_absolute():
        return (directory / path).resolve()
    call = next((p for p in directory.parents if p.parent.name == "calls"), None)
    if call:
        for i, part in enumerate(path.parts[:-1]):
            if part == "calls" and path.parts[i + 1] == call.name:
                local = call.joinpath(*path.parts[i + 2:])
                if local.is_file():
                    return local
    return path


def progress(message):
    print("voice-job " + json.dumps({"message": str(message)}), flush=True)


def corpus(directory, model_dir, count, cache=None, prompts=None, adapt_teacher=False):
    from .diarize import EMBEDDING
    from .distill_teacher import QualityChecker, prepare_reference, generate_corpus

    inputs = read(directory / "inputs.json")
    train = [locate(directory, p) for p in inputs["training_clips"]]
    held = [locate(directory, p) for p in inputs["heldout_clips"]]
    checker = QualityChecker(model_dir / EMBEDDING, asr_dir=os.environ.get("VOICE_LAB_WHISPER_MODEL"))
    ref, transcript = prepare_reference(train, directory / "reference", holdouts=held, checker=checker, progress=progress)
    # Only the genuine evaluation audio is transcribed here. Its text and audio
    # stay outside both the generated corpus and the student optimization loop.
    reviews = []
    for path in held:
        text = checker.transcribe(path).strip()
        if text:
            reviews.append({"original": os.path.relpath(path, directory), "text": text})
    if not reviews:
        raise ValueError("The evaluation clips contained no clear words. Collect more clean speech.")
    write(directory / "review-lines.json", reviews)
    adapter = None
    corpus_dir = cache or directory / "corpus"
    excluded = [row["text"] for row in reviews]
    if adapt_teacher or count >= 128:
        from .qwen_adapt import choose_teacher
        from .distill_teacher import _snapshot, QWEN_REPO, QWEN_REVISION

        model = _snapshot(None, "VOICE_LAB_QWEN_MODEL", QWEN_REPO, QWEN_REVISION)
        try:
            ref, transcript, adapter, report = choose_teacher(
                train, held, corpus_dir / "teacher-adaptation", ref, transcript, checker, model, progress,
                excerpt_cache=corpus_dir.parent / "teacher-excerpts")
        except (OSError, RuntimeError, ValueError, MemoryError, ImportError, AttributeError, KeyError) as exc:
            report = {"selected": "base", "reason": f"Optional teacher fit unavailable: {exc}"}
        excluded.extend(row["text"] for row in report.get("before", ()))
        progress(report["reason"])
    else:
        report = {"selected": "base", "reason": "Quick preview uses the reference-conditioned teacher."}
    write(directory / "teacher-decision.json", report)
    manifest = generate_corpus(corpus_dir, ref, transcript, holdouts=held, count=count, prompts=prompts,
                               checker=checker, embedding_model=model_dir / EMBEDDING, progress=progress,
                               excluded_texts=excluded, adapter=adapter, freeze_manifest=True)
    write(directory / "manifest-path.json", os.path.relpath(manifest, directory))


def student(directory, steps):
    import torch
    from .distill_student import Synthesizer, train
    profile = directory / "student" / "profile.pt"
    if not profile.is_file():
        options = {"steps": steps} if steps else {}
        train(locate(directory, read(directory / "manifest-path.json")), profile.parent, progress=progress, resume=True, **options)
        # A repaired profile must be evaluated on audio rendered with its own weights.
        for index in range(len(read(directory / "review-lines.json"))):
            (directory / f"review-{index}.wav").unlink(missing_ok=True)
        (directory / "evaluation.json").unlink(missing_ok=True)
        (directory / "review.wav").unlink(missing_ok=True)
    synth = Synthesizer(profile)
    for index, row in enumerate(read(directory / "review-lines.json")):
        progress(f"Rendering evaluation line {index + 1}…")
        path = directory / f"review-{index}.wav"
        if not path.is_file():
            # Use identical excitation noise for the same held-out line across new builds.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(950 + index)
                synth.render(row["text"], path)


def _add_delivery(directory, rows):
    from . import prosody
    from .distill_teacher import _hash_file
    for index, row in enumerate(rows):
        original = locate(directory, row["original"])
        rendered = directory / f"review-{index}.wav"
        if original.is_file() and rendered.is_file():
            row.update(prosody.errors(original, rendered))
            row["original_sha256"] = _hash_file(original)
    return prosody.summary(rows)


def evaluate(directory, model_dir, incumbent=None):
    from .diarize import EMBEDDING
    from .distill_teacher import QualityChecker, word_error_rate

    checker = QualityChecker(model_dir / EMBEDDING, asr_dir=os.environ.get("VOICE_LAB_WHISPER_MODEL"))
    import numpy as np
    from .beta_fit import _passes_quality
    rows = read(directory / "review-lines.json")
    training = read(directory / "inputs.json")["training_clips"]
    center = np.mean([checker.embedding(locate(directory, p)) for p in training], axis=0)
    center /= max(np.linalg.norm(center), 1e-6)
    for index, row in enumerate(rows):
        original = locate(directory, row["original"])
        rendered = directory / f"review-{index}.wav"
        row["heard"] = checker.transcribe(rendered)
        row["word_error_rate"] = word_error_rate(row["text"], row["heard"])
        genuine = checker.embedding(original)
        row["genuine_to_training"] = float(genuine @ center)
        row["similarity"] = float(genuine @ checker.embedding(rendered))
        if index == 0:
            from .mic_preview import compare_takes
            if not (directory / "review.wav").is_file():
                compare_takes(original, rendered, directory / "review.wav")
    genuine = [r["genuine_to_training"] for r in rows]
    passed, floor = _passes_quality([r["similarity"] for r in rows], genuine)
    wer = statistics.mean(r["word_error_rate"] for r in rows)
    delivery = _add_delivery(directory, rows)
    if incumbent and incumbent != directory and (incumbent / "evaluation.json").is_file():
        previous = read(incumbent / "evaluation.json")
        previous["incumbent_build"] = incumbent.name
        previous["prosody"] = _add_delivery(incumbent, previous["lines"])
        write(directory / "incumbent-evaluation.json", previous)
    write(directory / "evaluation.json", {
        "prosody": delivery,
        "passes_quality": passed and wer <= .25, "quality_floor": floor,
        "genuine_cosine": statistics.mean(genuine), "lines": rows, "mean_cosine": statistics.mean(r["similarity"] for r in rows),
        "mean_word_error_rate": wer,
    })


def render_server(directory):
    from .distill_student import Synthesizer
    synth = Synthesizer(directory / "student" / "profile.pt")
    for line in sys.stdin:
        job = json.loads(line)
        text = job["text"]
        try:
            start = time.perf_counter()
            # Validated by the studio, and never used to construct arbitrary paths.
            path = directory / "utterance-next.wav"
            path.unlink(missing_ok=True)
            synth.render(text, path)
            path.replace(directory / "utterance.wav")
            reply = {"seconds": round(time.perf_counter() - start, 3), "text": text}
        except Exception as exc:
            reply = {"error": str(exc)}
        print("voice-render " + json.dumps(reply), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["corpus", "student", "evaluate", "render"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--adapt-teacher", action="store_true", help="Try a Qwen acoustic fit even for a quick corpus")
    parser.add_argument("--incumbent", type=Path, help="Current published voice for a paired evaluation")
    parser.add_argument("--cache", type=Path, help="Reusable corpus directory for this speaker and reference set")
    parser.add_argument("--prompts", type=Path, help="Optional UTF-8 file with one English teaching line per row")
    args = parser.parse_args(argv)
    expected = os.environ.get("VOICE_LAB_JOB_PARENT")
    if expected:
        def watch_studio():
            while os.getppid() == int(expected):
                time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(target=watch_studio, daemon=True).start()
    if args.stage == "corpus":
        corpus(args.directory, args.model_dir, args.count, args.cache, args.prompts, args.adapt_teacher)
    elif args.stage == "student":
        student(args.directory, args.steps)
    elif args.stage == "evaluate":
        evaluate(args.directory, args.model_dir, args.incumbent)
    else:
        render_server(args.directory)


if __name__ == "__main__":
    main()
