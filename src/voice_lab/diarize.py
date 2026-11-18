"""Split a local recording into clips with one detected speaker at a time."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from typing import NamedTuple
from urllib.error import URLError
from urllib.request import urlopen
import wave

from .tracks import _now, new_directory


OUTPUT_RATE = 48_000
MODEL_RATE = 16_000
RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION = "pyannote-segmentation-3.0.onnx"
EMBEDDING = "wespeaker_en_voxceleb_resnet34.onnx"
SEGMENTATION_ARCHIVE = "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEGMENTATION_MEMBER = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
SEGMENTATION_HASH = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
SEGMENTATION_ARCHIVE_HASH = "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"
EMBEDDING_HASH = "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94"


class Turn(NamedTuple):
    """A half-open range of PCM frames from a single anonymous speaker."""

    start: int
    end: int
    speaker: int


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("input", type=Path, help="Local audio file containing mixed playback")
    parser.add_argument("output", type=Path, help="New directory for the split session")
    parser.add_argument("--speakers", type=int, help="Number of speakers if known")
    parser.add_argument("--min-seconds", type=float, default=1.5,
                        help="Shortest isolated turn to keep (default: 1.5)")
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/diarization"),
                        help="Model cache (default: data/models/diarization)")
    parser.add_argument("--local-files-only", action="store_true", help="Run with cached models only")


def _verify(path: Path, expected: str):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Model checksum mismatch: {path}. Remove or replace that file to proceed.")


def _models(directory: Path, local_only: bool):
    assets = [
        (directory / SEGMENTATION, SEGMENTATION_HASH,
         f"{RELEASE}/speaker-segmentation-models/{SEGMENTATION_ARCHIVE}"),
        (directory / EMBEDDING, EMBEDDING_HASH,
         f"{RELEASE}/speaker-recongition-models/{EMBEDDING}"),
    ]
    for model, checksum, url in assets:
        if model.is_file():
            _verify(model, checksum)
            continue
        if local_only:
            raise ValueError(f"Model not cached: {model}. Run once without --local-files-only to download it.")
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Downloading diarization model: {model.name}", flush=True)
        with TemporaryDirectory(prefix=".download-", dir=directory) as temporary:
            package = Path(temporary) / "download"
            with urlopen(url, timeout=30) as response, package.open("wb") as file:
                shutil.copyfileobj(response, file)
            if model.name == SEGMENTATION:
                _verify(package, SEGMENTATION_ARCHIVE_HASH)
                with tarfile.open(package, "r:bz2") as archive:
                    member = archive.getmember(SEGMENTATION_MEMBER)
                    if not member.isfile():
                        raise ValueError("The downloaded segmentation archive has no model file.")
                    extracted = Path(temporary) / model.name
                    with archive.extractfile(member) as content, extracted.open("wb") as file:
                        shutil.copyfileobj(content, file)
                package = extracted
            _verify(package, checksum)
            package.replace(model)
    return directory / SEGMENTATION, directory / EMBEDDING


def _ffmpeg(ffmpeg, input_file, *output_args):
    result = subprocess.run([
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
        "-protocol_whitelist", "file", "-i", str(input_file), "-map", "0:a:0",
        "-map_metadata", "-1", "-ac", "1", *output_args,
    ], capture_output=True)
    if result.returncode:
        error = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"Could not decode {input_file}: {error}")
    return result.stdout


def _diarizer(sherpa, segmentation, embedding, speakers):
    config = sherpa.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(segmentation)),
            num_threads=4,
        ),
        embedding=sherpa.SpeakerEmbeddingExtractorConfig(model=str(embedding), num_threads=4),
        clustering=sherpa.FastClusteringConfig(num_clusters=speakers or -1, threshold=0.5),
        min_duration_on=0.0,
        min_duration_off=0.0,
    )
    if not config.validate():
        raise ValueError("Sherpa-ONNX rejected the configured diarization models.")
    return sherpa.OfflineSpeakerDiarization(config)


def _exclusive_turns(segments, length: int):
    """Remove detected overlaps while retaining the surrounding single-speaker ranges."""
    events = defaultdict(list)
    for segment in segments:
        if not math.isfinite(segment.start) or not math.isfinite(segment.end):
            continue
        start = min(length, max(0, round(segment.start * OUTPUT_RATE)))
        end = min(length, max(0, round(segment.end * OUTPUT_RATE)))
        if end > start:
            events[start].append((segment.speaker, 1))
            events[end].append((segment.speaker, -1))

    turns = []
    previous = 0
    active = {}
    overlap = 0
    for frame, updates in sorted(events.items()):
        if len(active) > 1:
            overlap += frame - previous
        elif len(active) == 1 and frame > previous:
            speaker = next(iter(active))
            if turns and turns[-1].end == previous and turns[-1].speaker == speaker:
                turns[-1] = Turn(turns[-1].start, frame, speaker)
            else:
                turns.append(Turn(previous, frame, speaker))
        for speaker, change in updates:
            count = active.get(speaker, 0) + change
            if count:
                active[speaker] = count
            else:
                active.pop(speaker, None)
        previous = frame
    return turns, overlap


def _write_session(staging: Path, source: wave.Wave_read, turns):
    counts = defaultdict(int)
    speaker_ids = {}
    manifest = []
    for turn in turns:
        speaker = speaker_ids.setdefault(turn.speaker, str(len(speaker_ids) + 1))
        counts[speaker] += 1
        relative_clip = Path("raw") / speaker / f"{counts[speaker]:06d}.wav"
        clip_path = staging / relative_clip
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        source.setpos(turn.start)
        with wave.open(str(clip_path), "wb") as clip:
            clip.setparams((1, 2, OUTPUT_RATE, 0, "NONE", "not compressed"))
            remaining = turn.end - turn.start
            while remaining:
                requested = min(remaining, OUTPUT_RATE * 10)
                frames = source.readframes(requested)
                if len(frames) != requested * 2:
                    raise ValueError("Decoded recording ended before the expected audio frame.")
                clip.writeframesraw(frames)
                remaining -= requested
        manifest.append({
            "speaker_id": speaker,
            "clip": relative_clip.as_posix(),
            "start_s": round(turn.start / OUTPUT_RATE, 6),
            "end_s": round(turn.end / OUTPUT_RATE, 6),
        })
    return list(speaker_ids.values()), manifest


def run(args: argparse.Namespace):
    if args.speakers is not None and args.speakers < 1:
        raise ValueError("--speakers must be a positive integer.")
    if not math.isfinite(args.min_seconds) or args.min_seconds < 0.1:
        raise ValueError("--min-seconds must be at least 0.1.")
    input_file = args.input.expanduser().resolve(strict=True)
    if not input_file.is_file():
        raise ValueError(f"Input is not a regular audio file: {input_file}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH.")
    try:
        import numpy as np
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError("Install diarization support with: python -m pip install -e '.[diarize]'") from exc

    with new_directory(args.output.expanduser()) as staging:
        try:
            models = _models(args.model_dir.expanduser(), args.local_files_only)
        except URLError as exc:
            raise RuntimeError(f"Could not download diarization models: {exc.reason}") from exc
        engine = _diarizer(sherpa_onnx, *models, args.speakers)
        if engine.sample_rate != MODEL_RATE:
            raise RuntimeError(f"Unexpected diarization sample rate: {engine.sample_rate}")

        raw = staging / ".mixed.wav"
        _ffmpeg(ffmpeg, input_file, "-ar", str(OUTPUT_RATE), "-c:a", "pcm_s16le", str(raw))
        with wave.open(str(raw), "rb") as source:
            length = source.getnframes()
            if not length:
                raise ValueError("Input contains no audio.")
            model_audio = _ffmpeg(ffmpeg, raw, "-ar", str(MODEL_RATE), "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1")
            samples = np.frombuffer(model_audio, dtype="<f4").astype(np.float32, copy=False)
            print(f"Diarizing {length / OUTPUT_RATE:.1f}s of audio on CPU", flush=True)
            result = engine.process(samples).sort_by_start_time()
            turns, overlap = _exclusive_turns(result, length)
            minimum = args.min_seconds * OUTPUT_RATE
            kept = [turn for turn in turns if turn.end - turn.start >= minimum]
            short = sum(turn.end - turn.start for turn in turns if turn.end - turn.start < minimum)
            if not kept:
                raise ValueError("No usable single-speaker turns found. Try longer speech, less cross-talk, or a shorter --min-seconds.")
            speaker_ids, manifest = _write_session(staging, source, kept)
        raw.unlink()
        metadata = {
            "schema_version": 1,
            "source": "discord-desktop",
            "processed_at": _now(),
            "participants": [{"id": speaker, "display_name": f"Speaker {speaker}"} for speaker in speaker_ids],
            "segments": manifest,
            "diarization": {
                "engine": "sherpa-onnx",
                "expected_speakers": args.speakers,
                "input_duration_s": round(length / OUTPUT_RATE, 6),
                "discarded_overlap_s": round(overlap / OUTPUT_RATE, 6),
                "discarded_short_s": round(short / OUTPUT_RATE, 6),
            },
        }
        (staging / "session.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(manifest)} turns for {len(speaker_ids)} detected speakers in {args.output}. "
          f"Discarded {overlap / OUTPUT_RATE:.2f}s of detected overlap.")
