"""Prepare separate speaker tracks for local voice enrollment or training."""

import argparse
from array import array
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import wave


SOURCE_RATE = 48_000
OUTPUT_RATE = 24_000
ROOM_RUMBLE_FILTER = "highpass=f=70"
SILENCE_EVENT = re.compile(
    r"^\[(?:Parsed_)?silencedetect(?:_\d+)? @ [^\]\r\n]+\]\s+"
    r"silence_(start|end):\s*(-?[\d.]+(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)
SPEAKER_ID = re.compile(r"[0-9]+")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("session", type=Path, help="directory containing session.json and raw/")
    parser.add_argument("output", type=Path, help="new dataset directory")
    parser.add_argument("--min-seconds", type=float, default=1.5, help="shortest retained clip (default: 1.5)")
    parser.add_argument("--max-seconds", type=float, default=12.0, help="longest retained clip (default: 12)")
    parser.add_argument("--pause-seconds", type=float, default=0.55, help="pause length that starts a new clip (default: 0.55)")
    parser.add_argument("--silence-db", type=float, default=-42.0, help="silence threshold in dBFS (default: -42)")
    parser.add_argument("--denoise", action="store_true", help="apply light FFT reduction for steady background noise")
    parser.add_argument("--whisper-model", default="small.en", help="English Faster-Whisper model name or local directory (default: small.en)")
    parser.add_argument("--local-files-only", action="store_true", help="use a locally cached Whisper model without downloading")
    parser.add_argument("--no-transcribe", action="store_true", help="make audio clips without loading Faster-Whisper")


def _load_session(session: Path) -> tuple[str, set[str], dict[str, float]]:
    path = session / "session.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("session.json must be a regular file in the session directory")
    with path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("participants"), list):
        raise ValueError("session.json must contain a participants array")
    source = metadata.get("source", "local")
    if source not in ("discord", "discord-desktop", "local", "synthetic"):
        raise ValueError("session.json source must be discord, discord-desktop, local, or synthetic")
    if source == "discord":
        try:
            ended_at = datetime.fromisoformat(metadata["ended_at"].replace("Z", "+00:00"))
            if ended_at.tzinfo is None:
                raise ValueError
        except (AttributeError, KeyError, TypeError, ValueError) as err:
            raise ValueError("Discord session is missing a valid ended_at timestamp from the recorder") from err

    seen = set()
    allowed = set()
    for participant in metadata["participants"]:
        speaker_id = participant.get("id") if isinstance(participant, dict) else None
        if not isinstance(speaker_id, str) or not SPEAKER_ID.fullmatch(speaker_id):
            raise ValueError("participant IDs must contain only ASCII digits and be stored as JSON strings")
        if speaker_id in seen:
            raise ValueError(f"duplicate participant ID in session.json: {speaker_id}")
        seen.add(speaker_id)
        if source == "discord-desktop" or participant.get("consented") is True:
            allowed.add(speaker_id)

    offsets = {}
    if source == "discord-desktop":
        segments = metadata.get("segments", [])
        if not isinstance(segments, list):
            raise ValueError("desktop session segments must be an array")
        for segment in segments:
            clip = segment.get("clip") if isinstance(segment, dict) else None
            start = segment.get("start_s") if isinstance(segment, dict) else None
            if (not isinstance(clip, str) or clip in offsets or isinstance(start, bool)
                    or not isinstance(start, (float, int)) or not math.isfinite(start) or start < 0):
                raise ValueError("desktop session contains an invalid clip or recording timestamp")
            offsets[clip] = start
    return source, allowed, offsets


def _ffmpeg(*arguments: str) -> subprocess.CompletedProcess:
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-xerror", *arguments]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg exited with status {result.returncode}:\n{details[-2000:]}")
    return result


def _input_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as track:
            if track.getframerate() != SOURCE_RATE or track.getsampwidth() != 2 or track.getnchannels() not in (1, 2):
                raise ValueError(f"expected 48 kHz mono or stereo 16-bit PCM: {path}")
            return track.getnframes() / SOURCE_RATE
    except (wave.Error, EOFError) as err:
        raise ValueError(f"invalid PCM WAV: {path}") from err


def _find_silences(path: Path, duration: float, threshold: float, scan_pause: float) -> list[tuple[float, float]]:
    result = _ffmpeg(
        "-loglevel", "info", "-i", str(path), "-map", "0:a:0", "-af",
        f"{ROOM_RUMBLE_FILTER},silencedetect=n={threshold:g}dB:d={scan_pause:.6f}",
        "-f", "null", "-",
    )
    pauses = []
    start = None
    for kind, value in SILENCE_EVENT.findall(result.stderr.decode("utf-8", errors="replace")):
        at = min(duration, max(0.0, float(value)))
        if kind == "start":
            start = at
        elif start is not None:
            if at > start:
                pauses.append((start, at))
            start = None
    if start is not None and duration > start:
        pauses.append((start, duration))
    return pauses


def _segments(
    duration: float, pauses: list[tuple[float, float]], minimum: float, maximum: float, split_pause: float,
) -> list[tuple[float, float]]:
    # Keep a little context around each utterance without crossing the center of a pause.
    speech_regions = []
    speech_start = 0.0
    clip_start = 0.0
    for start, end in [*pauses, (duration, duration)]:
        is_eof = end == duration
        if end - start + 1 / SOURCE_RATE < split_pause and start > 0 and not is_eof:
            continue
        pad = min(0.12, (end - start) / 2)
        if start > speech_start:
            speech_regions.append((clip_start, start + pad))
        speech_start = end
        clip_start = end - pad

    clips = []
    pause_centers = [(start + end) / 2 for start, end in pauses]
    for start, end in speech_regions:
        while end - start > maximum:
            length = end - start
            parts = math.ceil(length / maximum)
            ideal = start + length / parts
            if ideal - start < minimum:
                ideal = start + maximum
            earliest = max(start + minimum, end - (parts - 1) * maximum, ideal - 0.75)
            latest = min(start + maximum, end - (parts - 1) * minimum, ideal + 0.75)
            near_pauses = (center for center in pause_centers if earliest <= center <= latest)
            cut = min(near_pauses, key=lambda center: abs(center - ideal), default=ideal)
            clips.append((start, cut))
            start = cut
        if end - start >= minimum:
            clips.append((start, end))
    return clips


def _clean_clip(source: Path, output: Path, start: float, end: float, denoise: bool) -> tuple[float, float] | None:
    filters = ROOM_RUMBLE_FILTER
    if denoise:
        filters += ",afftdn=nr=8:nf=-45"
    content = _ffmpeg(
        "-loglevel", "error", "-ss", f"{start:.8f}", "-i", str(source), "-map", "0:a:0",
        "-t", f"{end - start:.8f}", "-af", filters, "-ac", "1", "-ar", str(OUTPUT_RATE),
        "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1",
    ).stdout
    samples = array("f")
    samples.frombytes(content)
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return None
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    if rms < 10 ** (-55 / 20):
        return None
    peak = max(abs(sample) for sample in samples)
    gain = min(10 ** (9 / 20), 10 ** (-3 / 20) / peak)
    fade_frames = min(round(0.005 * OUTPUT_RATE), len(samples) // 2)
    pcm = array("h", (round(sample * gain * 32767) for sample in samples))
    for frame in range(fade_frames):
        taper = frame / fade_frames
        pcm[frame] = round(pcm[frame] * taper)
        pcm[-1 - frame] = round(pcm[-1 - frame] * taper)
    if sys.byteorder == "big":
        pcm.byteswap()
    with wave.open(str(output), "wb") as wav:
        wav.setparams((1, 2, OUTPUT_RATE, 0, "NONE", "not compressed"))
        wav.writeframes(pcm.tobytes())
    return len(samples) / OUTPUT_RATE, 20 * math.log10(rms)


def _load_whisper(model_name: str, local_only: bool):
    try:
        from faster_whisper import WhisperModel
    except ImportError as err:
        raise RuntimeError("Install the project's transcription extra or use --no-transcribe.") from err
    return WhisperModel(model_name, device="cpu", compute_type="int8", local_files_only=local_only)


def _transcribe(model, clip: Path) -> tuple[str | None, float | None]:
    import numpy as np

    # Decode through the same FFmpeg binary as the preparation step. This avoids
    # Faster-Whisper's file decoder and uses its native 16 kHz waveform input.
    content = _ffmpeg(
        "-loglevel", "error", "-i", str(clip), "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1",
    ).stdout
    audio = np.frombuffer(content, dtype="<f4").astype(np.float32, copy=False)
    segments, _ = model.transcribe(
        audio, language="en", beam_size=5, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500}, condition_on_previous_text=False,
    )
    segments = [segment for segment in segments if segment.text.strip()]
    text = " ".join(" ".join(segment.text.split()) for segment in segments)
    scores = [segment.avg_logprob for segment in segments if math.isfinite(segment.avg_logprob)]
    return text or None, round(sum(scores) / len(scores), 4) if scores else None


def run(args: argparse.Namespace) -> None:
    if not 0.25 <= args.min_seconds <= args.max_seconds <= 60:
        raise ValueError("clip limits must be between 0.25 and 60 seconds, with min <= max")
    if not 0.10 <= args.pause_seconds <= 10:
        raise ValueError("pause length must be between 0.10 and 10 seconds")
    if not -90 <= args.silence_db <= -5:
        raise ValueError("silence threshold must be between -90 and -5 dBFS")
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"choose a new output directory: {output}")
    session = args.session.expanduser().resolve()
    source, allowed, recording_offsets = _load_session(session)
    raw = session / "raw"
    if raw.is_symlink() or not raw.is_dir():
        raise ValueError("the session must contain a real raw/ directory")
    speaker_dirs = sorted(path for path in raw.iterdir() if not path.is_symlink() and path.is_dir())
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg was not found on PATH")
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = {
        "schema_version": 1,
        "source": source,
        "sample_rate_hz": OUTPUT_RATE,
        "settings": {
            "min_seconds": args.min_seconds,
            "max_seconds": args.max_seconds,
            "pause_seconds": args.pause_seconds,
            "silence_db": args.silence_db,
            "denoise": args.denoise,
            "whisper_model": None if args.no_transcribe else Path(args.whisper_model).name,
        },
        "speakers": [],
        "skipped_unapproved_speaker_dirs": sum(path.name not in allowed for path in speaker_dirs),
    }
    model = None
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=output.parent) as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir(mode=0o700)
        for track_dir in speaker_dirs:
            speaker_id = track_dir.name
            if speaker_id not in allowed:
                continue
            speaker = stage / speaker_id
            audio_dir = speaker / "clips"
            audio_dir.mkdir(parents=True)
            count = transcribed = 0
            total_duration = 0.0
            with (speaker / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
                for path in sorted(track_dir.iterdir()):
                    if path.suffix.lower() != ".wav" or path.is_symlink() or not path.is_file():
                        continue
                    duration = _input_duration(path)
                    if duration < args.min_seconds:
                        continue
                    relative_path = path.relative_to(session).as_posix()
                    recording_offset = recording_offsets.get(relative_path)
                    pauses = _find_silences(path, duration, args.silence_db, min(0.15, args.pause_seconds))
                    for start, end in _segments(duration, pauses, args.min_seconds, args.max_seconds, args.pause_seconds):
                        clip = audio_dir / f"clip-{count + 1:06}.wav"
                        audio_info = _clean_clip(path, clip, start, end, args.denoise)
                        if audio_info is None:
                            continue
                        clean_duration, rms_db = audio_info
                        transcript = confidence = None
                        if not args.no_transcribe:
                            if model is None:
                                model = _load_whisper(args.whisper_model, args.local_files_only)
                            transcript, confidence = _transcribe(model, clip)
                        row = {
                            "clip": clip.relative_to(speaker).as_posix(),
                            "speaker_id": speaker_id,
                            "text": transcript,
                            "asr_avg_logprob": confidence,
                            "duration_s": round(clean_duration, 5),
                            "source": relative_path,
                            "start_s": round(start, 5),
                            "end_s": round(end, 5),
                            "pre_gain_rms_dbfs": round(rms_db, 2),
                        }
                        if recording_offset is not None:
                            row["recording_start_s"] = round(recording_offset + start, 5)
                            row["recording_end_s"] = round(recording_offset + end, 5)
                        print(json.dumps(row, ensure_ascii=False), file=manifest)
                        count += 1
                        transcribed += bool(transcript)
                        total_duration += clean_duration
            if count == 0:
                shutil.rmtree(speaker)
                continue
            dataset["speakers"].append({
                "speaker_id": speaker_id,
                "clips": count,
                "duration_s": round(total_duration, 3),
                "transcribed_clips": transcribed,
            })
        if not dataset["speakers"]:
            raise ValueError("no usable clips in the opted-in speaker tracks; check the input audio and clip limits")
        (stage / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output directory was created during processing: {output}")
        stage.rename(output)

    count = len(dataset["speakers"])
    print(f"Prepared {sum(speaker['clips'] for speaker in dataset['speakers'])} clips "
          f"for {count} {'speaker' if count == 1 else 'speakers'} in {output}")
    if dataset["skipped_unapproved_speaker_dirs"]:
        detail = "outside the session manifest" if source == "discord-desktop" else "without recorded consent"
        print(f"Skipped {dataset['skipped_unapproved_speaker_dirs']} speaker directories {detail}.", file=sys.stderr)
