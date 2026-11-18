"""Import isolated local microphone tracks or generate an offline smoke fixture."""

import json
import math
import re
import shutil
import struct
import subprocess
import tempfile
import wave
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


@contextmanager
def new_directory(destination: Path):
    """Build a private sibling directory and publish it once complete."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".voice-lab-", dir=destination.parent))
    try:
        yield staging
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Output already exists: {destination}")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _metadata(staging, source, speaker_ids, started):
    fields = {
        "schema_version": 1,
        "source": source,
        "started_at": started,
        "ended_at": _now(),
        "participants": [
            {"id": speaker, "display_name": f"Speaker {speaker}", "consented": True}
            for speaker in sorted(speaker_ids)
        ],
    }
    (staging / "session.json").write_text(json.dumps(fields, indent=2) + "\n", encoding="utf-8")


def _parse_tracks(specs):
    tracks = []
    seen = set()
    for spec in specs:
        speaker, sep, path = spec.partition("=")
        if not sep or not re.fullmatch(r"[0-9]{1,20}", speaker):
            raise ValueError("Use --track SPEAKER_ID=FILE with a numeric speaker ID.")
        file = Path(path).expanduser().resolve(strict=True)
        if not file.is_file():
            raise ValueError(f"Track is not a regular file: {file}")
        if file in seen:
            raise ValueError(f"Track was supplied twice: {file}")
        seen.add(file)
        tracks.append((speaker, file))
    return tracks


def import_tracks(args):
    if not args.consent:
        raise ValueError("Pass --consent after each speaker has agreed to custom voice use.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install FFmpeg before importing tracks.")
    tracks = _parse_tracks(args.track)
    started = _now()
    with new_directory(args.output) as staging:
        counts = {}
        for speaker, file in tracks:
            counts[speaker] = counts.get(speaker, 0) + 1
            owner = staging / "raw" / speaker
            owner.mkdir(parents=True, exist_ok=True)
            target = owner / f"{counts[speaker]:06d}.wav"
            result = subprocess.run([
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist",
                "file", "-i", str(file), "-map", "0:a:0", "-map_metadata", "-1",
                "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(target),
            ], check=False, capture_output=True, text=True)
            if result.returncode:
                raise ValueError(f"Could not import {file}: {result.stderr.strip()}")
            with wave.open(str(target), "rb") as audio:
                if audio.getnframes() == 0:
                    raise ValueError(f"Track contains no audio: {file}")
        _metadata(staging, "local", counts, started)
    print(f"Imported {len(tracks)} track(s) for {len(counts)} speaker(s): {args.output}")


def _tones(target, fundamental, take):
    """Write a modulated tone with leading and trailing quiet; no human voice."""
    rate = 48000
    frames = bytearray()
    for index in range(5 * rate):
        time = index / rate
        envelope = max(0, min(1, (time - 0.25) / 0.08, (4.75 - time) / 0.08))
        amplitude = envelope * (0.65 + 0.30 * math.sin(2 * math.pi * 3.2 * time))
        phase = 2 * math.pi * (fundamental + take * 8) * time + 0.2 * math.sin(8 * time)
        sample = int(8500 * amplitude * (math.sin(phase) + 0.2 * math.sin(phase * 2)))
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(target), "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes(frames)


def fixture(args):
    started = _now()
    speakers = {"101": 190, "202": 265}
    with new_directory(args.output) as staging:
        for speaker, pitch in speakers.items():
            directory = staging / "raw" / speaker
            directory.mkdir(parents=True)
            for take in range(1, 7):
                _tones(directory / f"{take:06d}.wav", pitch, take)
        _metadata(staging, "synthetic", speakers, started)
    print(f"Created synthetic two-speaker session: {args.output}")


def configure(commands):
    parser = commands.add_parser("import-tracks", help="Import local, isolated microphone tracks")
    parser.add_argument("output", type=Path, help="New session directory")
    parser.add_argument("--track", metavar="ID=FILE", action="append", required=True,
                        help="One local mic take for one speaker; repeat for more tracks")
    parser.add_argument("--consent", action="store_true", help="Speakers consented to custom voice use")
    parser.set_defaults(handler=import_tracks)

    parser = commands.add_parser("fixture", help="Generate a two-speaker session using synthetic tones")
    parser.add_argument("output", type=Path)
    parser.set_defaults(handler=fixture)
