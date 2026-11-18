"""Assemble chosen clips from a single speaker into one reviewed reference."""

import json
import re
import wave
from pathlib import Path

from .tracks import new_directory

RATE = 24000
GAP = b"\x00\x00" * (RATE * 120 // 1000)


def pcm_frames(path, max_seconds=30):
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, RATE):
            raise ValueError(f"Expected a mono 24 kHz PCM16 WAV: {path}")
        if not 0 < audio.getnframes() <= max_seconds * RATE:
            raise ValueError(f"Expected an audio duration greater than 0 and at most {max_seconds}s: {path}")
        frames = audio.readframes(audio.getnframes())
        if len(frames) != audio.getnframes() * 2:
            raise ValueError(f"Truncated WAV: {path}")
        return frames


def _entries(speaker_dir):
    manifest = speaker_dir / "manifest.jsonl"
    result = {}
    if not re.fullmatch(r"[0-9]{1,20}", speaker_dir.name):
        raise ValueError("Choose a single numeric speaker directory produced by prepare.")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if not isinstance(entry, dict) or entry.get("speaker_id") != speaker_dir.name:
            raise ValueError(f"Speaker mismatch in {manifest}")
        name = entry.get("clip")
        if not isinstance(name, str) or not re.fullmatch(r"clips/[a-zA-Z0-9_-]+\.wav", name):
            raise ValueError(f"Invalid clip name in {manifest}")
        if name in result:
            raise ValueError(f"Duplicate clip in {manifest}: {name}")
        result[name] = entry
    return result


def run(args):
    speaker_dir = args.speaker_dir.resolve(strict=True)
    entries = _entries(speaker_dir)
    if len(args.clip) != len(set(args.clip)):
        raise ValueError("Choose each clip only once.")
    tracks = []
    transcripts = []
    frames_so_far = 0
    for name in args.clip:
        if name not in entries:
            raise ValueError(f"{name} is not listed in this speaker's manifest.")
        wav_path = (speaker_dir / name).resolve(strict=True)
        if not wav_path.is_relative_to(speaker_dir):
            raise ValueError(f"Clip is outside the selected speaker directory: {name}")
        frames = pcm_frames(wav_path)
        frames_so_far += len(frames) // 2 + (len(GAP) // 2 if tracks else 0)
        if frames_so_far > 30 * RATE:
            raise ValueError("Choose fewer clips: the reference must be at most 30 seconds with gaps.")
        tracks.append(frames)
        text = entries[name].get("text")
        transcripts.append(" ".join(text.split()) if isinstance(text, str) else "")

    seconds = frames_so_far / RATE
    if seconds < 3:
        raise ValueError("Choose enough clips for at least 3 seconds; aim for 20–30 seconds for fitting.")
    source = "unknown"
    dataset = speaker_dir.parent / "dataset.json"
    if dataset.is_file():
        source = json.loads(dataset.read_text(encoding="utf-8")).get("source", "unknown")

    with new_directory(args.output) as staging:
        with wave.open(str(staging / "reference.wav"), "wb") as wav:
            wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            wav.writeframes(GAP.join(tracks))
        # Never quietly ship a transcript with holes; missing ASR needs manual transcription.
        text = " ".join(transcripts) + "\n" if all(transcripts) else ""
        (staging / "transcript.txt").write_text(text, encoding="utf-8")
        (staging / "reference.json").write_text(json.dumps({
            "speaker_id": speaker_dir.name, "source": source, "clips": args.clip,
            "duration_s": round(seconds, 3), "has_complete_asr": bool(text),
        }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {seconds:.2f}s reference to {args.output}. Listen and review the transcript before fitting.")
    if not text:
        print("Transcript is empty because at least one clip has no ASR text. Transcribe it before fitting.")


def configure(commands):
    parser = commands.add_parser("reference", help="Bundle selected clips from one speaker")
    parser.add_argument("speaker_dir", type=Path, help="Prepared directory with manifest.jsonl")
    parser.add_argument("output", type=Path, help="New directory for reference.wav and transcript.txt")
    parser.add_argument("--clip", required=True, action="append", metavar="clips/NAME.wav",
                        help="A reviewed manifest clip; repeat in the desired order")
    parser.set_defaults(handler=run)
