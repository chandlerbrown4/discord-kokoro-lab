"""Create an English text/audio corpus with a local Qwen3-TTS teacher on Apple silicon."""

from contextlib import contextmanager
import fcntl
import hashlib
from itertools import chain
import json
import math
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

QWEN_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
QWEN_REVISION = "e7dd0585652209fa0d7783659aad4e8a324de11c"
ASR_REPO = "Systran/faster-whisper-small.en"
ASR_REVISION = "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
RATE = 24000
DEFAULT_COUNT = 128
MIN_COSINE = .62
MAX_WER = .15

# Punctuation, questions, short answers and longer clauses vary the speech task.
_LINES = """Can you hear the rain outside?
Give me a minute to find my keys.
I thought we were meeting around noon.
That's exactly what I wanted to know.
Wait, which one did you mean?
The little yellow boat made it across the lake.
I'd pick the second one if it were up to me.
Do you remember where we left the charger?
Okay, I can see it now.
She laughed so hard that she dropped her pencil.
Could you lower the volume just a little?
We should probably leave before the traffic builds up.
That sounds much better than my original idea.
How did you manage to get there so quickly?
Actually, I was going to ask you the same thing.
We'll find out when the doors open.
Yes, but the instructions were written in pencil.
There were three boxes underneath the shelf.
I didn't notice the sign until we walked past it.
Did the dog really jump over that fence?
Please save me a seat by the window.
After the storm, the whole town smelled like pine.
I can't quite remember how the story ends.
Would you like some tea while we wait?
Try turning the handle in the other direction.
What a strange place to leave a bicycle.
My first guess was closer than I expected.
Let's give the next person a chance.
Which of these two pictures looks clearer?
The train slowed down just before the bridge.
Usually I take a different way home.
I hope the neighbors heard our good news.
That's a fair question. Let me think.
Tell me again how the clock got upstairs.
There might be a spare in the blue drawer.
Somebody moved my jacket while I was outside.
I think we have everything except the tickets.
Before you start, could you check the time?
They've already finished painting the front porch.
A quiet room is all I really need.
How many people are coming over on Friday?
I'd never seen a sky quite that color.
Just leave the package beside the stairs.
I was about to say the very same thing.
It's surprisingly warm for this time of year.
The second verse sounds different from the first.
Well, what would you do in my place?
We forgot the umbrella again, didn't we?
You'll have to show me how that works.
I kept a copy of the letter in my desk.
Is your microphone picking up that ticking sound?
The map has a tiny coffee stain in the corner.
I can bring an extra blanket if it gets cold.
Nobody knew the answer, so we looked it up.
Let's walk a little farther before we stop.
His watch was fifteen minutes fast all afternoon.
Is this really the shortest path to the station?
I couldn't stop smiling when I opened the envelope.
There's a sharp bend just beyond the orchard.
Could you say the last part one more time?
At first it was quiet. Then everyone started singing.
I didn't expect the soup to taste so good.
You'll recognize her by the bright green scarf.
What's the easiest way to carry all this?
Hang on, I think the kettle just clicked off.
Take your time. I'll be right here.
Who taught you to play that tune?
My favorite part was the silence between the two verses.
There's probably enough left for both of us.
You'd be surprised how often that happens.
Oh, I see what you were pointing at!
Nothing about the route looked familiar after dark.
A little more ginger might make a difference.
It's tucked behind the loose brick near the fireplace.
How did the interview go this morning?
I'm still thinking about what you said yesterday.
The upstairs neighbor practices violin on Thursday evenings.
Check the spelling on the label before you print.
What if we try a different entrance?
Something rattles when I shake the box.
I might be able to get the stain out.
We were so close to catching the last bus.
I can swap shifts with you next week.
Does the light look brighter from where you're standing?
At the time it seemed like a perfectly sensible idea.
My cousin insists there's a shortcut through the alley.
This is the first ripe peach of the summer.
Do you know a good place for breakfast nearby?
So far, so good. How are things on your end?
You can borrow mine until yours is fixed.
The cat hasn't forgiven me for moving her chair.
It looks ordinary until you turn it sideways.
Where do you think that music is coming from?
Nobody remembers who brought the board game.
She's taking the ferry back on Sunday afternoon.
Honestly, I hadn't even considered that possibility.
What made you change your mind?
It fits in the suitcase if you fold it twice.
The kitchen timer rang just as the phone started buzzing.
There's still room for a few more chairs.
I only remember bits and pieces of that trip.
If you can hold the ladder, I can reach it.
Will that be enough for everyone at the table?
I found a quicker way to sort the photographs.
Careful, the floor is slippery by the sink.
I just realized why the numbers looked so odd.
Could you catch the door before it closes?
He waved to us from the back of the crowd.
All the benches were taken except the one in the shade.
I wasn't sure whether to mention it earlier.
How far did you get with the crossword?
That's the sort of thing my grandfather would say.
The recipe works with frozen berries, too.
Can we take a break after the next chapter?
I heard footsteps and assumed it was the mail carrier.
There's something I keep meaning to ask you.
It took a while for my eyes to adjust.
Some afternoons I can hear the school bell from here.
I nearly forgot about the towels in the dryer.
They moved the start time forward by half an hour.
What's the first thing you'd like to do tomorrow?
A single yellow flower grew through the crack.
She kept the old receipt inside her passport.
I suppose we could always paint it again.
You were right about the wheels needing oil.
There's no rush. The bread is still in the oven.
The wind gets pretty strong along this stretch of road.
I wonder which team ended up winning.
Please remind me to call before we leave.
The last page made the rest of the book make sense.
How about we try it once at normal speed?
I keep hearing an echo when I speak.
Everyone got quiet when the lights dimmed.
Would it help if I drew you a picture?
They built the stage from pieces of the old dock.
Let me know when the water starts bubbling.
The glass looked almost purple against the white wall.
I'd like to hear how that sounds with the whole band.
What do you usually do with the leftovers?
Somehow I managed to grab two left gloves.
I need to rewind a bit to understand what she meant.
The last few steps were steeper than they looked.
I can stay until someone else gets here.
What was that little shop on the corner called?
You can tell how windy it was from the way the trees lean.""".splitlines()

_PEOPLE = ("Alex", "Jordan", "Maya", "Sam", "Ellie", "Morgan", "Drew", "Robin", "Jess", "Lee", "Nora", "Jamie")
_THINGS = ("the parcel", "your coat", "my glasses", "our tickets", "the map", "that book", "the spare key",
           "the camera", "our umbrella", "the menu", "the blue folder", "a postcard", "my notebook", "the photograph")
_PLACES = ("by the stairs", "beside the window", "in the kitchen", "under the bench", "near the fountain",
           "at the station", "in the hallway", "inside the box", "under the table", "on the shelf",
           "next to the door", "behind the counter")
_TIMES = ("today", "yesterday", "on Monday", "last night", "before lunch", "this morning", "last weekend", "on Friday")
_PATTERNS = (
    "{p} noticed {t} {l} {w}.", "Who left {t} {l} {w}, {p}?",
    "I think {p} spotted {t} {l} {w}.", "I didn't know {p} had left {t} {l} {w}.",
    "Apparently {p} found {t} {l} {w}. Has anyone checked?",
    "Did {p} really leave {t} {l} {w}?", "Remember when {p} put {t} {l} {w}?",
    "Nobody told {p} about {t} {l} until {w}.",
)


def _words(text):
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[a-z0-9]+", normalized.replace("'", ""))


def word_error_rate(expected, heard):
    """Word-level edit distance divided by the expected word count."""
    target, prediction = _words(expected), _words(heard)
    if not target:
        raise ValueError("Expected transcript needs at least one English word.")
    row = list(range(len(prediction) + 1))
    for i, word in enumerate(target, 1):
        next_row = [i]
        for j, other in enumerate(prediction, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (word != other)))
        row = next_row
    return row[-1] / len(target)


def _prompts(source):
    if source is None:
        # Permute mixed-radix indices with a coprime stride to interleave patterns,
        # nouns and locations. Memory use is independent of the available combinations.
        def variations():
            pools = (_PATTERNS, _PEOPLE, _THINGS, _PLACES, _TIMES)
            total = math.prod(map(len, pools))
            for i in range(total):
                index = (739 * i + 193) % total
                values = []
                for pool in pools:
                    index, j = divmod(index, len(pool))
                    values.append(pool[j])
                pattern, person, thing, place, time = values
                yield pattern.format(p=person, t=thing, l=place, w=time)
        source = chain(_LINES, variations())
    elif isinstance(source, (str, Path)):
        source = Path(source).read_text(encoding="utf-8-sig").splitlines()
    seen = set()
    for line in source:
        text = " ".join(line.split())
        words = tuple(_words(text))
        if not words or words in seen:
            continue
        if len(words) > 70:
            raise ValueError("Split custom prompts into utterances of at most 70 words.")
        seen.add(words)
        yield text


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path, rate):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Empty or invalid audio in {Path(path).name}.")
    mono = np.mean(audio, axis=1)
    if source_rate == rate:
        return mono
    divisor = math.gcd(rate, source_rate)
    return resample_poly(mono, rate // divisor, source_rate // divisor).astype(np.float32)


def _publish(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as folder:
        tmp = Path(folder) / path.name
        write(tmp)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)


def _json(path, obj):
    _publish(path, lambda temp: temp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8"))


def _wav(path, audio):
    import soundfile as sf
    _publish(path, lambda temp: sf.write(temp, audio, RATE, subtype="PCM_16"))


@contextmanager
def _locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".teacher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This voice already has a teacher job running.") from error
        yield


def _snapshot(local, env, repo, revision):
    directory = local or os.environ.get(env)
    if directory:
        path = Path(directory).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"Expected a model directory at {path}.")
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo, revision=revision,
                                  allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"]))


class QualityChecker:
    """Load local ASR and speaker verification only when the worker needs them."""

    def __init__(self, embedding_model=None, asr_dir=None):
        self.embedding_model = embedding_model
        self.asr_dir = asr_dir
        self._asr = self._extractor = None

    def transcribe_audio(self, audio16):
        if self._asr is None:
            from faster_whisper import WhisperModel
            directory = _snapshot(self.asr_dir, "VOICE_LAB_WHISPER_MODEL", ASR_REPO, ASR_REVISION)
            self._asr = WhisperModel(str(directory), device="cpu", compute_type="int8", cpu_threads=4)
        segments, _ = self._asr.transcribe(audio16, language="en", beam_size=3,
                                           condition_on_previous_text=False, temperature=0)
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe(self, path):
        return self.transcribe_audio(_read(path, 16000))

    def embedding(self, path):
        import sherpa_onnx
        from .beta_enrollment import _unit
        if self._extractor is None:
            if self.embedding_model is None:
                from .diarize import _models
                _, self.embedding_model = _models(Path.home() / ".cache" / "voice-lab" / "diarization", False)
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(self.embedding_model), num_threads=2)
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        stream = self._extractor.create_stream()
        stream.accept_waveform(16000, _read(path, 16000))
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            raise ValueError("The audio is too short to measure speaker similarity.")
        return _unit(self._extractor.compute(stream))

    def similarity(self, left, right):
        return float(self.embedding(left) @ self.embedding(right))


def _audio_digest(path):
    import numpy as np

    # PCM identity survives changes to a WAV container's headers or metadata.
    pcm = np.rint(np.clip(_read(path, RATE), -1, 1) * 32767).astype("<i2")
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def _disjoint(paths, holdouts):
    held = {_audio_digest(p) for p in holdouts}
    if held and any(_audio_digest(p) in held for p in paths):
        raise ValueError("Training and held-out audio must come from different recordings.")


def _reference_candidates(clips):
    import numpy as np
    import soundfile as sf

    candidates = []
    seen = set()
    for path, digest in clips:
        if digest in seen:
            continue
        seen.add(digest)
        info = sf.info(path)
        if info.duration > 120:
            raise ValueError("Use isolated training clips of two minutes or less.")
        audio = _read(path, RATE)
        frame = RATE // 50
        blocks = len(audio) // frame
        if blocks < 60:
            continue
        rms = np.sqrt(np.mean(audio[:blocks * frame].reshape(-1, frame) ** 2, axis=1))
        voiced = rms > max(.002, np.quantile(rms, .9) * .07)
        active = np.flatnonzero(voiced)
        if len(active) < 55:
            continue
        first, last = max(0, active[0] - 4), min(blocks, active[-1] + 5)
        if last - first > 700:
            pauses = np.flatnonzero(~voiced[first + 350:first + 600])
            last = first + (350 + pauses[-1] if len(pauses) else 600)
        excerpt = audio[first * frame:last * frame]
        density = float(np.mean(voiced[first:last]))
        if density >= .48 and np.mean(abs(excerpt) >= .998) <= .002:
            candidates.append((density, path, excerpt.copy(), digest))
    return sorted(candidates, key=lambda item: -item[0])


def prepare_reference(training_clips, directory, *, holdouts=(), asr_dir=None, checker=None, progress=None):
    """Select full clean excerpts, transcribe their actual audio and join them into one prompt."""
    import numpy as np

    paths = [Path(p) for p in training_clips]
    directory = Path(directory)
    _disjoint(paths, holdouts)
    clips = [(path, _hash_file(path)) for path in paths]
    checker = checker or QualityChecker(asr_dir=asr_dir)
    asr_path = getattr(checker, "asr_dir", None) or os.environ.get("VOICE_LAB_WHISPER_MODEL")
    inputs = {"version": 1, "clips": [digest for _, digest in clips], "asr_revision": ASR_REVISION,
              "asr_files": _model_key(Path(asr_path), ["model.bin"]) if asr_path else []}
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    reference = directory / "teacher-reference.wav"
    metadata = directory / "teacher-reference.json"
    chunks, texts, sources = [], [], []
    frames = 0
    gap = np.zeros(round(RATE * .16), dtype=np.float32)
    with _locked(directory), TemporaryDirectory(dir=directory) as scratch:
        if metadata.exists() and reference.exists():
            saved = json.loads(metadata.read_text(encoding="utf-8"))
            if saved.get("key") == key and saved.get("sha256") == _hash_file(reference):
                if progress:
                    progress("Reusing the verified reference and transcript")
                return reference, saved["text"]
        if progress:
            progress("Selecting and transcribing clean reference speech")
        for _, path, audio, digest in _reference_candidates(clips):
            if frames + len(audio) + len(gap) > 30 * RATE:
                continue
            temp = Path(scratch) / "excerpt.wav"
            _wav(temp, audio)
            transcript = checker.transcribe(temp)
            if len(_words(transcript)) < 2:
                continue
            if chunks:
                chunks.append(gap)
                frames += len(gap)
            chunks.append(audio)
            frames += len(audio)
            texts.append(transcript)
            sources.append({"file": path.name, "sha256": digest, "text": transcript})
            if frames >= 26 * RATE:
                break
        if frames < 3 * RATE:
            raise ValueError("Collect at least three seconds of clear, transcribable reference speech.")
        combined, transcript = np.concatenate(chunks), " ".join(texts)
        _wav(reference, combined)
        _json(metadata, {"key": key, "sha256": _hash_file(reference), "text": transcript,
                         "seconds": frames / RATE, "sources": sources})
    return reference, transcript

class _Qwen:
    def __init__(self, model_dir, reference, transcript):
        from mlx_audio.tts.utils import load_model
        self.model = load_model(str(model_dir))
        self.set_reference(reference, transcript)

    def set_reference(self, reference, transcript):
        import mlx.core as mx
        self.audio = mx.array(_read(reference, RATE))
        self.transcript = transcript

    def render(self, text, seed):
        import mlx.core as mx
        import numpy as np

        mx.random.seed(seed)
        outputs = self.model.generate(
            text=text, ref_audio=self.audio, ref_text=self.transcript, lang_code="English",
            max_tokens=min(900, max(128, len(_words(text)) * 11 + 32)),
            temperature=.55, top_p=.9, stream=False, verbose=False,
        )
        chunks = []
        for output in outputs:
            if output.sample_rate != RATE:
                raise ValueError(f"Unexpected teacher output rate {output.sample_rate}.")
            chunks.append(np.asarray(output.audio, dtype=np.float32).reshape(-1))
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


def _screen(checker, reference_embedding, text, audio, destination):
    import numpy as np

    seconds = len(audio) / RATE
    words = _words(text)
    if not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) < .005:
        return {"reason": "empty or silent generation"}
    if seconds < max(.45, len(words) * .10) or seconds > max(5, len(words) * 1.15):
        return {"reason": "implausible speech duration", "seconds": seconds}
    peak = float(np.max(np.abs(audio)))
    if np.mean(np.abs(audio) >= .999) > .002:
        return {"reason": "excessive clipping"}
    _wav(destination, audio / max(1.0, peak / .98))
    heard = checker.transcribe(destination)
    wer = word_error_rate(text, heard)
    score = float(reference_embedding @ checker.embedding(destination))
    result = {"seconds": round(seconds, 3), "heard": heard, "wer": round(wer, 4),
              "reference_cosine": round(score, 4)}
    if not math.isfinite(score) or score < MIN_COSINE:
        result["reason"] = "speaker similarity below threshold"
    elif wer > MAX_WER or len(_words(heard)) > len(words) + 1:
        result["reason"] = "transcript mismatch"
    return result


def _model_key(directory, filenames):
    # Include local overrides in cache identity, even if the name is unchanged.
    return [(name, (directory / name).stat().st_size, (directory / name).stat().st_mtime_ns)
            for name in filenames if (directory / name).is_file()]


def generate_corpus(directory, reference, reference_text, *, holdouts=(), count=DEFAULT_COUNT, prompts=None,
                    model_dir=None, asr_dir=None, embedding_model=None, checker=None, progress=None, excluded_texts=(),
                    adapter=None, freeze_manifest=False):
    """Publish accepted, paired utterances. Per-item results survive interruption and larger repeat runs."""
    if not isinstance(count, int) or count < 3:
        raise ValueError("Request at least three accepted lines for training and validation.")
    if not _words(reference_text):
        raise ValueError("Provide the transcript of the teacher reference.")
    directory, reference = Path(directory), Path(reference)
    _disjoint([reference], holdouts)
    model_dir = _snapshot(model_dir, "VOICE_LAB_QWEN_MODEL", QWEN_REPO, QWEN_REVISION)
    checker = checker or QualityChecker(embedding_model=embedding_model, asr_dir=asr_dir)
    reference_embedding = checker.embedding(reference)
    asr_path = asr_dir or os.environ.get("VOICE_LAB_WHISPER_MODEL")
    embedding_path = getattr(checker, "embedding_model", None)
    recipe = {"version": 1, "reference_sha256": _hash_file(reference), "reference_text": reference_text,
              "teacher": QWEN_REPO, "teacher_revision": QWEN_REVISION,
              "teacher_files": _model_key(model_dir, ["model.safetensors", "speech_tokenizer/model.safetensors"]),
              "asr_revision": ASR_REVISION, "asr_files": _model_key(Path(asr_path), ["model.bin"]) if asr_path else [],
              "embedding_sha256": _hash_file(embedding_path) if embedding_path else None,
              "min_cosine": MIN_COSINE, "max_wer": MAX_WER}
    if adapter is not None:
        recipe["acoustic_adapter_sha256"] = _hash_file(Path(adapter))
    recipe_key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    cache = directory / "teacher-cache" / recipe_key[:24]
    manifest = directory / "manifest.jsonl"
    teacher, rows = None, []
    excluded = {tuple(_words(line)) for line in chain((reference_text,), excluded_texts)}
    consecutive_rejections = 0
    with _locked(directory), TemporaryDirectory(dir=directory) as scratch:
        cache.mkdir(parents=True, exist_ok=True)
        _json(cache / "recipe.json", recipe)
        for text in _prompts(prompts):
            if tuple(_words(text)) in excluded:
                continue
            key = hashlib.sha256(text.encode()).hexdigest()[:24]
            record, wav = cache / (key + ".json"), cache / (key + ".wav")
            metadata = json.loads(record.read_text()) if record.exists() else {}
            if "accepted" in metadata and metadata["accepted"]:
                if not wav.is_file() or _hash_file(wav) != metadata["sha256"]:
                    metadata = {}
            if not metadata:
                if teacher is None:
                    if progress:
                        progress("Loading the Qwen speech teacher")
                    teacher = _Qwen(model_dir, reference, reference_text)
                    if adapter is not None:
                        from .qwen_adapt import apply_adapter
                        apply_adapter(teacher, adapter)
                temp = Path(scratch) / "candidate.wav"
                attempts = []
                for attempt in range(2):
                    seed = (int(key[:8], 16) + attempt) % (2 ** 31)
                    audio = teacher.render(text, seed)
                    stats = _screen(checker, reference_embedding, text, audio, temp)
                    attempts.append(dict(seed=seed, **stats))
                    if "reason" not in stats:
                        os.replace(temp, wav)
                        break
                metadata = dict(text=text, accepted="reason" not in attempts[-1], attempts=attempts)
                if metadata["accepted"]:
                    metadata["sha256"] = _hash_file(wav)
                _json(record, metadata)
            if metadata["accepted"]:
                # Ensure even a three-line pilot has a disjoint validation text.
                split = "validation" if len(rows) % 8 == 1 else "train"
                rows.append(dict(text=text, audio=str(wav.relative_to(directory)), split=split))
                consecutive_rejections = 0
            else:
                consecutive_rejections += 1
            if progress:
                progress(f"{len(rows)} of {count} teacher lines passed speech and transcript checks")
            if len(rows) == count:
                break
            if consecutive_rejections >= 16:
                raise ValueError(f"Sixteen consecutive lines failed QA. Check the reference and its transcript. See {cache}.")
        if len(rows) != count:
            raise ValueError(f"Only {len(rows)} of {count} requested lines passed QA. Add more distinct prompts or use a cleaner reference.")
        content = "".join(json.dumps(row) + "\n" for row in rows)
        _publish(manifest, lambda temp: temp.write_text(content, encoding="utf-8"))
        if freeze_manifest:
            # Keep relative audio paths valid while freezing this exact teacher and split.
            digest = hashlib.sha256(content.encode()).hexdigest()[:16]
            manifest = directory / f"accepted-{count}-{digest}.jsonl"
            _publish(manifest, lambda temp: temp.write_text(content, encoding="utf-8"))
    return manifest
