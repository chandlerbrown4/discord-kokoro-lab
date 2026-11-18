"""Accumulate isolated speaker turns across completed call chunks and fit local voice profiles."""

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import wave
from collections import OrderedDict
from contextlib import closing
from pathlib import Path

from . import diarize, vc


RATE = 24000
REQUIRED_SECONDS = 30
REQUIRED_CLIPS = 6
MAX_CLIPS = 64
TEXT_UPDATE_CLIPS = 6
SIGNATURE_CACHE_SIZE = 64
MATCH = .65
NEW_SPEAKER = .58
MARGIN = .06
CONSISTENCY = .66


def _unit(vec):
    import numpy as np

    vec = np.asarray(vec, dtype="float32")
    norm = np.linalg.norm(vec)
    if vec.ndim != 1 or not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("Speaker analysis returned an invalid voice embedding.")
    return vec / norm


def _write_wav(path, samples):
    """Publish a fully written immutable PCM file, including on concurrent materialization."""
    import numpy as np

    if path.exists():
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with wave.open(str(tmp_path), "wb") as out:
            out.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            out.writeframes(np.asarray(samples, dtype="<i2").tobytes())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _clean_turn(samples):
    """Take a contiguous, active excerpt while retaining its natural pauses and dynamics."""
    import numpy as np

    frame = RATE // 50
    n = len(samples) // frame
    if n < 75:
        return None
    rms = np.sqrt((samples[:n*frame].reshape(n, frame) ** 2).mean(axis=1))
    active = rms >= max(.003, float(np.quantile(rms, .90)) * .08)
    # Retain one excerpt per independent diarized turn. A long monologue cannot
    # supply every training and validation example via adjacent slices.
    if n > 400:
        start = int(np.argmax(np.convolve(active.astype(int), np.ones(400, int), mode="valid")))
        samples = samples[start*frame:(start + 400)*frame]
        active = active[start:start + 400]
    voiced = np.flatnonzero(active)
    if len(voiced) < 65:
        return None
    first, end = max(0, voiced[0] - 4), min(len(active), voiced[-1] + 5)
    x = samples[first*frame:end*frame]
    if len(voiced) < .50 * (end-first) or np.mean(np.abs(x) >= .998) > .001:
        return None
    gain = min(6.0, .82 / max(float(np.max(np.abs(x))), 1e-6))
    return np.rint(x * gain * 32767).astype("<i2"), len(voiced) / 50


def _split(rows):
    # A fixed partition by insertion order keeps validation takes out of model
    # fitting and reference preparation as the recording grows.
    train = [row for i, row in enumerate(rows) if i % 4 != 1]
    held = [row for i, row in enumerate(rows) if i % 4 == 1]
    return train, held


def _coherent_groups(groups):
    """A short chunk's diarizer label may still contain turns from two voices."""
    import numpy as np

    for turns in groups.values():
        clusters = []
        for turn in turns:
            emb = turn[2]
            scores = sorted(((float(emb @ _unit(np.mean([t[2] for t in cluster], axis=0))), i)
                             for i, cluster in enumerate(clusters)), reverse=True)
            if not scores or scores[0][0] < NEW_SPEAKER:
                clusters.append([turn])
            elif scores[0][0] >= MATCH and (len(scores) == 1 or scores[0][0] - scores[1][0] >= MARGIN):
                clusters[scores[0][1]].append(turn)
        yield from clusters


def _audio_signature(pcm):
    """A low-rate waveform keeps repeat checks inexpensive across slightly shifted cuts."""
    import numpy as np
    from scipy.signal import butter, sosfilt

    # A low-pass below the signature's Nyquist frequency also makes sub-sample
    # movement of the same utterance at different chunk offsets comparable.
    centered = np.asarray(pcm, dtype="float32") - np.mean(pcm, dtype="float32")
    values = sosfilt(butter(4, 500, fs=RATE, output="sos"), centered)[::8].astype("float32")
    return values - values.mean()


def _same_excerpt(left, right):
    import numpy as np
    from scipy.signal import correlate

    short, long = sorted((left, right), key=len)
    # Different diarizer boundaries can add/remove a few low-energy syllables.
    # Use normalized waveform agreement over the interior, not voice similarity.
    n = len(short) // 10
    short = short[n:len(short)-n]
    magnitude = np.linalg.norm(short)
    if magnitude < 1 or len(short) < 3000:
        return False
    long = np.pad(long, 480)
    energy = np.concatenate(([0.], np.cumsum(long.astype("float64") ** 2)))
    denominator = np.sqrt(np.maximum(energy[len(short):] - energy[:-len(short)], 1.)) * magnitude
    return np.max(correlate(long, short, mode="valid", method="fft") / denominator) > .93


def _summary(rows):
    import numpy as np

    seconds = sum(row["voiced"] for row in rows)
    matrix = np.stack([np.frombuffer(row["embedding"], dtype="<f4") for row in rows])
    similarity = None
    if len(rows) > 1:
        others = matrix.sum(axis=0) - matrix
        similarity = float(np.median((others * matrix).sum(axis=1) /
                                     np.maximum(np.linalg.norm(others, axis=1), 1e-6)))
    ready = (len(rows) >= REQUIRED_CLIPS and seconds >= REQUIRED_SECONDS and similarity >= CONSISTENCY)
    return seconds, similarity, ready


def _training_set(rows):
    import numpy as np

    train, held = _split(rows)
    # Cover early and later evidence when retrying with more speech. Validation
    # always stays in its original partition, even after a reference-only fit.
    indices = np.linspace(0, len(train)-1, min(len(train), 12), dtype=int)
    return [train[i] for i in indices], held[:3]


class Enrollment:
    """One call's persistent speaker inventory. Model downloads are optional and lazy."""

    def __init__(self, session_dir, model_dir, *, local_only=True):
        self.root = Path(session_dir).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.files = self.root / "voices"
        self.files.mkdir(mode=0o700, exist_ok=True)
        self.database = self.root / "enrollment.sqlite3"
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.local_only = local_only
        self._model_lock = threading.RLock()
        self._detector = self._extractor = None
        self._signatures = OrderedDict()
        with closing(self._db()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (digest TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS clips (
                    id INTEGER PRIMARY KEY, speaker INTEGER NOT NULL, digest TEXT NOT NULL UNIQUE,
                    voiced REAL NOT NULL, embedding BLOB NOT NULL);
                CREATE INDEX IF NOT EXISTS by_speaker ON clips(speaker, id);
                CREATE TABLE IF NOT EXISTS text_updates (
                    speaker INTEGER PRIMARY KEY, enabled INTEGER NOT NULL, attempted_count INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS models (
                    speaker INTEGER PRIMARY KEY, count INTEGER NOT NULL, reference TEXT NOT NULL,
                    adapter TEXT, validation TEXT NOT NULL);
            """)

    def _db(self):
        db = sqlite3.connect(self.database, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def _load_models(self):
        import sherpa_onnx as sherpa

        if self._detector is None:
            seg, emb = diarize._models(self.model_dir, self.local_only)
            detector = diarize._diarizer(sherpa, seg, emb, None)
            extractor = sherpa.SpeakerEmbeddingExtractor(
                sherpa.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=2))
            self._detector, self._extractor = detector, extractor

    def _embed(self, audio16):
        with self._model_lock:
            self._load_models()
            stream = self._extractor.create_stream()
            stream.accept_waveform(diarize.MODEL_RATE, audio16)
            stream.input_finished()
            if not self._extractor.is_ready(stream):
                raise ValueError("The speech excerpt is too short to identify its speaker.")
            return _unit(self._extractor.compute(stream))

    def _analyze(self, chunk, *, imported=False):
        import numpy as np
        from scipy.signal import resample_poly

        # The native recorder publishes finite, completed files. The time cap
        # prevents an accidental full-length recording exhausting RAM.
        pcm = vc.read_audio(chunk, max_seconds=120, rate=RATE)
        if len(pcm) < RATE or np.max(np.abs(pcm)) < .003:
            return {}
        x16 = resample_poly(pcm, 2, 3)
        with self._model_lock:
            self._load_models()
            labels = self._detector.process(x16).sort_by_start_time()
        turns, _ = diarize._exclusive_turns(labels, 2 * len(pcm))
        groups = {}
        excerpts = []
        for start, end, label in turns:
            start, end = start // 2, end // 2
            # Split a long imported utterance at its quietest nearby boundary.
            # Adjacent windows have no samples in common. Live capture retains
            # its stricter, one-excerpt-per-turn admission rule.
            while imported and end - start > 7 * RATE:
                region = pcm[start + 4*RATE:start + 6*RATE].reshape(-1, RATE//50)
                position = int(np.argmin((region ** 2).mean(axis=1)))
                cut = start + 4*RATE + position*(RATE//50)
                excerpts.append((start, cut, label))
                start = cut
            excerpts.append((start, end, label))
        for start, end, label in excerpts:
            cleaned = _clean_turn(pcm[start:end])
            if cleaned is None:
                continue
            audio, seconds = cleaned
            emb = self._embed(resample_poly(audio.astype("float32") / 32768, 2, 3))
            groups.setdefault(label, []).append((audio, seconds, emb))
        return groups

    def _is_duplicate(self, audio, emb, rows):
        import numpy as np

        signature = _audio_signature(audio)
        for position, row in enumerate(rows):
            if float(np.frombuffer(row["embedding"], dtype="<f4") @ emb) < .85:
                continue
            digest = row["digest"]
            previous = self._signatures.get(digest)
            if previous is None:
                with wave.open(str(self.files / (digest + ".wav")), "rb") as f:
                    previous = _audio_signature(np.frombuffer(f.readframes(f.getnframes()), dtype="<i2"))
                # Retain recent turns and avoid a sequential scan of old history
                # flushing the working set; all older WAVs are still checked.
                if position < SIGNATURE_CACHE_SIZE:
                    self._signatures[digest] = previous
                    if len(self._signatures) > SIGNATURE_CACHE_SIZE:
                        self._signatures.popitem(last=False)
            else:
                self._signatures.move_to_end(digest)
            if _same_excerpt(signature, previous):
                return True
        return False

    def ingest(self, chunk, *, imported=False):
        """Process one completed WAV; content hashes make retries idempotent."""
        import numpy as np

        path = Path(chunk)
        h = hashlib.sha256()
        with path.open("rb") as src:
            for block in iter(lambda: src.read(1 << 20), b""):
                h.update(block)
        digest = h.hexdigest()
        with closing(self._db()) as db:
            if db.execute("SELECT 1 FROM chunks WHERE digest=?", (digest,)).fetchone():
                return self.snapshot()
        groups = self._analyze(path, imported=True) if imported else self._analyze(path)
        with closing(self._db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("INSERT OR IGNORE INTO chunks VALUES (?)", (digest,)).rowcount:
                voices = {}
                for row in db.execute("SELECT speaker, embedding FROM clips ORDER BY id"):
                    voices.setdefault(row["speaker"], []).append(np.frombuffer(row["embedding"], dtype="<f4"))
                completed = {r["speaker"] for r in db.execute("SELECT speaker,validation FROM models")
                             if json.loads(r["validation"]).get("passes_quality", True)}
                growing = {r[0] for r in db.execute("SELECT speaker FROM text_updates WHERE enabled=1")}
                claimed = set()
                for turns in _coherent_groups(groups):
                    local = _unit(np.mean([t[2] for t in turns], axis=0))
                    clean = [t for t in turns if float(t[2] @ local) >= MATCH]
                    if not clean:
                        continue
                    scores = sorted(((float(local @ _unit(np.mean(vectors, axis=0))), speaker)
                                     for speaker, vectors in voices.items()), reverse=True)
                    if (not scores or scores[0][0] < NEW_SPEAKER or
                            (scores[0][1] in claimed and scores[0][0] < MATCH)):
                        speaker = max(voices, default=0) + 1
                    else:
                        best, speaker = scores[0]
                        if (best < MATCH or speaker in claimed or
                                (len(scores) > 1 and best - scores[1][0] < MARGIN)):
                            continue
                    claimed.add(speaker)
                    existing = voices.get(speaker, [])
                    centroid = _unit(np.mean(existing, axis=0)) if existing else local
                    for audio, seconds, emb in clean:
                        if (speaker in completed and speaker not in growing and len(existing) >= MAX_CLIPS
                                or float(emb @ centroid) < MATCH):
                            continue
                        key = hashlib.sha256(audio.tobytes()).hexdigest()
                        if db.execute("SELECT 1 FROM clips WHERE digest=?", (key,)).fetchone():
                            continue
                        prior = db.execute("SELECT digest, embedding FROM clips WHERE speaker=? ORDER BY id DESC", (speaker,))
                        if self._is_duplicate(audio, emb, prior):
                            continue
                        result = db.execute("INSERT OR IGNORE INTO clips(speaker,digest,voiced,embedding) "
                                            "VALUES (?,?,?,?)", (speaker, key, seconds, emb.astype("<f4").tobytes()))
                        if result.rowcount:
                            _write_wav(self.files / (key + ".wav"), audio)
                            existing.append(emb)
                    if existing:
                        voices[speaker] = existing
        return self.snapshot()

    @staticmethod
    def _rows(db, speaker):
        if not str(speaker).isascii() or not str(speaker).isdigit() or len(str(speaker)) > 12:
            raise ValueError("Choose a detected speaker.")
        rows = list(db.execute("SELECT * FROM clips WHERE speaker=? ORDER BY id", (int(speaker),)))
        if not rows:
            raise ValueError("Choose a detected speaker.")
        return rows

    def snapshot(self):
        with closing(self._db()) as db:
            db.execute("BEGIN")
            speakers = []
            for (speaker,) in db.execute("SELECT DISTINCT speaker FROM clips ORDER BY speaker"):
                rows = self._rows(db, speaker)
                total, consistency, ready = _summary(rows)
                watch = db.execute("SELECT enabled,attempted_count FROM text_updates WHERE speaker=?", (speaker,)).fetchone()
                training_count = len(_split(rows)[0])
                model = db.execute("SELECT * FROM models WHERE speaker=?", (speaker,)).fetchone()
                status = "ready_to_fit" if ready else "collecting"
                validation = None
                passed = False
                if model:
                    validation = json.loads(model["validation"])
                    passed = validation.get("passes_quality", True)
                    status = ("optimized" if model["adapter"] else "reference_only") if passed else "needs_more_audio"
                speakers.append(dict(id=str(speaker), name=f"Speaker {speaker}", usable_seconds=round(total, 2),
                                     clip_count=len(rows), required_seconds=REQUIRED_SECONDS, required_clips=REQUIRED_CLIPS,
                                     consistency=round(consistency, 4) if consistency is not None else None,
                                     model_status=status, ready=passed if model else ready, text_eligible=ready,
                                     needs_fit=ready and (not model or len(rows) >= model["count"] + (12 if passed else 6)),
                                     training_count=training_count, auto_update=bool(watch and watch["enabled"]),
                                     new_training_clips=max(0, training_count - watch["attempted_count"]) if watch else 0,
                                     update_after_clips=TEXT_UPDATE_CLIPS, validation=validation))
            return {"speakers": speakers, "processed_chunks": db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]}

    def sample(self, speaker_id):
        with closing(self._db()) as db:
            rows = self._rows(db, speaker_id)
        row = max(rows, key=lambda r: r["voiced"])
        return self.files / (row["digest"] + ".wav")

    def _reference(self, train):
        import numpy as np

        center = _unit(np.mean([np.frombuffer(r["embedding"], dtype="<f4") for r in train], axis=0))
        ranked = sorted(train, key=lambda r: float(np.frombuffer(r["embedding"], dtype="<f4") @ center), reverse=True)
        selected, frames = [], []
        gap = np.zeros(round(RATE * .15), dtype="<i2")
        total = 0
        for row in ranked:
            with wave.open(str(self.files / (row["digest"] + ".wav")), "rb") as f:
                size = f.getnframes()
                if total + size + len(gap) > 29 * RATE:
                    continue
                audio = np.frombuffer(f.readframes(size), dtype="<i2")
            frames.extend((audio, gap))
            selected.append(row["digest"])
            total += size + len(gap)
        key = hashlib.sha256(" ".join(selected).encode("ascii")).hexdigest()
        path = self.files / ("reference-" + key + ".wav")
        _write_wav(path, np.concatenate(frames[:-1]))
        return path

    def reference(self, speaker_id):
        with closing(self._db()) as db:
            rows = self._rows(db, speaker_id)
            model = db.execute("SELECT reference FROM models WHERE speaker=?", (int(speaker_id),)).fetchone()
            if model:
                return self.files / model[0]
        if not _summary(rows)[2]:
            raise ValueError("Collect more clean speech before preparing this voice.")
        return self._reference(_training_set(rows)[0])

    def adapter(self, speaker_id):
        with closing(self._db()) as db:
            self._rows(db, speaker_id)
            model = db.execute("SELECT adapter FROM models WHERE speaker=?", (int(speaker_id),)).fetchone()
        return self.files / model[0] if model and model[0] else None

    def selected_files(self, speaker_id):
        """Read the reference and optional adapter from the same published fit."""
        with closing(self._db()) as db:
            model = db.execute("SELECT reference,adapter FROM models WHERE speaker=?", (speaker_id,)).fetchone()
        if not model:
            raise ValueError("The voice profile is still being prepared.")
        return self.files / model["reference"], self.files / model["adapter"] if model["adapter"] else None

    def ensure_text_watch(self, speaker_id, previous=()):
        """Migrate an intentionally built voice once. User preferences survive subsequent launches."""
        with closing(self._db()) as db, db:
            if db.execute("SELECT 1 FROM text_updates WHERE speaker=?", (speaker_id,)).fetchone():
                return
            rows = self._rows(db, speaker_id)
            # Earlier builds sampled training rows up to and including their newest
            # one. A copied library retains the content hashes in its file names.
            known = {Path(path).stem for path in previous}
            last = max((i + 1 for i, row in enumerate(rows) if row["digest"] in known), default=len(rows))
            count = len(_split(rows[:last])[0])
            db.execute("INSERT OR IGNORE INTO text_updates VALUES (?,1,?)", (speaker_id, count))

    def set_text_updates(self, speaker_id, enabled):
        if type(enabled) is not bool:
            raise ValueError("Choose whether to update this speaker automatically.")
        with closing(self._db()) as db, db:
            if not db.execute("UPDATE text_updates SET enabled=? WHERE speaker=?",
                              (int(enabled), speaker_id)).rowcount:
                raise ValueError("Build this speaker’s text voice first.")

    def mark_text_attempt(self, speaker_id, training_count):
        with closing(self._db()) as db, db:
            db.execute("UPDATE text_updates SET attempted_count=MAX(attempted_count,?) WHERE speaker=?",
                       (training_count, speaker_id))

    def text_watches(self):
        """Read only counters for background scheduling across closed call libraries."""
        with closing(self._db()) as db:
            rows = db.execute("""
                SELECT w.speaker,w.attempted_count,COUNT(c.id) AS total
                FROM text_updates w JOIN clips c ON c.speaker=w.speaker
                WHERE w.enabled=1 GROUP BY w.speaker
            """)
            return [(str(row["speaker"]), row["total"] - (row["total"] + 2) // 4, row["attempted_count"])
                    for row in rows]

    def text_inputs(self, speaker_id):
        with closing(self._db()) as db:
            rows = self._rows(db, speaker_id)
            if not _summary(rows)[2]:
                raise ValueError("Collect 30 seconds across six consistent, clean speech excerpts first.")
            train, held = _split(rows)
            held = held[:3]
        def paths(group):
            return [str(self.files / (row["digest"] + ".wav")) for row in group]
        return {"training_clips": paths(train), "heldout_clips": paths(held)}

    def training_clips(self, speaker_id):
        with closing(self._db()) as db:
            train, _ = _training_set(self._rows(db, speaker_id))
        return [self.files / (row["digest"] + ".wav") for row in train]

    def score_preview(self, speaker_id, converted_wav):
        """Score the user's converted mic against genuine clips withheld from the fit."""
        import numpy as np

        with closing(self._db()) as db:
            rows = self._rows(db, speaker_id)
            model = db.execute("SELECT count FROM models WHERE speaker=?", (int(speaker_id),)).fetchone()
        if not model:
            raise ValueError("Prepare this speaker before scoring a converted microphone take.")
        train, held = _training_set(rows[:model[0]])
        held_center = _unit(np.mean([np.frombuffer(r["embedding"], dtype="<f4") for r in held], axis=0))
        genuine = [float(np.frombuffer(r["embedding"], dtype="<f4") @ held_center) for r in train]
        x = vc.read_audio(converted_wav, rate=diarize.MODEL_RATE)
        if len(x) < diarize.MODEL_RATE or np.max(np.abs(x)) < .003:
            raise ValueError("The mic preview needs at least a second of audible speech to score.")
        score = float(self._embed(x) @ held_center)
        return {"metric": "WeSpeaker cosine to withheld genuine turns", "score": score,
                "genuine": float(np.mean(genuine)), "sample_count": len(held)}

    def fit(self, speaker_id, *, device="cpu", weights=None, threads=2):
        import numpy as np
        from .beta_fit import fit_target

        with closing(self._db()) as db:
            rows = self._rows(db, speaker_id)
            previous = db.execute("SELECT * FROM models WHERE speaker=?", (int(speaker_id),)).fetchone()
        if previous and previous["count"] == len(rows):
            return self.snapshot()
        if not _summary(rows)[2]:
            raise ValueError("More consistent, isolated speech is needed before fitting.")
        train, held = _training_set(rows)
        reference = self._reference(train)
        paths = [self.files / (r["digest"] + ".wav") for r in train]
        validation_paths = [self.files / (r["digest"] + ".wav") for r in held]
        center = _unit(np.mean([np.frombuffer(r["embedding"], dtype="<f4") for r in train], axis=0))
        old_target = None
        if previous:
            old_target = (self.files / previous["reference"],
                          self.files / previous["adapter"] if previous["adapter"] else None)
        adapter, validation = fit_target(reference, paths, validation_paths, self._embed,
                                         comparison_center=center, previous=old_target, device=device,
                                         weights=weights, threads=threads)
        adapter_name = None
        if validation.get("retained_previous") and previous:
            reference, adapter_name = old_target[0], previous["adapter"]
        elif adapter is not None:
            adapter_name = "adapter-" + hashlib.sha256(adapter.tobytes()).hexdigest() + ".npy"
            with tempfile.NamedTemporaryFile(dir=self.files, delete=False) as tmp:
                temp_path = Path(tmp.name)
                try:
                    np.save(tmp, adapter, allow_pickle=False)
                    tmp.flush()
                    os.replace(tmp.name, self.files / adapter_name)
                finally:
                    temp_path.unlink(missing_ok=True)
        with closing(self._db()) as db, db:
            db.execute("INSERT INTO models VALUES(?,?,?,?,?) ON CONFLICT(speaker) DO UPDATE SET "
                       "count=excluded.count, reference=excluded.reference, adapter=excluded.adapter, "
                       "validation=excluded.validation WHERE excluded.count >= models.count",
                       (int(speaker_id), len(rows), reference.name, adapter_name, json.dumps(validation, allow_nan=False)))
        return self.snapshot()
