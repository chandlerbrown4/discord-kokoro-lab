"""Per-speaker text voices. Only worker processes load the optional speech models."""
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time

from .distill_jobs import locate, read, write

RECIPES = {"preview": (12, 24), "full": (128, 128)}
RETRY_DELAYS = (30, 120)


def _command(kind, stage, directory, model_dir):
    interpreter = os.environ.get("VOICE_LAB_" + kind.upper() + "_PYTHON")
    if interpreter:
        command = [interpreter]
    else:
        script = os.environ.get("VOICE_LAB_TEXT_RUNTIME")
        if not script:
            script = Path(__file__).resolve().parents[2] / "packaging/macos/text-runtime.sh"
        if not Path(script).is_file():
            raise RuntimeError("Install the Mac beta or set the optional teacher and student Python paths.")
        command = ["/bin/bash", str(script), kind]
    return command + ["-m", "voice_lab.distill_jobs", stage, str(directory), str(model_dir)]


def _remaining_stages(directory, cache, count, steps):
    """A resumed job can repair missing products without reinstalling or retraining a healthy voice."""
    profile = directory / "student" / "profile.pt"
    reviews = directory / "review-lines.json"
    have_lines = reviews.is_file() and bool(read(reviews))
    synthesized = profile.is_file() and have_lines and all(
        (directory / f"review-{i}.wav").is_file() for i in range(len(read(reviews))))
    if synthesized and (directory / "review.wav").is_file() and (directory / "evaluation.json").is_file():
        return
    if not synthesized:
        marker = directory / "manifest-path.json"
        manifest = locate(directory, read(marker)) if marker.is_file() else None
        valid = have_lines and manifest is not None and manifest.is_file()
        if valid and not profile.is_file():
            pairs = [json.loads(line) for line in manifest.read_text().splitlines() if line]
            valid = len(pairs) >= count and all((manifest.parent / row["audio"]).is_file() for row in pairs)
        if not valid:
            yield ("teacher", "corpus", ["--count", str(count), "--cache", str(cache)], "Preparing teaching examples…")
        yield ("student", "student", ["--steps", str(steps)], "Fitting a compact Kokoro voice…")
    yield ("teacher", "evaluate", [], "Checking separate original speech…")


def _recovers_failed_identity(prior, candidate):
    """Allow a modest pitch-range trade-off only when every held-out line recovers the voice."""
    if prior.get("passes_quality") is not False or candidate.get("passes_quality") is not True:
        return False
    old, new = prior["lines"], candidate["lines"]
    # Require paired, distinct genuine recordings, reliable pitch and complete measurements.
    hashes = [row.get("original_sha256") for row in old]
    if len(old) < 2 or not all(hashes) or len(set(hashes)) != len(old):
        return False
    fields = ("similarity", "word_error_rate", "duration_log_error", "pitch_center_error_st", "pitch_span_error_st")
    timing = []
    for a, b in zip(old, new):
        if a["original_sha256"] != b.get("original_sha256") or any(
                a.get(field) is None or b.get(field) is None for field in fields):
            return False
        delta = b["duration_log_error"] - a["duration_log_error"]
        if not (b["similarity"] >= .80 and b["similarity"] - a["similarity"] >= .15
                and b["word_error_rate"] <= a["word_error_rate"]
                and delta <= .025 and b["duration_log_error"] <= .20
                and b["pitch_center_error_st"] <= 3. and a["pitch_center_error_st"] - b["pitch_center_error_st"] >= 3.
                and b["pitch_span_error_st"] <= 4.):
            return False
        timing.append(delta)
    return statistics.mean(timing) <= .01


def _prefer_update(prior, previous_dir, candidate, candidate_dir):
    """Compare only paired original utterances. Close or conflicting results keep the incumbent."""
    old, new = prior.get("lines", []), candidate.get("lines", [])
    if not old or len(old) != len(new):
        return False, "The two builds have different evaluation recordings."
    for a, b in zip(old, new):
        if a.get("original_sha256") and b.get("original_sha256"):
            same_audio = a["original_sha256"] == b["original_sha256"]
        else:
            same_audio = locate(previous_dir, a["original"]) == locate(candidate_dir, b["original"])
        if not same_audio or a["text"].casefold().split() != b["text"].casefold().split():
            return False, "The two builds have different evaluation recordings or transcripts."
    if prior.get("passes_quality") and not candidate.get("passes_quality"):
        return False, "The new build missed the voice quality check."
    gain = candidate["mean_cosine"] - prior["mean_cosine"]
    clearer = prior["mean_word_error_rate"] - candidate["mean_word_error_rate"]
    if (gain < -.005 or clearer < -.01 or
            any(b.get("similarity", candidate["mean_cosine"]) < a.get("similarity", prior["mean_cosine"]) - .025
                for a, b in zip(old, new)) or
            any(b.get("word_error_rate", 0) > a.get("word_error_rate", 0) + .10 for a, b in zip(old, new))):
        return False, "Likeness or intelligibility declined on the same original lines."
    better_delivery = False
    recovered = False
    # Log duration is dimensionless. Pitch center and range are measured in semitones.
    for metric, tolerance, large in (("duration_log_error", .06, .18), ("pitch_center_error_st", 1., 3.),
                                     ("pitch_span_error_st", 1., 3.)):
        if any(a.get(metric) is not None and b.get(metric) is None for a, b in zip(old, new)):
            return False, "The new build lost a measurable pitch or timing comparison on an original line."
        deltas = [b[metric] - a[metric] for a, b in zip(old, new)
                  if a.get(metric) is not None and b.get(metric) is not None]
        if deltas:
            if statistics.mean(deltas) > tolerance or max(deltas) > large:
                if (metric == "pitch_span_error_st" and statistics.mean(deltas) <= 2. and max(deltas) <= 4.
                        and _recovers_failed_identity(prior, candidate)):
                    recovered = True
                else:
                    return False, "Timing or pitch moved further from the original delivery."
            if statistics.mean(deltas) < -tolerance:
                better_delivery = True
    if recovered:
        return True, "The new build restored speaker likeness and pitch center on every held-out line."
    if gain >= .005 or clearer >= .02 or better_delivery:
        return True, "The new build improved on the same held-out lines."
    return False, "The same-line comparison did not show a clear improvement."


def _stop(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)
    except ProcessLookupError:
        pass


class TextVoices:
    def __init__(self, model_dir, local_only=False):
        self.model_dir = model_dir
        self.env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
        source = str(Path(__file__).resolve().parents[1])
        self.env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([source, *filter(None, self.env.get("PYTHONPATH", "").split(os.pathsep))]))
        if local_only:
            self.env.update(HF_HUB_OFFLINE="1", UV_OFFLINE="1")
        self.lock = threading.RLock()
        self.job = self.process = self.warm = self.warm_directory = self.timer = self.thread = None

    @staticmethod
    def _root(call, sid):
        if not str(sid).isascii() or not str(sid).isdecimal():
            raise ValueError("Choose a saved speaker.")
        return call / "text-voices" / str(sid)

    def _state(self, root):
        path = root / "state.json"
        state = read(path) if path.is_file() else {}
        if state.get("state") in ("building", "rendering") and self.job != root:
            if state["state"] == "building" and state.get("build") and not state.get("pending_update"):
                state["pending_update"] = {"build": state["build"], "mode": state["mode"]}
            state.update(state="paused", message="The previous job stopped. Resume the build or generate a new line.")
        return state

    def snapshot(self, call, speakers):
        with self.lock:
            return {s["id"]: state for s in speakers
                    if (state := self._state(self._root(call, s["id"])))}

    def job_info(self):
        with self.lock:
            if self.job is None:
                return None
            state = self._state(self.job)
            return {"call": self.job.parent.parent.name, "speaker": self.job.name, "task": state.get("task"),
                    "automatic": bool(state.get("automatic")), "message": state.get("message", "")}

    def baseline_files(self, call, sid, state):
        build = state.get("build") or state.get("available")
        if not build:
            return []
        path = self._root(call, sid) / build / "inputs.json"
        if not path.is_file():
            return []
        inputs = read(path)
        return [*inputs.get("training_clips", []), *inputs.get("heldout_clips", [])]

    def _publish(self, root, **fields):
        path = root / "state.json"
        state = read(path) if path.is_file() else {}
        state.update(fields)
        write(path, state)

    def _release_warm(self):
        if self.timer:
            self.timer.cancel()
        if self.warm is not None:
            _stop(self.warm)
            self.warm.stdin.close()
            self.warm.stdout.close()
        self.timer = self.warm = self.warm_directory = None

    def _launch(self, command, **options):
        return subprocess.Popen(command, start_new_session=True, env=dict(self.env, VOICE_LAB_JOB_PARENT=str(os.getpid())),
                                text=True, **options)

    def build(self, call, sid, inputs, mode, *, automatic=False):
        if mode != "resume" and mode not in RECIPES:
            raise ValueError("Choose a preview or full voice build.")
        root = self._root(call, sid)
        with self.lock:
            if self.job is not None:
                raise ValueError("A text voice job is already running. Pause it to build another voice.")
            state = self._state(root)
            resuming = mode == "resume"
            pending = state.get("pending_update") if resuming else None
            if resuming:
                if pending:
                    mode, directory = pending["mode"], root / pending["build"]
                elif state.get("task") == "build" and state.get("mode") in RECIPES and state.get("build"):
                    mode, directory = state["mode"], root / state["build"]
                else:
                    raise ValueError("Choose a build size to start a text voice.")
                inputs = read(directory / "inputs.json")
            else:
                evidence = {key: [os.path.relpath(p, call) if Path(p).is_absolute() else p for p in paths]
                            for key, paths in inputs.items()}
                # An early six-clip build can have just two held-out clips. Freeze its
                # genuine controls when the permanent holdout partition later grows.
                if state.get("available"):
                    previous = root / state["available"]
                    prior_inputs = previous / "inputs.json"
                    if prior_inputs.is_file():
                        held = [os.path.relpath(locate(previous, p).resolve(), call.resolve()) for p in read(prior_inputs)["heldout_clips"]]
                        if held and set(held).issubset(evidence["heldout_clips"]):
                            evidence["heldout_clips"] = held
                signature = hashlib.sha256(json.dumps([2, evidence], sort_keys=True).encode()).hexdigest()[:16]
                directory = root / (mode + "-" + signature)
                inputs = {key: [os.path.relpath(call / p, directory) for p in paths]
                          for key, paths in evidence.items()}
            self._release_warm()
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            write(directory / "inputs.json", inputs)
            failures = (pending or {}).get("failures", 0) if automatic else 0
            pending = {"build": directory.name, "mode": mode, "failures": failures}
            training_count = state.get("training_count", len(inputs["training_clips"])) if resuming else len(inputs["training_clips"])
            self._publish(root, state="building", task="build", automatic=automatic, pending_update=pending,
                          training_count=training_count,
                          build=directory.name, mode=mode, message="Preparing the voice builder…")
            self.job = root
            self.thread = threading.Thread(target=self._build, args=(root, directory, mode), daemon=True)
            self.thread.start()

    def _build(self, root, directory, mode):
        count, steps = RECIPES[mode]
        cache = root / ("corpus-" + directory.name.split("-", 1)[1])
        try:
            previous = self._state(root).get("available")
            for kind, stage, extra, message in _remaining_stages(directory, cache, count, steps):
                with self.lock:
                    if self.job != root or self._state(root)["state"] != "building":
                        return
                    self._publish(root, message=message)
                    if stage == "evaluate" and previous and previous != directory.name:
                        extra = [*extra, "--incumbent", str(root / previous)]
                    command = _command(kind, stage, directory, self.model_dir) + extra
                    proc = self._launch(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    self.process = proc
                last = ""
                with (directory / "build.log").open("a", encoding="utf-8") as log, proc, proc.stdout as output:
                    for line in output:
                        log.write(line)
                        if line.startswith("voice-job "):
                            with self.lock:
                                if self.job == root and self._state(root)["state"] == "building":
                                    self._publish(root, message=json.loads(line[10:])["message"])
                        elif line.strip():
                            last = line.strip()
                if proc.returncode:
                    raise RuntimeError(last[-350:] or "The voice worker stopped. Resume to try again.")
            report = read(directory / "evaluation.json")
            with self.lock:
                if self.job == root and self._state(root)["state"] == "building":
                    passed = report.get("passes_quality", False)
                    message = ("Listen to the original, then the trained voice on the same line." if passed else
                               "The held-out check was weak. Listen to the comparison and add clearer speech to improve it.")
                    comparison = {}
                    if previous and previous != directory.name:
                        prev_dir = root / previous
                        enriched = directory / "incumbent-evaluation.json"
                        prior = read(enriched) if enriched.is_file() else {}
                        if prior.get("incumbent_build") != previous:
                            prior_path = prev_dir / "evaluation.json"
                            prior = read(prior_path) if prior_path.is_file() else {"lines": []}
                        if (prev_dir / "student" / "profile.pt").is_file():
                            promote, reason = _prefer_update(prior, prev_dir, report, directory)
                        else:
                            promote, reason = True, "The published profile was missing. The rebuilt voice is available."
                        other_dir, other = (prev_dir, prior) if promote else (directory, report)
                        comparison = dict(last_decision="promoted" if promote else "kept", decision_reason=reason,
                                          alternate=other_dir.name, alternate_label="Previous build" if promote else "New build",
                                          alternate_similarity=other.get("mean_cosine"), alternate_wer=other.get("mean_word_error_rate"))
                        if not promote:
                            self._publish(root, state="ready", task=None, automatic=False, pending_update=None,
                                          message=reason + " Your previous voice is still selected.", **comparison)
                            return
                        message = reason
                    self._publish(root, state="ready", task=None, automatic=False, pending_update=None,
                                  available=directory.name, mode=mode, rendered=None,
                                  message=message, passes_quality=passed, genuine=report.get("genuine_cosine"),
                                  similarity=report["mean_cosine"], wer=report["mean_word_error_rate"],
                                  lines=len(report["lines"]), review_text=report["lines"][0]["text"], **comparison)
        except Exception as exc:
            with self.lock:
                if self.job == root and self._state(root)["state"] == "building":
                    state = self._state(root)
                    retry = None
                    message = str(exc)
                    permanent = isinstance(exc, (ValueError, FileNotFoundError)) or message.startswith(("ValueError:", "FileNotFoundError:"))
                    if state.get("automatic") and not permanent:
                        pending = state.get("pending_update") or {}
                        failures = pending.get("failures", 0)
                        if failures < len(RETRY_DELAYS):
                            delay = RETRY_DELAYS[failures]
                            retry = dict(pending, failures=failures + 1, retry_at=time.time() + delay)
                            message += f" The studio will retry in {delay} seconds."
                    self._publish(root, state="error", automatic=False, pending_update=retry, message=message)
        finally:
            with self.lock:
                if self.job == root:
                    self.process = self.job = None

    def suspend(self, *, keep_update=False, automatic_only=False):
        with self.lock:
            thread = self.thread if self.job else None
            if automatic_only and (not self.job or not self._state(self.job).get("automatic")):
                return
            if self.job:
                state = self._state(self.job)
                task = state.get("task")
                message = "Paused. Finished teaching examples are saved." if task == "build" else "Playback paused. Generate the line again."
                pending = state.get("pending_update") if keep_update or task == "render" else None
                self._publish(self.job, state="paused", pending_update=pending, automatic=False, message=message)
                _stop(self.process)
            self._release_warm()
        if thread is not None:
            thread.join()

    def render(self, call, sid, text):
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 400:
            raise ValueError("Enter between 1 and 400 characters to hear.")
        text = text.strip()
        root = self._root(call, sid)
        with self.lock:
            if self.job:
                raise ValueError("Let the active voice job finish first.")
            state = self._state(root)
            if not state.get("available"):
                raise ValueError("Build this speaker’s text voice first.")
            directory = root / state["available"]
            if self.timer:
                self.timer.cancel()
            if self.warm_directory != directory or not self.warm or self.warm.poll() is not None:
                self._release_warm()
                self.warm = self._launch(_command("student", "render", directory, self.model_dir),
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.warm_directory = directory
            self.job, self.process = root, self.warm
            self._publish(root, state="rendering", task="render", automatic=False, message="Generating your line…", rendered=None)
            self.thread = threading.Thread(target=self._render, args=(root, text, self.warm), daemon=True)
            self.thread.start()

    def _render(self, root, text, proc):
        try:
            proc.stdin.write(json.dumps({"text": text}) + "\n")
            proc.stdin.flush()
            last = ""
            for line in proc.stdout:
                if line.startswith("voice-render "):
                    reply = json.loads(line[13:])
                    if "error" in reply:
                        raise ValueError(reply["error"])
                    with self.lock:
                        if self._state(root)["state"] == "rendering":
                            self._publish(root, state="ready", task=None, rendered=text, seconds=reply["seconds"],
                                          message="Your generated line is ready to play.")
                    break
                if line.strip():
                    last = line.strip()
            else:
                raise RuntimeError(last[-350:] or "The speech renderer stopped. Generate the line again.")
        except Exception as exc:
            with self.lock:
                if self._state(root)["state"] == "rendering":
                    self._publish(root, state="error", message=str(exc))
        finally:
            with self.lock:
                self.job = self.process = None
                if self.warm is proc:
                    self.timer = threading.Timer(60, self._idle, args=(proc,))
                    self.timer.daemon = True
                    self.timer.start()

    def _idle(self, proc):
        with self.lock:
            if self.warm is proc and self.job is None:
                self._release_warm()

    def audio(self, call, sid, role):
        root = self._root(call, sid)
        with self.lock:
            state = self._state(root)
            if role == "alternate-review" and state.get("alternate"):
                return root / state["alternate"] / "review.wav"
            if role == "profile" and state.get("available"):
                return root / state["available"] / "student" / "profile.pt"
            if state.get("available") and (role == "review" or role == "tts" and state.get("rendered")):
                return root / state["available"] / ("review.wav" if role == "review" else "utterance.wav")
        raise ValueError("Build this voice and generate a line to hear it.")
