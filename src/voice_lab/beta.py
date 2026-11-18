"""Local call studio. Capture produces finalized chunks; one worker enrolls their speakers."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time

from . import mic_preview
from .local_preview import decode_take

CALL_ID = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")
APP_HOME = Path.home() / "Library" / "Application Support" / "Voice Lab"


def _atomic_text(path, text):
    tmp = path.with_name("." + path.name + "." + secrets.token_hex(4))
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


class Studio:
    def __init__(self, args):
        from .beta_audio import CaptureProcess, RouteProcess
        from .beta_enrollment import Enrollment
        from .beta_text import TextVoices

        self.root = args.data_dir.expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.calls = self.root / "calls"
        self.calls.mkdir(mode=0o700, exist_ok=True)
        self.model_dir = args.model_dir.expanduser().resolve()
        self.local_only = args.local_files_only
        self.text = TextVoices(self.model_dir, self.local_only)
        self.requested_device = args.device
        self.device = "cpu"
        self.capture = CaptureProcess(args.capture_tool)
        self.route_factory = RouteProcess
        self.route = RouteProcess(device=self.device)
        self.enroll_factory = Enrollment
        self.call_id = None
        self.enroll = None
        self.processed = set()
        self.failed = {}
        self.fit_failed = {}
        self.voices = {"speakers": [], "processed_chunks": 0}
        self.analysis = ""
        self.working = False
        self.lock = threading.RLock()
        self.actions = threading.Lock()
        self.wake = threading.Event()
        # After startup, only ingest/import, foreground preemption and preference
        # changes need to revisit saved libraries. Idle polling never scans all calls.
        self.text_checks = dict.fromkeys(sorted(
            p.name for p in self.calls.iterdir() if CALL_ID.fullmatch(p.name) and (p / "text-voices").is_dir()))
        self.text_migrated = set()
        self.text_errors = {}
        self.closing = threading.Event()
        self.setup = {"state": "loading", "message": "Preparing speech models…"}
        self.setup_thread = None
        self.setup_process = None
        self.previewing = False
        self.preview_file = None
        self.preview_speaker = None
        self.preview_error = None
        self.preview_score = None
        self.preview_job = None
        self.installing_driver = False
        self.driver_message = ""
        self.calibration = self.root / "my-microphone.wav"
        self.calibration_job = None
        self.calibration_error = None
        remembered = self.root / "selected-call.txt"
        if remembered.is_file():
            last = remembered.read_text().strip()
            if CALL_ID.fullmatch(last) and (self.calls / last).is_dir():
                self._load(last)
        self.worker = threading.Thread(target=self._analyze, daemon=True, name="call-enrollment")
        self.worker.start()
        self.prepare_models()

    def prepare_models(self):
        with self.lock:
            if self.setup_thread and self.setup_thread.is_alive():
                return
            self.setup = {"state": "loading", "message": "Checking speech models…"}
            self.setup_thread = threading.Thread(target=self._prepare, daemon=True, name="model-setup")
            self.setup_thread.start()

    def _prepare(self):
        # Checking Torch/Metal and the model cache can use hundreds of MB.
        # Let that memory go as soon as the idle studio is ready.
        args = [sys.executable, "-m", "voice_lab.beta_setup", str(self.model_dir), self.requested_device]
        if self.local_only:
            args.append("--local-files-only")
        try:
            with self.lock:
                if self.closing.is_set():
                    return
                proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
                self.setup_process = proc
            device = None
            error = "Speech setup stopped before it finished. Retry to check the runtime."
            with proc, proc.stdout as output:
                for line in output:
                    if not line.startswith("studio-setup "):
                        continue
                    event = json.loads(line[len("studio-setup "):])
                    if "message" in event:
                        with self.lock:
                            self.setup["message"] = event["message"]
                    elif "error" in event:
                        error = event["error"]
                    elif "device" in event:
                        device = event["device"]
            if proc.returncode or device not in ("mps", "cpu"):
                raise RuntimeError(error)
            with self.lock:
                self.device = device
                self.route = self.route_factory(device=device)
                self.setup = {"state": "ready", "message": "Models ready"}
        except Exception as exc:
            with self.lock:
                self.setup = {"state": "error", "message": str(exc)}
        finally:
            with self.lock:
                self.setup_process = None
            self.wake.set()

    def _load(self, call_id):
        enroll = self.enroll_factory(self.calls / call_id / "voices", self.model_dir, local_only=True)
        self._migrate_text(call_id, enroll)
        voices = enroll.snapshot()
        self.text_checks[call_id] = None
        _atomic_text(self.root / "selected-call.txt", call_id)
        self.call_id, self.enroll, self.voices = call_id, enroll, voices
        self.processed, self.failed, self.fit_failed = set(), {}, {}
        self.analysis = ""
        self.preview_file = self.preview_speaker = self.preview_error = self.preview_score = None
        self.wake.set()

    def _new_call(self):
        call_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + secrets.token_hex(4)
        (self.calls / call_id).mkdir(mode=0o700)
        self._load(call_id)

    @property
    def directory(self):
        return self.calls / self.call_id if self.call_id else None

    def _chunks(self):
        return sorted(self.directory.glob("run-*/chunk-*.wav")) if self.directory else []

    def select(self, session):
        with self.lock:
            if (self.capture.status()["state"] in ("starting", "running")
                    or self.route.status()["state"] in ("starting", "running") or self.previewing or self.working):
                raise ValueError("Stop active audio and let the current voice job finish before changing calls.")
            if session is None:
                if not (self.text.job_info() or {}).get("automatic"):
                    self.text.suspend(keep_update=True)
                self._new_call()
            elif isinstance(session, str) and CALL_ID.fullmatch(session) and (self.calls / session).is_dir():
                if not (self.text.job_info() or {}).get("automatic"):
                    self.text.suspend(keep_update=True)
                self._load(session)
            else:
                raise ValueError("Choose a saved call.")

    def attach(self):
        with self.lock:
            if self.capture.status()["state"] in ("starting", "running"):
                raise ValueError("Discord capture is already active.")
            if not self.call_id:
                self._new_call()
            run = self.directory / (datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ-") + secrets.token_hex(2))
            self.capture.start(run)
        self.wake.set()

    def _next_job(self):
        with self.lock:
            if self.setup["state"] != "ready" or self.enroll is None:
                return None
            # Fit as soon as enough examples pass the checks, even if capture is still running.
            # Diarization can continue collecting speakers while the faster live
            # path has the output device. Defer a new optimizer run until it stops.
            routing = self.route.status()["state"] in ("starting", "running")
            for speaker in self.voices["speakers"]:
                sid = speaker["id"]
                if not routing and speaker["needs_fit"] and sid not in self.fit_failed:
                    self.working = True
                    self.analysis = f"Fitting {speaker['name']} and checking separate clips…"
                    return self.enroll, "fit", sid
            for path in self._chunks():
                if path not in self.processed and path not in self.failed:
                    self.working = True
                    self.analysis = "Separating speakers in the latest audio…"
                    return self.enroll, "ingest", path
            self.analysis = ""
        return None

    def _analyze(self):
        while not self.closing.is_set():
            job = self._next_job()
            if job is None:
                self._schedule_text()
                self.wake.wait(0.8)
                self.wake.clear()
                continue
            enroll, kind, key = job
            try:
                if kind == "fit":
                    enroll.fit(key, device="cpu", threads=2)
                else:
                    if key.parent.name.startswith("run-import-"):
                        enroll.ingest(key, imported=True)
                    else:
                        enroll.ingest(key)
                with self.lock:
                    self.voices = enroll.snapshot()
                    self.text_checks[self.call_id] = None
                    if kind == "ingest":
                        self.processed.add(key)
            except Exception as exc:
                with self.lock:
                    (self.failed if kind == "ingest" else self.fit_failed)[key] = str(exc)
                    self.voices = enroll.snapshot()
            finally:
                with self.lock:
                    self.working = False
            # A continuous capture may already have another chunk available.
            self._schedule_text()

    def _migrate_text(self, call_id, enroll):
        if call_id in self.text_migrated:
            return
        call = self.calls / call_id
        for path in (call / "text-voices").glob("*/state.json"):
            sid = path.parent.name
            if not sid.isascii() or not sid.isdecimal():
                continue
            try:
                state = self.text.snapshot(call, [{"id": sid}]).get(sid, {})
                if state.get("available") or state.get("build"):
                    enroll.ensure_text_watch(sid, self.text.baseline_files(call, sid, state))
                    if isinstance(state.get("training_count"), int):
                        enroll.mark_text_attempt(sid, state["training_count"])
            except (OSError, ValueError, sqlite3.Error) as exc:
                self.text_errors[call_id] = f"Speaker {sid}: {exc}"
        self.text_migrated.add(call_id)

    def _schedule_text(self):
        from .beta_enrollment import TEXT_UPDATE_CLIPS

        with self.lock:
            if (self.closing.is_set() or self.setup["state"] != "ready" or not self.text_checks
                    or self.previewing or self.calibration_job or self.text.job_info() or self.text.warm
                    or self.route.status()["state"] in ("starting", "running")):
                return
            now = time.time()
            call_id = next((cid for cid, after in self.text_checks.items() if not after or after <= now), None)
            if call_id is None:
                return
            self.text_checks.pop(call_id)
            call = self.calls / call_id
            errors = []
            try:
                enroll = self.enroll if self.call_id == call_id else self.enroll_factory(
                    call / "voices", self.model_dir, local_only=True)
                self._migrate_text(call_id, enroll)
                for sid, count, attempted in enroll.text_watches():
                    try:
                        state = self.text.snapshot(call, [{"id": sid}]).get(sid, {})
                        pending = state.get("pending_update")
                        newer = count - attempted >= TEXT_UPDATE_CLIPS
                        # Fresh evidence can replace an unsuccessful candidate during its backoff.
                        if pending and pending.get("failures") and newer:
                            pending = None
                        if pending and pending.get("retry_at", 0) > now:
                            when = pending["retry_at"]
                            self.text_checks[call_id] = min(when, self.text_checks.get(call_id) or when)
                            continue
                        if not pending and (not state.get("available") or not newer):
                            continue
                        mode = "resume" if pending else "full"
                        inputs = {} if pending else enroll.text_inputs(sid)
                        self.text.build(call, sid, inputs, mode, automatic=True)
                        enroll.mark_text_attempt(sid, state.get("training_count", attempted) if pending else count)
                        if self.call_id == call_id:
                            self.voices = enroll.snapshot()
                        # Check the next speaker once this job finishes.
                        self.text_checks[call_id] = None
                        break
                    except (ValueError, RuntimeError, OSError) as exc:
                        errors.append(f"Speaker {sid}: {exc}")
            except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
                errors.append(str(exc))
            if errors:
                self.text_errors[call_id] = " ".join(errors)
            else:
                self.text_errors.pop(call_id, None)

    def _give_audio_priority(self, *, only_automatic):
        job = self.text.job_info()
        if job and job.get("automatic"):
            self.text_checks[job["call"]] = None
        self.text.suspend(keep_update=True, automatic_only=only_automatic)

    def set_text_updates(self, sid, enabled):
        with self.lock:
            self._voice(sid)
            if type(enabled) is not bool:
                raise ValueError("Choose whether to update this speaker automatically.")
            job = self.text.job_info()
            if (not enabled and job and job["automatic"] and job["call"] == self.call_id and job["speaker"] == sid):
                self.text.suspend()
            self.enroll.set_text_updates(sid, enabled)
            self.voices = self.enroll.snapshot()
            self.text_checks[self.call_id] = None
        self.wake.set()

    def snapshot(self):
        with self.lock:
            calls = sorted((p.name for p in self.calls.iterdir() if CALL_ID.fullmatch(p.name) and p.is_dir()), reverse=True)
            return {"call": self.call_id, "calls": calls, "setup": dict(self.setup), "device": self.device,
                    "capture": self.capture.status(), "route": self.route.status(), "voices": self.voices,
                    "analysis": self.analysis, "backlog": sum(p not in self.processed and p not in self.failed for p in self._chunks()),
                    "analysis_errors": list(self.failed.values())[-1:] + list(self.fit_failed.values())[-1:],
                    "calibrated": self.calibration.is_file(), "calibrating": self.calibration_job is not None,
                    "calibration_error": self.calibration_error, "previewing": self.previewing,
                    "preview": self.preview_speaker if self.preview_file else None, "preview_error": self.preview_error,
                    "preview_score": self.preview_score, "installing_driver": self.installing_driver,
                    "driver_message": self.driver_message,
                    "text_voices": self.text.snapshot(self.directory, self.voices["speakers"]) if self.directory else {},
                    "text_job": self.text.job_info(), "text_update_error": self.text_errors.get(self.call_id)}

    def install_driver(self):
        from .beta_driver import open_installer
        with self.lock:
            if self.installing_driver:
                return
            self.installing_driver = True
            self.driver_message = "Downloading and verifying BlackHole 2ch…"
        def launch():
            try:
                open_installer(self.root / "driver-cache")
                message = "Complete the macOS installer, restart your Mac if prompted, then refresh devices."
            except Exception as exc:
                message = str(exc)
            with self.lock:
                self.driver_message = message
                self.installing_driver = False
        threading.Thread(target=launch, daemon=True, name="driver-installer").start()

    def retry(self):
        with self.lock:
            self.failed.clear()
            self.fit_failed.clear()
            if self.setup["state"] == "error":
                self.prepare_models()
        self.wake.set()

    def _voice(self, sid):
        if not isinstance(sid, str):
            raise ValueError("Select a speaker.")
        speaker = next((s for s in self.voices["speakers"] if s["id"] == sid), None)
        if not speaker:
            raise ValueError("This speaker is unavailable in the selected call.")
        return speaker

    def _target(self, sid, *, preview=False):
        voice = self._voice(sid)
        if not (voice["model_status"] in ("optimized", "reference_only")
                or (preview and voice["model_status"] == "needs_more_audio")):
            raise ValueError("This speaker is still collecting speech or fitting a voice.")
        if not self.calibration.is_file():
            raise ValueError("Record a mic sample first.")
        if self.calibration_job is not None:
            raise ValueError("Let the microphone sample finish first.")
        return self.enroll.selected_files(sid)

    def import_recording(self, data):
        import tempfile
        with tempfile.TemporaryDirectory(dir=self.root) as scratch:
            decoded = Path(scratch) / "source.wav"
            decode_take(data, decoded, max_seconds=120)
            with self.lock:
                if not self.call_id:
                    self._new_call()
                run = self.directory / ("run-import-" + secrets.token_hex(8))
                run.mkdir(mode=0o700)
                decoded.replace(run / "chunk-000001.wav")
        self.wake.set()

    def build_text(self, sid, mode):
        with self.lock:
            self._voice(sid)
            if self.route.status()["state"] in ("starting", "running"):
                raise ValueError("Stop live output before building a text voice.")
            inputs = {} if mode == "resume" else self.enroll.text_inputs(sid)
            self._give_audio_priority(only_automatic=True)
            self.text.build(self.directory, sid, inputs, mode)
            self.enroll.ensure_text_watch(sid)
            if mode != "resume":
                self.enroll.mark_text_attempt(sid, len(inputs["training_clips"]))
            self.voices = self.enroll.snapshot()
            self.text_checks[self.call_id] = None

    def speak(self, sid, text):
        with self.lock:
            self._voice(sid)
            if self.route.status()["state"] in ("starting", "running"):
                raise ValueError("Stop live output before generating a text preview.")
            if not isinstance(text, str) or not 1 <= len(text.strip()) <= 400:
                raise ValueError("Enter between 1 and 400 characters to hear.")
            if not self.text.snapshot(self.directory, [{"id": sid}]).get(sid, {}).get("available"):
                raise ValueError("Build this speaker’s text voice first.")
            self._give_audio_priority(only_automatic=True)
            self.text.render(self.directory, sid, text)

    def calibrate(self, data):
        with self.lock:
            if self.calibration_job or self.previewing or self.route.status()["state"] in ("starting", "running"):
                raise ValueError("Stop the active voice before replacing your mic sample.")
            tmp = self.root / (".mic-" + secrets.token_hex(8) + ".wav")
            try:
                decode_take(data, tmp)
                tmp.replace(self.calibration)
            finally:
                tmp.unlink(missing_ok=True)
            self.preview_file = self.preview_speaker = self.preview_score = None
            self.preview_error = None
            self.calibration_error = None

    def record_mic(self, mic):
        from .beta_audio import audio_devices
        if type(mic) is not int or not any(d["id"] == mic and d["inputs"] and not d["virtual"] for d in audio_devices()):
            raise ValueError("Select your physical microphone.")
        with self.lock:
            if self.calibration_job or self.previewing or self.route.status()["state"] in ("starting", "running"):
                raise ValueError("Finish the active audio operation before recording.")
            self._give_audio_priority(only_automatic=True)
            tmp = self.root / (".mic-" + secrets.token_hex(8) + ".wav")
            proc = subprocess.Popen([sys.executable, "-m", "voice_lab", "mic-record", str(tmp),
                "--mic", str(mic), "--seconds", "6"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.calibration_job = proc
            self.calibration_error = None

        def finish():
            try:
                output, _ = proc.communicate(timeout=25)
                if proc.returncode:
                    raise RuntimeError(output.strip().splitlines()[-1] if output.strip() else "Microphone capture stopped.")
                with self.lock:
                    tmp.replace(self.calibration)
                    self.preview_file = self.preview_speaker = self.preview_error = self.preview_score = None
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                with self.lock:
                    self.calibration_error = "Microphone timed out. Check macOS Microphone permission."
            except (OSError, RuntimeError) as exc:
                with self.lock:
                    self.calibration_error = str(exc)
            finally:
                tmp.unlink(missing_ok=True)
                with self.lock:
                    self.calibration_job = None
        threading.Thread(target=finish, daemon=True, name="microphone-calibration").start()

    def preview(self, sid):
        with self.lock:
            if self.previewing:
                raise ValueError("A preview is already rendering.")
            if self.route.status()["state"] in ("starting", "running"):
                raise ValueError("Stop live output before rendering a preview.")
            target, adapter = self._target(sid, preview=True)
            self._give_audio_priority(only_automatic=True)
            take = self.directory / "previews" / secrets.token_hex(8)
            take.mkdir(mode=0o700, parents=True)
            wet, pair = take / "converted.wav", take / "comparison.wav"
            args = [sys.executable, "-m", "voice_lab", "vc-replay", str(self.calibration), str(wet),
                    "--target", str(target), "--device", self.device, "--device-rate", "48000"]
            if adapter:
                args.extend(["--target-adapter", str(adapter)])
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.preview_job = proc
            self.previewing = True
            self.preview_file = self.preview_speaker = self.preview_error = self.preview_score = None

        def render():
            try:
                output, _ = proc.communicate(timeout=240)
                if proc.returncode:
                    raise RuntimeError(output.strip().splitlines()[-1] if output.strip() else "Conversion stopped.")
                mic_preview.compare_takes(self.calibration, wet, pair)
                try:
                    score = self.enroll.score_preview(sid, wet)
                except (ValueError, OSError, RuntimeError):
                    score = None
                with self.lock:
                    self.preview_file, self.preview_speaker = pair, sid
                    self.preview_score = score
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                with self.lock:
                    self.preview_error = "Conversion timed out."
            except Exception as exc:
                with self.lock:
                    self.preview_error = str(exc)
            finally:
                with self.lock:
                    self.previewing = False
                    self.preview_job = None
        threading.Thread(target=render, daemon=True, name="voice-preview").start()

    def route_voice(self, sid, mic, output, check=False):
        if type(mic) is not int or type(output) is not int:
            raise ValueError("Choose a microphone and virtual cable.")
        with self.lock:
            if self.previewing:
                raise ValueError("Let the local preview finish first.")
            target, adapter = self._target(sid)
            self._give_audio_priority(only_automatic=False)
            self.route.start(sid, target, self.calibration, mic, output, check=check, target_adapter=adapter)

    def route_microphone(self, mic, output):
        if type(mic) is not int or type(output) is not int:
            raise ValueError("Choose a physical microphone and virtual cable.")
        with self.lock:
            if self.previewing or self.calibration_job or self.setup["state"] != "ready":
                raise ValueError("Let the active setup or audio job finish first.")
            self._give_audio_priority(only_automatic=True)
            self.route.use_microphone(mic, output)

    def audio(self, role, sid):
        with self.lock:
            self._voice(sid)
            if role in ("tts", "review", "alternate-review", "profile"):
                return self.text.audio(self.directory, sid, role)
            if role == "sample":
                return self.enroll.sample(sid)
            if role == "preview" and sid == self.preview_speaker and self.preview_file:
                return self.preview_file
        raise ValueError("This clip is unavailable.")

    def close(self):
        self.closing.set()
        self.wake.set()
        with self.lock:
            self.text.suspend(keep_update=True)
            setup = self.setup_process
            for proc in (setup, self.calibration_job, self.preview_job):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
        if setup is not None:
            try:
                setup.wait(timeout=2)
            except subprocess.TimeoutExpired:
                setup.kill()
                setup.wait()
        try:
            self.capture.stop()
        finally:
            self.route.stop()
        self.worker.join(timeout=3)


def run(args):
    import fcntl
    from .beta_server import serve
    root = args.data_dir.expanduser().resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (root / ".studio.lock").open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Voice Lab is already using this library. Open its existing window.") from exc
        studio = Studio(args)
        try:
            serve(studio, args)
        finally:
            studio.close()


def configure(commands):
    parser = commands.add_parser("beta", help="Open the local Discord call and voice studio")
    parser.add_argument("--open", action="store_true", help="Open the listening studio in your browser")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("VOICE_LAB_DATA_DIR", APP_HOME / "Library")))
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("VOICE_LAB_MODEL_DIR", APP_HOME / "Models" / "diarization")))
    parser.add_argument("--capture-tool", type=Path, default=Path(os.environ.get("VOICE_LAB_CAPTURE_TOOL", "desktop/.build/capture-discord")))
    parser.add_argument("--local-files-only", action="store_true", help="Use models already cached on this Mac")
    parser.set_defaults(handler=run)
