"""Manage the beta's local capture helper and explicitly selected voice route."""

import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from .live import _sounddevice


def audio_devices():
    """List selectable PortAudio devices. IDs are valid for the current device list."""
    sd = _sounddevice()
    try:
        devices = sd.query_devices()
    except sd.PortAudioError as exc:
        raise RuntimeError(f"Could not list audio devices: {exc}") from exc
    rows = []
    for index, device in enumerate(devices):
        name = str(device["name"])
        cable = any(term in name.casefold() for term in ("blackhole", "loopback", "soundflower", "vb-cable",
                                                         "vb-audio", "vb audio", "virtual cable"))
        rows.append(dict(id=index, name=name, inputs=int(device["max_input_channels"]),
                         outputs=int(device["max_output_channels"]), virtual=cable))
    return rows


class _AudioChild:
    """The control lock serializes changes. A separate lock keeps UI status reads quick."""

    def __init__(self):
        self._control = threading.RLock()
        self._lock = threading.Lock()
        self._process = None
        self._reader = None
        self._state = {"state": "idle", "message": ""}
        self._stopping = False

    def status(self):
        with self._lock:
            return self._state.copy()

    def _active(self):
        return self._process is not None and self._process.poll() is None

    def _launch(self, arguments, label, ready_message, *, check=False, **details):
        # Caller holds _control. Each start owns a new child and its stdin lifetime pipe.
        try:
            proc = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, errors="replace",
                                    bufsize=1, start_new_session=True)
        except OSError as exc:
            with self._lock:
                self._state = dict(state="error", message=f"Could not start {label}: {exc}", **details)
            raise
        with self._lock:
            self._process = proc
            self._stopping = False
            self._state = dict(state="starting", message=f"Starting {label}…", **details)

        def watch():
            last_line = ""
            diagnostic = ""
            try:
                with proc.stdout as output:
                    for line in output:
                        line = line.strip()[-1500:]
                        if not line:
                            continue
                        last_line = line
                        if line.startswith(("capture-discord: ", "voice-lab: ")):
                            diagnostic = line.split(": ", 1)[1]
                        with self._lock:
                            if self._process is proc:
                                if line.startswith("Live timing: "):
                                    self._state["timing"] = line.removeprefix("Live timing: ")
                                elif "input hops dropped;" in line:
                                    self._state["buffer_stats"] = line
                                elif line.startswith("Maximum estimated input-to-output delay:"):
                                    self._state["delay"] = line
                                if line.startswith(ready_message) and not self._stopping:
                                    self._state.update(state="running", message=f"{label.capitalize()} active.")
                code = proc.wait()
            except (OSError, ValueError) as exc:
                code, diagnostic = -1, str(exc)
            with self._lock:
                if self._process is not proc:
                    return
                if self._stopping and code == -signal.SIGKILL:
                    self._state.update(state="error", message=f"{label.capitalize()} was forced to stop. "
                                       "Previously finalized recordings are available.")
                elif self._stopping and code in (0, 130, -signal.SIGINT, -signal.SIGTERM):
                    self._state.update(state="stopped", message=f"{label.capitalize()} stopped.")
                elif code:
                    self._state.update(state="error", message=diagnostic or last_line or f"{label} exited ({code}).")
                elif check:
                    self._state.update(state="checked", message=last_line or "Device and conversion checks passed.")
                else:
                    self._state.update(state="stopped", message=last_line or f"{label.capitalize()} stopped.")

        self._reader = threading.Thread(target=watch, name=f"voice-lab-{label}", daemon=True)
        self._reader.start()
        return self.status()

    def stop(self):
        with self._control:
            proc = self._process
            if proc is None:
                return self.status()
            if proc.poll() is None:
                with self._lock:
                    self._stopping = True
                    self._state.update(state="stopping", message="Stopping…")
                proc.stdin.close()
                # EOF triggers graceful stop and finalizes the capture tail. Escalate only for a stuck helper.
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    for sig in (signal.SIGTERM, signal.SIGKILL):
                        try:
                            os.killpg(proc.pid, sig)
                        except ProcessLookupError:
                            break
                        try:
                            proc.wait(timeout=1)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                    if proc.poll() is None:
                        raise RuntimeError("The audio helper did not stop. Retry before starting another route.")
            proc.stdin.close()
            self._reader.join(timeout=1)
            return self.status()


class CaptureProcess(_AudioChild):
    """Stream Discord playback into a fresh directory of finalized chunk-NNNNNN.wav files."""

    def __init__(self, executable):
        super().__init__()
        self.executable = Path(executable).expanduser().resolve()

    def start(self, chunks_dir, chunk_seconds=20):
        directory = Path(chunks_dir).expanduser().resolve()
        if type(chunk_seconds) is not int or not 1 <= chunk_seconds <= 300:
            raise ValueError("Choose a chunk duration between 1 and 300 whole seconds.")
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise ValueError("The Discord capture helper is missing. Reinstall the beta or build desktop/build.sh.")
        with self._control:
            if self._active():
                raise ValueError("A Discord capture is already running.")
            self.stop()
            if directory.exists():
                raise ValueError("Choose a new directory for this recording.")
            return self._launch([str(self.executable), "--chunks", str(directory), "--chunk-seconds",
                                 str(chunk_seconds), "--parent-pipe"],
                                "Discord capture", "Recording Discord audio.", directory=str(directory))


class RouteProcess(_AudioChild):
    """Route the calibrated microphone to an explicit named virtual output, one voice at a time."""

    def __init__(self, device="cpu"):
        super().__init__()
        if device not in ("cpu", "mps"):
            raise ValueError("Choose cpu or mps for voice conversion.")
        self.device = device

    @staticmethod
    def _cable_name(mic, output):
        if type(mic) is not int or type(output) is not int or mic == output:
            raise ValueError("Select a physical microphone and a distinct virtual cable.")
        devices = {row["id"]: row for row in audio_devices()}
        source, dest = devices.get(mic), devices.get(output)
        if not source or not source["inputs"] or source["virtual"]:
            raise ValueError("Select an available physical microphone.")
        if not dest or not dest["outputs"] or not dest["virtual"]:
            raise ValueError("Choose a virtual cable output such as BlackHole 2ch, Loopback or Soundflower.")
        return dest["name"]

    def use_microphone(self, mic, output):
        cable = self._cable_name(mic, output)
        args = [sys.executable, "-u", "-m", "voice_lab.beta_audio", "--passthrough",
                "--mic", str(mic), "--virtual-output", str(output), "--rate", "48000", "--effect", "clean"]
        with self._control:
            self.stop()
            return self._launch(args, "microphone", "Mic route active.",
                                speaker_id=None, mode="microphone", mic=mic, output=output, output_name=cable)

    def start(self, speaker_id, target, calibration, mic, output, *, check=False, target_adapter=None):
        cable = self._cable_name(mic, output)
        target, calibration = Path(target).resolve(strict=True), Path(calibration).resolve(strict=True)
        if not target.is_file() or not calibration.is_file():
            raise ValueError("Choose a target voice and a microphone calibration recording.")
        args = [sys.executable, "-u", "-m", "voice_lab.beta_audio", "--monitor",
                "--mic", str(mic), "--output-device", str(output),
                "--target", str(target), "--source-calibration", str(calibration),
                "--device", self.device, "--rate", "48000"]
        if target_adapter is not None:
            adapter = Path(target_adapter).resolve(strict=True)
            if not adapter.is_file():
                raise ValueError("Choose a fitted voice adapter file.")
            args.extend(["--target-adapter", str(adapter)])
        if check:
            args.append("--check")
        with self._control:
            self.stop()
            return self._launch(args, "voice route", "Monitor active.", check=check,
                                speaker_id=speaker_id, mode="voice", mic=mic, output=output, output_name=cable)


def _monitor_worker():
    # Pipe closure is authoritative even if the parent crashes. Use os.read so a daemon blocked
    # here never holds a Python buffered-stdin lock during interpreter shutdown.
    def stop_on_eof():
        while os.read(0, 1):
            pass
        os.kill(os.getpid(), signal.SIGINT)

    mode = sys.argv[1:2]
    if mode not in (["--monitor"], ["--passthrough"]):
        raise SystemExit("This module is launched by the beta voice router.")
    signal.signal(signal.SIGINT, signal.default_int_handler)
    threading.Thread(target=stop_on_eof, daemon=True).start()
    from .__main__ import main
    command = ["vc-monitor"] if mode == ["--monitor"] else ["live", "route"]
    return main(command + sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(_monitor_worker())
