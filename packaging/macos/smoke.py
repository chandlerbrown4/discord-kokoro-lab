"""Exercise a downloaded app and native launch. Uses only an unused test library."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler
import zipfile


def idle_window_mib(pid):
    """Read the native display process's physical footprint, excluding its audio service."""
    # struct rusage_info_v0 in the macOS SDK's sys/resource.h
    class Usage(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("cpu_wakeups_pageins", ctypes.c_uint64 * 5),
                    ("wired", ctypes.c_uint64), ("resident", ctypes.c_uint64), ("footprint", ctypes.c_uint64),
                    ("start_exit_times", ctypes.c_uint64 * 2)]
    usage = Usage()
    proc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    samples = []
    for _ in range(5):
        time.sleep(.4)
        if proc.proc_pid_rusage(pid, 0, ctypes.byref(usage)) != 0:
            raise OSError(ctypes.get_errno(), "Could not sample the test app's memory")
        samples.append(usage.footprint / 2**20)
    return statistics.median(samples)


def check(archive, scratch):
    marker = scratch / ".voice-lab-smoke"
    scratch.mkdir(parents=True, exist_ok=True)
    if not marker.exists() and any(scratch.iterdir()):
        raise ValueError("Choose a new empty smoke directory; an existing model cache created by this test is also supported.")
    marker.touch()
    support = scratch / "Application Support"
    with tempfile.TemporaryDirectory(dir=scratch) as dest:
        with zipfile.ZipFile(archive) as z:
            if "Read Me.txt" not in z.namelist():
                raise ValueError("First-run directions are missing from the ZIP.")
            for item in z.namelist():
                if item == "Read Me.txt":
                    continue
                parts = Path(item).parts
                if not parts or parts[0] != "Voice Lab Beta.app" or '..' in parts or any(
                    part in {".venv", ".git", "__pycache__", "data"} for part in parts
                ):
                    raise ValueError(f"Unexpected entry in release: {item}")
            if z.testzip() is not None:
                raise ValueError("Bad ZIP CRC.")
        subprocess.run(["/usr/bin/ditto", "-x", "-k", str(archive), dest], check=True)
        bundle = Path(dest) / "Voice Lab Beta.app"
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(bundle)], check=True)
        contents = bundle / "Contents"
        for entry in (contents / "Resources" / "SHA256SUMS").read_text().splitlines():
            expected, name = entry.split("  ", 1)
            assert hashlib.sha256((contents / name).read_bytes()).hexdigest() == expected, name
        # Trap unexpected browser opens without launching the tester's own browser.
        browser_probe = Path(dest) / "browser-probe"
        browser_marker = Path(dest) / "browser-opened"
        browser_probe.write_text('#!/bin/sh\ntouch "$VOICE_LAB_BROWSER_MARKER"\n')
        browser_probe.chmod(0o700)
        env = dict(os.environ, VOICE_LAB_APP_SUPPORT=str(support), BROWSER=str(browser_probe),
                   VOICE_LAB_BROWSER_MARKER=str(browser_marker))
        for key in ("VOICE_LAB_DATA_DIR", "VOICE_LAB_MODEL_DIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            env.pop(key, None)
        urlfile = support / "studio-url"
        urlfile.unlink(missing_ok=True)
        app = subprocess.Popen([str(contents / "MacOS" / "VoiceLab")], env=env, start_new_session=True)
        client = build_opener(ProxyHandler({}))
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if app.poll() is not None:
                    raise RuntimeError("The app closed before starting.")
                if urlfile.exists():
                    break
                time.sleep(.5)
            if not urlfile.exists():
                raise TimeoutError(f"Studio startup timed out. Inspect {support / 'setup.log'}")
            url = urlfile.read_text().strip()
            parsed = urlsplit(url)
            assert parsed.hostname == "127.0.0.1" and parsed.scheme == "http" and parsed.port
            with client.open(url, timeout=10) as response:
                assert b"Call studio" in response.read()
            while time.monotonic() < deadline:
                with client.open(url + "api/state", timeout=5) as response:
                    state = json.load(response)
                assert state["capture"]["state"] == "idle" and state["route"]["state"] == "idle"
                if state["setup"]["state"] == "error":
                    raise RuntimeError(state["setup"]["message"])
                if state["setup"]["state"] == "ready":
                    break
                time.sleep(.5)
            assert state["setup"]["state"] == "ready", state["setup"]
            print("Packaged app, local API, model setup, capture off and routing off:", state["setup"], flush=True)
            print(f"Idle native window process: {idle_window_mib(app.pid):.1f} MiB physical footprint.", flush=True)
            request = Request(url + "api/quit", data=json.dumps({"call": state["call"]}).encode(),
                              headers={"Origin": url.rstrip("/"), "Content-Type": "application/json"})
            with client.open(request, timeout=10) as response:
                assert json.load(response)["ok"]
            assert app.wait(timeout=30) == 0
            assert not urlfile.exists()
            assert not browser_marker.exists(), "Native launch requested a browser."
            print("Quit closed the native app and localhost service.")
        finally:
            if app.poll() is None:
                os.killpg(app.pid, signal.SIGTERM)
                try:
                    app.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(app.pid, signal.SIGKILL)
                    app.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--work-dir", type=Path, help="Empty test directory, or a prior smoke directory to reuse downloads")
    args = parser.parse_args()
    if args.work_dir:
        check(args.zip.resolve(), args.work_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="voice-lab-app-smoke-") as root:
            check(args.zip.resolve(), Path(root))
