"""Loopback API for the call studio. Only server-owned audio paths are exposed."""

import json
from importlib.resources import files
import os
from pathlib import Path
import signal
import threading
from urllib.parse import urlsplit
import webbrowser

from .beta import _atomic_text
from .local_preview import ACCEPTED, MAX_BYTES, PreviewHandler, PreviewServer


class Handler(PreviewHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        super().end_headers()

    def send_json(self, code, payload):
        self.reply(code, json.dumps(payload, allow_nan=False).encode(), "application/json")

    def do_GET(self):
        if not self.check_host():
            return
        from .beta_audio import audio_devices
        studio = self.server.studio
        path = urlsplit(self.path).path
        try:
            if path == "/":
                self.reply(200, files("voice_lab").joinpath("beta_studio.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self.send_json(200, studio.snapshot())
            elif path == "/api/devices":
                self.send_json(200, {"devices": audio_devices()})
            elif path.startswith("/audio/"):
                parts = path.split("/")
                if len(parts) != 5 or parts[2] != studio.call_id:
                    raise ValueError("Choose a clip from the current call.")
                with studio.lock:
                    if parts[2] != studio.call_id:
                        raise ValueError("The selected call changed.")
                    audio = studio.audio(parts[3], parts[4])
                    if not audio:
                        raise ValueError("A sample is still being prepared.")
                    data = Path(audio).read_bytes()
                mime = "application/octet-stream" if parts[3] == "profile" else "audio/wav"
                self.reply(200, data, mime)
            else:
                self.reject(404, "Page not found.")
        except (ValueError, OSError, RuntimeError) as exc:
            self.send_json(400, {"error": str(exc)})

    def _body(self, limit):
        lengths = self.headers.get_all("Content-Length", [])
        if (len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal()
                or len(lengths[0]) > 9 or self.headers.get("Transfer-Encoding")
                or self.headers.get("Content-Encoding")):
            raise ValueError("Send a request with a valid content length.")
        n = int(lengths[0])
        if not 0 < n <= limit:
            raise ValueError("This request exceeds the upload size limit.")
        data = self.rfile.read(n)
        if len(data) != n:
            raise ValueError("The upload was interrupted.")
        return data

    def do_POST(self):
        if not self.check_host():
            return
        if self.headers.get_all("Origin") != ["http://" + self.server.host_port]:
            self.reject(403, "Use the local Voice Lab window for controls.")
            return
        studio = self.server.studio
        try:
            mime = self.headers.get_content_type()
            if self.path in ("/api/mic/file", "/api/voice/file"):
                if mime not in ACCEPTED:
                    raise ValueError("Choose a supported audio file.")
                audio = self._body(MAX_BYTES)
                with studio.actions:
                    if self.path == "/api/voice/file":
                        if self.headers.get("X-Voice-Call") != (studio.call_id or "new"):
                            raise ValueError("The call changed. Choose the recording again.")
                        studio.import_recording(audio)
                    else:
                        studio.calibrate(audio)
            else:
                if mime != "application/json":
                    raise ValueError("Send a JSON control request.")
                body = json.loads(self._body(4096))
                if not isinstance(body, dict):
                    raise ValueError("Invalid control request.")
                with studio.actions:
                    if studio.closing.is_set():
                        raise ValueError("The studio is shutting down.")
                    if body.get("call") != studio.call_id:
                        raise ValueError("The call changed. Refresh the studio and try again.")
                    sid = body.get("speaker")
                    if self.path == "/api/attach":
                        studio.attach()
                    elif self.path == "/api/detach":
                        studio.capture.stop()
                        studio.wake.set()
                    elif self.path == "/api/new":
                        studio.select(None)
                    elif self.path == "/api/select" and isinstance(body.get("id"), str):
                        studio.select(body["id"])
                    elif self.path == "/api/mic/record":
                        studio.record_mic(body.get("mic"))
                    elif self.path == "/api/text/build":
                        studio.build_text(sid, body.get("mode", "preview"))
                    elif self.path == "/api/text/updates":
                        studio.set_text_updates(sid, body.get("enabled"))
                    elif self.path == "/api/text/pause":
                        studio.text.suspend()
                    elif self.path == "/api/text/speak":
                        studio.speak(sid, body.get("text"))
                    elif self.path == "/api/preview":
                        studio.preview(sid)
                    elif self.path in ("/api/check", "/api/route"):
                        studio.route_voice(sid, body.get("mic"), body.get("output"), self.path == "/api/check")
                    elif self.path == "/api/route/microphone":
                        studio.route_microphone(body.get("mic"), body.get("output"))
                    elif self.path == "/api/route/stop":
                        studio.route.stop()
                    elif self.path == "/api/retry":
                        studio.retry()
                    elif self.path == "/api/cable/installer":
                        studio.install_driver()
                    elif self.path == "/api/quit":
                        studio.closing.set()
                        threading.Thread(target=self.server.shutdown, daemon=True).start()
                    else:
                        self.reject(404, "Unknown studio control.")
                        return
            self.send_json(200, {"ok": True})
        except (ValueError, OSError, RuntimeError) as exc:
            self.send_json(400, {"error": str(exc)})

    def log_message(self, fmt, *args):
        # Polling the local status every second should not fill the app log.
        if self.path != "/api/state":
            super().log_message(fmt, *args)


class Server(PreviewServer):
    def __init__(self, studio, args):
        super().__init__(args)
        self.RequestHandlerClass = Handler
        self.studio = studio


def serve(studio, args):
    if not 0 <= args.port <= 65535:
        raise ValueError("Choose a valid localhost port.")
    with Server(studio, args) as server:
        url = "http://" + server.host_port + "/"
        url_file = os.environ.get("VOICE_LAB_BETA_URL_FILE")
        if url_file:
            _atomic_text(Path(url_file), url + "\n")
        old = {}
        def stop(sig, frame):
            threading.Thread(target=server.shutdown, daemon=True).start()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                old[sig] = signal.signal(sig, stop)
            print("Voice Lab call studio at " + url, flush=True)
            if args.open:
                webbrowser.open(url)
            server.serve_forever(poll_interval=.2)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
            if url_file and Path(url_file).is_file() and Path(url_file).read_text().strip() == url:
                Path(url_file).unlink(missing_ok=True)
