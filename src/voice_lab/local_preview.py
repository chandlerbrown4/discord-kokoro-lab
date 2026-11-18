"""Serve the local microphone preview and convert one bounded take at a time."""

import argparse
import secrets
import shutil
import subprocess
import sys
import threading
import wave
import webbrowser
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import mic_preview, vc_stream
from .vc import calibration_options
from .models import _new_file
from .tracks import new_directory


MAX_BYTES = 16 * 1024 * 1024
ACCEPTED = {"audio/wav", "audio/wave", "audio/x-wav", "audio/webm", "audio/ogg", "audio/mp4",
            "audio/m4a", "audio/x-m4a", "audio/mpeg", "audio/flac", "audio/x-flac", "audio/aac",
            "audio/aiff", "audio/x-aiff"}


def decode_take(data, dry, *, max_seconds=30):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install FFmpeg to decode audio.")
    # A bounded, seekable stdin cache lets FFmpeg read M4A files with a trailing index.
    # Restrict input protocols and demuxers so an upload cannot reference external assets.
    try:
        result = subprocess.run([
            ffmpeg, "-nostdin", "-v", "error", "-xerror", "-max_alloc", "67108864",
            "-protocol_whitelist", "cache,pipe", "-format_whitelist", "wav,mov,matroska,webm,ogg,flac,mp3,aac,aiff",
            "-read_ahead_limit", "-1", "-i", "cache:pipe:0", "-map", "0:a:0", "-t", str(max_seconds + .05),
            "-ac", "1", "-ar", str(mic_preview.MIC_RATE),
            "-f", "s16le", "pipe:1",
        ], input=data, capture_output=True, timeout=15)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("The audio decoder timed out. Try a shorter WAV recording.") from exc
    if result.returncode:
        raise ValueError("Could not decode this audio file. Try a WAV, M4A, WebM, MP3, FLAC, Ogg or AIFF file.")
    frames = len(result.stdout) // 2
    if not mic_preview.MIC_RATE <= frames <= max_seconds * mic_preview.MIC_RATE:
        raise ValueError(f"Select a spoken take between 1 and {max_seconds} seconds.")
    samples = array("h", result.stdout)
    if sys.byteorder == "big":
        samples.byteswap()
    mic_preview.require_signal(samples)
    with _new_file(dry) as temp, wave.open(temp, "wb") as wav:
        wav.setparams((1, 2, mic_preview.MIC_RATE, 0, "NONE", "not compressed"))
        wav.writeframes(result.stdout)


def make_take(data, settings):
    destination = settings.output_dir / ("take-" + secrets.token_hex(8))
    with new_directory(destination) as stage:
        dry, wet, reel = (stage / name for name in ("dry.wav", "converted.wav", "side-by-side.wav"))
        decode_take(data, dry)
        vc_stream.replay(argparse.Namespace(input=dry, output=wet, target=settings.target,
                                            source_calibration=None, weights=settings.weights,
                                            threads=settings.threads, device=settings.device,
                                            profile=getattr(settings, "profile", None),
                                            device_rate=getattr(settings, "device_rate", None),
                                            **calibration_options(settings)))
        mic_preview.compare_takes(dry, wet, reel)
    return destination / "side-by-side.wav"


class PreviewHandler(BaseHTTPRequestHandler):
    timeout = 20

    def reply(self, code, data, mime, *, take=None):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; "
                         "style-src 'unsafe-inline'; connect-src 'self'; media-src blob:; frame-ancestors 'none'")
        self.send_header("Permissions-Policy", "camera=(), microphone=(self)")
        if take:
            self.send_header("X-Preview-Take", take)
        self.end_headers()
        self.wfile.write(data)

    def reject(self, code, reason):
        self.reply(code, reason.encode("utf-8"), "text/plain; charset=utf-8")

    def check_host(self):
        if self.headers.get_all("Host") != [self.server.host_port]:
            self.reject(403, "Open the exact local address shown in your terminal.")
            return False
        return True

    def do_GET(self):
        from .preview_page import PAGE

        if not self.check_host():
            return
        if self.path != "/":
            self.reject(404, "Page not found.")
            return
        self.reply(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        if not self.check_host():
            return
        if self.path != "/take":
            self.reject(404, "Page not found.")
            return
        if self.headers.get_all("Origin") != ["http://" + self.server.host_port]:
            self.reject(403, "Record or submit a file from this local preview page.")
            return
        if (self.headers.get_content_type() not in ACCEPTED or self.headers.get("Content-Encoding")
                or self.headers.get("Transfer-Encoding")):
            self.reject(415, "Send a supported audio file with its audio content type.")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal() or len(lengths[0]) > 9:
            self.reject(411, "Supply a valid audio file length.")
            return
        size = int(lengths[0])
        if not 0 < size <= MAX_BYTES:
            self.reject(413, "Select a nonempty audio file up to 16 MiB.")
            return
        if not self.server.converting.acquire(blocking=False):
            self.reject(409, "A take is already converting. Try again after it finishes.")
            return
        try:
            data = self.rfile.read(size)
            if len(data) != size:
                self.reject(400, "Audio upload was interrupted.")
                return
            reel = make_take(data, self.server.settings)
            self.reply(200, reel.read_bytes(), "audio/wav", take=reel.parent.name)
        except ValueError as exc:
            self.reject(422, str(exc))
        except (OSError, RuntimeError, ImportError, wave.Error) as exc:
            self.reject(500, str(exc))
        finally:
            self.server.converting.release()


class PreviewServer(ThreadingHTTPServer):
    def __init__(self, settings):
        super().__init__(("127.0.0.1", settings.port), PreviewHandler)
        self.settings = settings
        self.converting = threading.Lock()
        self.host_port = f"127.0.0.1:{self.server_port}"


def run(args):
    args.target = args.target.expanduser().resolve(strict=True)
    if not args.target.is_file():
        raise ValueError("Choose a target voice reference file.")
    if not 0 <= args.port <= 65535 or args.threads < 1:
        raise ValueError("Choose a valid port and a positive thread count.")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with PreviewServer(args) as server:
        url = f"http://{server.host_port}/"
        print(f"Local microphone preview: {url}", flush=True)
        print(f"Successful takes are saved under {args.output_dir}. Press Ctrl+C to stop.", flush=True)
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def configure(commands):
    parser = commands.add_parser("vc-studio", help="Open a local browser microphone comparison")
    parser.add_argument("--target", type=Path, required=True, help="Approved target voice reference")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for local preview takes")
    parser.add_argument("--weights", type=Path, help="Local pinned converter weights directory")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="OpenVoice compute device")
    vc_stream.add_stream_options(parser)
    vc_stream.add_device_replay_option(parser)
    parser.add_argument("--port", type=int, default=8765, help="Loopback port; 0 selects a free port")
    parser.add_argument("--open", action="store_true", help="Open the bound URL in your default browser")
    parser.set_defaults(handler=run)
