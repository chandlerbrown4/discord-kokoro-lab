"""Build an ad hoc signed Apple Silicon Voice Lab app and ZIP from public runtime inputs."""
import argparse
import hashlib
from pathlib import Path
import os
import plistlib
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RELEASE = ROOT / "data" / "artifacts" / "beta-distribution"
OPENVOICE_REVISION = "74a1d147b17a8c3092dd5430504bd83ef6c7eb23"
OPENVOICE_ARCHIVE_HASH = "a0bc7e5b2968ab7f8898d237b2b36f7de46710cbc3901f97dff29781eeae8abd"
OPENVOICE_FILES = ("__init__.py", "attentions.py", "commons.py", "mel_processing.py",
                   "models.py", "modules.py", "transforms.py")
PYTHON_ASSETS = {".py", ".html", ".css", ".js", ".svg", ".png", ".ico"}


def copy_openvoice(resources, cache_dir):
    name = "openvoice-source.tar.gz"
    cached = cache_dir / name
    if not cached.is_file() or hashlib.sha256(cached.read_bytes()).hexdigest() != OPENVOICE_ARCHIVE_HASH:
        url = f"https://codeload.github.com/myshell-ai/OpenVoice/tar.gz/{OPENVOICE_REVISION}"
        with tempfile.NamedTemporaryFile(dir=cache_dir) as download:
            with urlopen(url, timeout=90) as response:
                shutil.copyfileobj(response, download)
            download.flush()
            downloaded = Path(download.name)
            if hashlib.sha256(downloaded.read_bytes()).hexdigest() != OPENVOICE_ARCHIVE_HASH:
                raise RuntimeError("The OpenVoice source archive failed its SHA-256 check.")
            shutil.copyfile(downloaded, cached)
    code = resources / "core" / "openvoice"
    code.mkdir(parents=True)
    notices = resources / "Third-Party"
    notices.mkdir()
    with tarfile.open(cached, "r:gz") as source:
        root = "OpenVoice-" + OPENVOICE_REVISION + "/"
        files = [(root + "openvoice/" + module, code / module) for module in OPENVOICE_FILES]
        files.append((root + "LICENSE", notices / "OpenVoice-LICENSE.txt"))
        for name, dest in files:
            member = source.getmember(name)
            if not member.isfile():
                raise ValueError(f"Missing source file {name}")
            with source.extractfile(member) as member_file:
                dest.write_bytes(member_file.read())
    (notices / "OpenVoice-SOURCE.txt").write_text(
        f"OpenVoice conversion modules from https://github.com/myshell-ai/OpenVoice/tree/{OPENVOICE_REVISION}\n")


def build(output):
    if os.uname().sysname != "Darwin" or os.uname().machine != "arm64":
        raise RuntimeError("Build this release on an Apple Silicon Mac with Xcode Command Line Tools.")
    cache_dir = RELEASE / "build-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as stage_name:
        stage = Path(stage_name)
        bundle = stage / "Voice Lab Beta.app"
        contents = bundle / "Contents"
        resources = contents / "Resources"
        macos = contents / "MacOS"
        helpers = contents / "Helpers"
        for item in [resources, macos, helpers]:
            item.mkdir(parents=True)
        copy_openvoice(resources, cache_dir)
        core = resources / "core"
        code_root = ROOT / "src" / "voice_lab"
        # Only application code and UI assets are in scope. Never copy the checkout, caches or data.
        for src in sorted(code_root.rglob("*")):
            if src.is_symlink() or not src.is_file() or src.suffix not in PYTHON_ASSETS or "__pycache__" in src.parts:
                continue
            dest = core / "voice_lab" / src.relative_to(code_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

        for name in ("bootstrap.sh", "runtime.lock", "verify-runtime.py", "teacher.lock", "student.lock", "text-runtime.sh"):
            shutil.copyfile(HERE / name, resources / name)
        runtime_id = hashlib.sha256((HERE / "runtime.lock").read_bytes() + b"cpython-3.12.12-install-v1").hexdigest()[:16]
        (resources / "runtime-id").write_text(runtime_id + "\n")
        (resources / "bin").mkdir()
        command = resources / "bin" / "voice-lab"
        command.write_text('#!/bin/sh\nexec "$VOICE_LAB_PYTHON" -m voice_lab "$@"\n')
        command.chmod(0o755)
        properties = dict(
            CFBundleIdentifier="local.voicelab.studio", CFBundleName="Voice Lab Beta",
            CFBundleDisplayName="Voice Lab Beta", CFBundleExecutable="VoiceLab", CFBundlePackageType="APPL",
            CFBundleVersion="5", CFBundleShortVersionString="0.4.0", LSMinimumSystemVersion="14.0",
            CFBundleIconFile="AppIcon.icns",
            NSHighResolutionCapable=True,
            NSMicrophoneUsageDescription="Convert the microphone you select into a reviewed speaker voice.",
            NSScreenCaptureUsageDescription="Capture playback from the Discord app for speaker enrollment.",
            NSAudioCaptureUsageDescription="Capture Discord call playback for local speaker enrollment.",
        )
        with (contents / "Info.plist").open("wb") as info:
            plistlib.dump(properties, info)
        env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET="14.0")
        swift = ["xcrun", "swiftc", "-O", "-target", "arm64-apple-macos14.0", "-module-cache-path", str(stage / "modules")]
        studio_sources = [HERE / "Launcher.swift", *sorted(HERE.glob("Studio*.swift"))]
        subprocess.run(swift + ["-parse-as-library", *map(str, studio_sources), "-o", str(macos / "VoiceLab")],
                       check=True, env=env)
        iconset = stage / "AppIcon.iconset"
        iconset.mkdir()
        subprocess.run(swift + [str(HERE / "AppIcon.swift"), "-o", str(stage / "render-icon")], check=True, env=env)
        subprocess.run([str(stage / "render-icon"), str(iconset)], check=True)
        subprocess.run(["/usr/bin/iconutil", "-c", "icns", str(iconset), "-o", str(resources / "AppIcon.icns")], check=True)
        capture = helpers / "capture-discord"
        desktop = ROOT / "desktop"
        subprocess.run(swift + ["-parse-as-library", str(desktop / "CaptureDiscord.swift"), str(desktop / "WAVWriter.swift"),
                               "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist", "-Xlinker",
                               str(desktop / "Info.plist"), "-o", str(capture)], check=True, env=env)
        sign = ["codesign", "--force", "--timestamp=none", "--sign", "-"]
        subprocess.run(sign + [str(capture)], check=True)
        inventory = [capture, *sorted(path for path in resources.rglob("*") if path.is_file())]
        sums = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(contents)}" for path in inventory]
        (resources / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        subprocess.run(sign + [str(bundle)], check=True)
        subprocess.run(["codesign", "--verify", "--deep", "--strict", str(bundle)], check=True)
        delivery = stage / "delivery"
        delivery.mkdir()
        bundle.rename(delivery / bundle.name)
        shutil.copyfile(HERE / "Read Me.txt", delivery / "Read Me.txt")
        archive = stage / "release.zip"
        subprocess.run(["/usr/bin/ditto", "-c", "-k", "--norsrc", str(delivery), str(archive)], check=True)
        os.replace(archive, output)
    print(f"{output}\nSHA-256 {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RELEASE / "Voice-Lab-Beta-macOS-arm64.zip")
    build(parser.parse_args().output.resolve())
