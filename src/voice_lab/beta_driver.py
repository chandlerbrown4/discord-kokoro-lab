"""User-initiated download of the vendor's signed BlackHole installer."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.request import urlopen

FILENAME = "BlackHole2ch-0.7.1.pkg"
URL = "https://existential.audio/downloads/" + FILENAME
SHA256 = "57b540f27a3e29c37e310e01bee0fdfab76733087e47f997ef9dccf851400dcf"


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    installer = directory / FILENAME
    if not installer.exists() or hashlib.sha256(installer.read_bytes()).hexdigest() != SHA256:
        with urlopen(URL, timeout=25) as response:
            data = response.read(2 * 1024 * 1024 + 1)
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise RuntimeError("The audio driver download did not match the published package.")
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as tmp:
            tmp.write(data)
        os.replace(tmp.name, installer)
    result = subprocess.run(["/usr/sbin/pkgutil", "--check-signature", str(installer)],
                            capture_output=True, text=True, timeout=30)
    if result.returncode or "Q5C99V536K" not in result.stdout:
        raise RuntimeError("macOS could not verify the Existential Audio installer.")
    return installer


def open_installer(directory):
    if sys.platform != "darwin":
        raise ValueError("The BlackHole installer requires macOS.")
    installer = prepare(directory)
    subprocess.run(["/usr/bin/open", "-a", "Installer", str(installer)], check=True, timeout=10)
