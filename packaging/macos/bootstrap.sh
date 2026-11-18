#!/bin/bash
set -eu
umask 077
resources="$(cd "$(dirname "$0")" && pwd -P)"
support="${VOICE_LAB_APP_SUPPORT:-$HOME/Library/Application Support/Voice Lab Beta}"
mkdir -p "$support"
status_file="$support/launch-status"
stage() {
    printf '%s\n' "$1" > "$status_file.tmp.$$"
    mv -f "$status_file.tmp.$$" "$status_file"
    printf '%s\n' "$1"
}
child=""
downloads=""
stop() {
    if [ -n "$child" ]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    exit 130
}
cleanup() {
    if [ -n "$downloads" ]; then rm -rf "$downloads"; fi
}
trap stop INT TERM HUP
trap cleanup EXIT
run() { "$@" & child=$!; wait "$child"; child=""; }

export UV_NO_CONFIG=1 UV_NO_PROGRESS=1 UV_PYTHON_DOWNLOADS=automatic
export UV_PYTHON_INSTALL_DIR="$support/python"
export UV_CACHE_DIR="$support/download-cache"
export UV_DEFAULT_INDEX=https://pypi.org/simple
unset UV_INDEX UV_EXTRA_INDEX_URL UV_INDEX_URL UV_FIND_LINKS PIP_INDEX_URL PIP_EXTRA_INDEX_URL PYTHONHOME
export HF_HOME="$support/model-cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
export VOICE_LAB_DATA_DIR="${VOICE_LAB_DATA_DIR:-$support/library}"
export VOICE_LAB_MODEL_DIR="${VOICE_LAB_MODEL_DIR:-$support/model-cache/diarization}"
export VOICE_LAB_CAPTURE_TOOL="$resources/../Helpers/capture-discord"
export VOICE_LAB_BETA_URL_FILE="$support/studio-url"
export NUMBA_CACHE_DIR="$support/numba-cache"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
mkdir -p "$support/tools" "$VOICE_LAB_DATA_DIR" "$NUMBA_CACHE_DIR"

# Every change to the runtime pins gets a separate environment. The app's own
# Python modules stay in Resources and can be updated without reinstalling ML wheels.
version=$(cat "$resources/runtime-id")
env_dir="$support/runtimes/$version"
export PYTHONPATH="$resources/core"
export PYTHONPYCACHEPREFIX="$support/bytecode/$version"
uv="$support/tools/uv-0.11.16"
if [ ! -x "$uv" ]; then
    stage "Downloading the setup tool…"
    downloads=$(mktemp -d "$support/tools/.download.XXXXXX")
    url="https://github.com/astral-sh/uv/releases/download/0.11.16/uv-aarch64-apple-darwin.tar.gz"
    run /usr/bin/curl --fail --silent --show-error --location --retry 3 --connect-timeout 20 \
        --proto '=https' --proto-redir '=https' "$url" -o "$downloads/uv.tar.gz"
    printf '%s  %s\n' '2b25be1af546be330b340b0a76b99f989daa6d92678fdffb87438e661e9d88fb' "$downloads/uv.tar.gz" | /usr/bin/shasum -a 256 -c -
    /usr/bin/tar -xzf "$downloads/uv.tar.gz" -C "$downloads" uv-aarch64-apple-darwin/uv
    chmod 755 "$downloads/uv-aarch64-apple-darwin/uv"
    mv "$downloads/uv-aarch64-apple-darwin/uv" "$uv"
    cleanup
    downloads=""
fi

if [ ! -f "$env_dir/.ready" ] || [ ! -x "$env_dir/bin/python" ] || [ ! -x "$env_dir/bin/ffmpeg" ]; then
    stage "Installing private Python 3.12…"
    run "$uv" python install --no-bin 3.12.12
    if [ ! -x "$env_dir/bin/python" ]; then
        mkdir -p "$(dirname "$env_dir")"
        run "$uv" venv --managed-python --python 3.12.12 "$env_dir"
    fi
    stage "Installing speech and audio tools. The first download can take several minutes…"
    run "$uv" pip sync --python "$env_dir/bin/python" --index-url https://pypi.org/simple \
        --only-binary :all: --require-hashes "$resources/runtime.lock"
    stage "Checking the audio runtime…"
    run "$env_dir/bin/python" "$resources/verify-runtime.py" "$env_dir"
    touch "$env_dir/.ready"
fi
export VOICE_LAB_TEXT_RUNTIME="$resources/text-runtime.sh"
export VOICE_LAB_PYTHON="$env_dir/bin/python"
export PATH="$resources/bin:$env_dir/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export SSL_CERT_FILE="$("$VOICE_LAB_PYTHON" -c 'import certifi; print(certifi.where())')"
cd "$support"
if [ "${1:-}" = "--verify-runtime" ]; then
    "$VOICE_LAB_PYTHON" "$resources/verify-runtime.py" "$env_dir" --verify-only
    "$VOICE_LAB_PYTHON" -m voice_lab beta --help
    exit 0
fi
stage "Opening the local studio…"
exec "$VOICE_LAB_PYTHON" -m voice_lab beta --port 0
