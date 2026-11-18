#!/bin/bash
# Run one training stage in its own pinned environment. Heavy wheels are optional.
set -eu
umask 077
kind="$1"
shift
case "$kind" in teacher|student) ;; *) exit 2 ;; esac
resources="$(cd "$(dirname "$0")" && pwd -P)"
support="${VOICE_LAB_APP_SUPPORT:-$HOME/Library/Application Support/Voice Lab Beta}"
uv="${VOICE_LAB_UV:-$support/tools/uv-0.11.16}"
if [ ! -x "$uv" ]; then uv="$(command -v uv)"; fi
export UV_NO_CONFIG=1 UV_NO_PROGRESS=1 UV_PYTHON_INSTALL_DIR="$support/python" UV_CACHE_DIR="$support/download-cache"
export HF_HOME="${HF_HOME:-$support/model-cache/huggingface}" HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
unset UV_INDEX UV_EXTRA_INDEX_URL UV_INDEX_URL UV_FIND_LINKS PIP_INDEX_URL PIP_EXTRA_INDEX_URL PYTHONHOME
lock="$resources/$kind.lock"
version="$(/usr/bin/shasum -a 256 "$lock" | cut -c1-16)"
env_dir="$support/runtimes/$kind-$version"
if [ ! -f "$env_dir/.ready" ] || [ ! -x "$env_dir/bin/python" ]; then
    echo "voice-job {\"message\":\"Installing optional $kind tools on this Mac…\"}"
    mkdir -p "$support/runtimes"
    "$uv" python install --no-bin 3.12.12
    if [ ! -x "$env_dir/bin/python" ]; then "$uv" venv --managed-python --python 3.12.12 "$env_dir"; fi
    "$uv" pip sync --python "$env_dir/bin/python" --index-url https://pypi.org/simple --require-hashes "$lock"
    touch "$env_dir/.ready"
fi
export PATH="$env_dir/bin:$PATH"
exec "$env_dir/bin/python" "$@"
