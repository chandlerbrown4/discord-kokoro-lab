#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p .build
xcrun swiftc -O -parse-as-library -target "$(uname -m)-apple-macos13.0" \
  -module-cache-path .build/ModuleCache CaptureDiscord.swift WAVWriter.swift \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist \
  -o .build/capture-discord
