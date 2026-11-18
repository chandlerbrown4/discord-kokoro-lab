# Voice Lab Beta for Mac

Unzip the release, move **Voice Lab Beta.app** to Applications, and double-click it. The beta targets Apple Silicon Macs running macOS 14 or later. The first launch downloads a private Python runtime, audio tools, and speech models. Allow about 3 GB of free space before recording. The launcher shows setup progress and opens the native AppKit call studio. Use **Window → View Setup Log** for startup details.

Join a call in the Discord desktop app and start capture from the studio. Allow Screen & System Audio Recording when macOS prompts. Your microphone preview uses the microphone you choose. Live conversion into Discord uses a virtual audio output. If your Mac needs one, choose **Open BlackHole installer** in the studio and complete the vendor’s installer. Select the virtual output in Voice Lab and the same device as Discord’s input. Keep your usual headphones as Discord’s output. [BlackHole](https://github.com/ExistentialAudio/BlackHole)

The local beta uses an ad hoc signature. If macOS blocks a download you trust, follow Apple’s instructions for opening software from an unidentified developer. Managed Macs may require administrator approval. [Apple’s app-opening guide](https://support.apple.com/guide/mac-help/mh40616/mac)

The app stores recordings, fitted voices, downloaded tools, and the setup log under `~/Library/Application Support/Voice Lab Beta/`. You can close the studio window during a call and reopen it from the Dock. **Quit Voice Lab** stops the local service and live audio.

## Build and check

On an Apple Silicon Mac with Xcode Command Line Tools and Python 3.11 or newer, run these commands from the repository root.

```sh
python3 packaging/macos/build.py
python3 packaging/macos/smoke.py data/artifacts/beta-distribution/Voice-Lab-Beta-macOS-arm64.zip
```

The build compiles the launcher and capture helper, verifies pinned OpenVoice source, signs the app, and writes a ZIP. `Contents/Resources/SHA256SUMS` lists the bundled conversion modules, helper, and wheel lock. The app fetches the pinned arm64 uv release and its managed CPython 3.12.12 runtime at first launch. It installs the checksum-locked wheels in its private runtime. The upstream OpenVoice license and revision are included in the app. [OpenVoice source](https://github.com/myshell-ai/OpenVoice/tree/74a1d147b17a8c3092dd5430504bd83ef6c7eb23) · [uv release](https://github.com/astral-sh/uv/releases/tag/0.11.16)

The smoke test starts the native app against an empty temporary library, checks the extracted bundle and model initialization, verifies capture and routing are idle, and measures the display process's physical footprint before quitting. The memory reading is the median of five macOS kernel samples and excludes the audio service. Use `--work-dir /path/to/empty-test-directory` to retain and reuse its isolated caches.

On the release test Mac, the 0.2.1 window used about 33 MiB with two saved speakers visible. The 0.2.0 launcher and a dedicated Chrome session showing the same call used 282–285 MiB. With Chrome already open, the old launcher and the added studio tab accounted for about 107–116 MiB. These are idle measurements on one Apple Silicon Mac. Active speaker fitting and live voice conversion have separate memory costs.

Regenerate the wheel lock on Apple Silicon with the beta's uv release after intentionally changing `runtime.in`.

```sh
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile --only-binary :all: \
  --python-version 3.12.12 --python-platform aarch64-apple-darwin \
  --no-annotate --no-header --generate-hashes \
  --output-file packaging/macos/runtime.lock packaging/macos/runtime.in
```

## Optional text voices

The first text build installs two private environments from `teacher.lock` and `student.lock`. The studio process can remain open between jobs without loading either model. Both environments share the app's public model cache. To regenerate either lock on Apple Silicon, specify `teacher.in` or `student.in` and the matching output name in the command above and omit `--only-binary :all:`.

After the first build, the studio watches each saved text voice for new speech. Six distinct new training clips queue a full update. The app keeps its evaluation recordings separate and reuses them across versions. It automatically compares the candidate's likeness, intelligibility and delivery against the current profile. The selected voice stays available for export and playback. The user can turn updates off for an individual speaker. Older saved voices get the same setting.

Automatic updates run in a separate worker while the app is open, including during capture. Foreground microphone work and typed playback take priority. The worker saves accepted teaching lines, alignment progress and Kokoro optimizer checkpoints. The app can return to an interrupted upgrade after a foreground request or a restart. It also retries transient failures with a delay. After repeated failures it waits for new speech.

The source checkout can use the Mac packaging runtime via uv, or use `VOICE_LAB_TEACHER_PYTHON` and `VOICE_LAB_STUDENT_PYTHON` to point at preinstalled, compatible interpreters. To generate speech from an exported profile:

```sh
PYTHONPATH=src bash packaging/macos/text-runtime.sh student \
  -m voice_lab.distill_student speak /path/to/profile.pt /path/to/new.wav \
  --text "Here is a new line."
```

Developers can provide custom teaching lines with `python -m voice_lab.distill_jobs corpus --help`. The local job directory uses an `inputs.json` with `training_clips` and `heldout_clips` arrays of separate WAV paths. The production worker accepts a UTF-8 prompt file and an accepted-line count. Training, evaluation and rendering each run in their designated environment. The two-speaker acceptance harness is `packaging/macos/smoke-text.py --help`.

### Release measurements

The signed 0.4.0 archive opened an isolated library on an Apple Silicon Mac, initialized its cached public models, and used 33.8 MiB for the idle native window. This measurement excludes the service and model workers. The update acceptance harness can reuse a previous two-voice library with `--previous-run` and feed additional stock turns with `--additional`. It checks the selected profile's bytes, both original-to-generated comparisons, foreground playback, opt-out, restart and replay of duplicate audio.

On an Apple Silicon test Mac, the 0.3.0 signed app used 33.6 MiB of physical memory for its idle native window. The separate local service loads the audio engines as needed. The two-speaker acceptance test imported English stock recordings, built a 12-line text voice for each, generated new lines, and reopened the library. Six new utterances had zero transcribed word errors. Their WeSpeaker cosine against the intended stock voice ranged from 0.761 to 0.845, and against the other voice from 0.455 to 0.494. The second warm request for each profile took 0.36 to 0.72 seconds for 2.60 to 2.75 seconds of audio. Stock-voice results describe these fixtures. Similarity on a person's recorded speech and naturalness by ear can differ.

In the separate live check, two 50-second runs per version exercised the real Metal converter on a simulated 48 kHz audio clock. Input-to-output alignment decreased from 643 ms in 0.2.1 to 453 ms in 0.3.0 on each target. No input blocks or late output packets were recorded in those runs. Physical hardware and Discord transport add their own latency.
