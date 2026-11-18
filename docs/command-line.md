# Command-line and model workflows

For the downloadable Mac app, see the [Voice Lab quick start](../README.md). Run the commands below from the repository root. Keep recordings, transcripts, generated audio, and fitted models in the Git-ignored `data/` directory.

## Setup

Desktop capture requires macOS 13 or newer and the Xcode Command Line Tools (`xcode-select --install`). Use Python 3.11 or 3.12 and FFmpeg (`brew install ffmpeg`). Kokoro's English pronunciation fallback may also require `brew install espeak-ng`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[diarize,asr]'
```

The first run of diarization, Whisper, Inno, or Kokoro may download public model files. Audio, transcripts, voice packs, and synthesized speech are processed and saved on your machine. Diarization caches its models in `data/models/diarization` by default. Set `HF_HOME` to choose the Hugging Face cache. The optional backends below can be installed separately.

## 1. Capture Discord playback

Open the Discord desktop app and join the call yourself. From the repository root, build the recorder and start it. Press Ctrl+C to stop and finalize the WAV. Use `--seconds 10` for a short test.

```sh
./desktop/build.sh
./desktop/.build/capture-discord data/recordings/call.wav
```

When macOS prompts for access, grant **Screen & System Audio Recording** to the requesting app in **System Settings → Privacy & Security**. Restart that app if macOS asks. Use **Discord → Settings → Voice & Video → Mic Test → Let's Check** to test capture with just your own voice. [Apple permission guide](https://support.apple.com/guide/mac-help/mchld6aa7d23/mac) · [Discord Mic Test](https://support.discord.com/hc/en-us/articles/360020641332-Mic-Testing)

The WAV contains the app's playback mix, including Discord sounds and any mic monitoring that Discord plays back. Your own mic normally goes to the call as input and is absent from the playback WAV. [Apple's application audio filter](https://developer.apple.com/videos/play/wwdc2022/10156/?time=341)

An existing audio or video recording from OBS also works as the input to `diarize`. In OBS, use the **macOS Audio Capture** source and select Discord. [OBS macOS audio guide](https://obsproject.com/kb/macos-desktop-audio-capture-guide)

## 2. Group speech by speaker and prepare clips

```sh
voice-lab diarize data/recordings/call.wav data/sessions/call
voice-lab prepare data/sessions/call data/datasets/call
```

Diarization estimates the speaker count by default and assigns each voice a number for this recording. If you know the count, supply it with `--speakers`. It discards intervals flagged as overlapping speech. The model can miss overlap or assign speech to the wrong group, so listen to the clips before selecting a reference. The session manifest and prepared clips include timestamps from the original recording. [Sherpa-ONNX speaker diarization](https://k2-fsa.github.io/sherpa/onnx/c-api/html/speaker_diarization.html)

If you have separate microphone takes, use `voice-lab import-tracks data/sessions/take --consent --track 101=/path/to/speaker-a.wav` and then run `prepare` on that session. Repeat `--track` for additional takes and use the same numeric ID for the same speaker.

Preparation trims quiet, splits at pauses, rolls off rumble, and levels peaks. It writes 24 kHz mono PCM16 WAVs and draft transcripts, grouped by speaker. Add `--denoise` for steady noise or `--no-transcribe` to skip Whisper. The default transcription model is `small.en`. Both `diarize` and `prepare` accept `--local-files-only` once their models are cached.

Listen to the clips in `data/datasets/call/1/clips/` and select a few clean utterances from the same speaker. Pass the filenames in the order you want them joined.

```sh
voice-lab reference data/datasets/call/1 data/references/speaker-a \
  --clip clips/clip-000001.wav --clip clips/clip-000002.wav \
  --clip clips/clip-000003.wav
```

Choose filenames and the numeric group from your own dataset. F5 accepts a 3–12 second bundle. For Kokoro fitting, aim for 20–30 seconds. Listen to `reference.wav` and edit `transcript.txt` to match it exactly. If any chosen clip lacks a transcript, the bundle starts with an empty transcript file. Skip clips with clipping, echo, music, or another person's voice.

## 3. Generate a baseline with F5

On an Apple Silicon Mac running macOS 14 or later, fetch the pinned model and vocoder once. Subsequent renders read those weights from the local cache. This example uses a short, reviewed reference to generate new words. [MLX platform builds](https://pypi.org/project/mlx/0.32.4/#files)

```sh
python -m pip install -e '.[f5]'
voice-lab f5-fetch
voice-lab f5-speak --reference data/references/speaker-a/reference.wav \
  --transcript data/references/speaker-a/transcript.txt --seed 391 \
  --text 'Can you give me a second to catch up?' --output data/takes/baseline.wav
```

To try fitting, choose at least four separate, clean clips from the same speaker. Check their words in the prepared speaker's `manifest.jsonl` and correct any transcription errors. Keep both the conditioning reference and your listening-test recordings separate from the training clips. The default fit trains a rank-8 adapter on the final six attention blocks for 24 steps. The adapter bundle records the model revision, training seed, settings and hashes of the input audio and text.

```sh
voice-lab f5-fit data/datasets/call/1 data/models/f5-speaker-a \
  --reference data/references/speaker-a \
  --clip clips/clip-000004.wav --clip clips/clip-000005.wav \
  --clip clips/clip-000006.wav --clip clips/clip-000007.wav
voice-lab f5-speak --reference data/references/speaker-a/reference.wav \
  --transcript data/references/speaker-a/transcript.txt --seed 391 \
  --adapter data/models/f5-speaker-a --text 'Can you give me a second to catch up?' \
  --output data/takes/fitted.wav
```

F5's upstream code is MIT-licensed and its original pretrained weights are distributed under noncommercial terms. The MLX model card identifies its checkpoint as a reshaping of those original weights. This repository fetches the model separately. [F5 license](https://github.com/SWivid/F5-TTS#license) · [MLX model card](https://huggingface.co/lucasnewman/f5-tts-mlx)

## 4. Compare likeness and naturalness

Use the same text in both synthesized takes. For a same-line comparison, have the speaker record that line separately. For an entirely new line, use a separate recording of the same speaker and add `--different-text`.

```sh
voice-lab audition --reference data/held-out/actual.wav \
  --baseline data/takes/baseline.wav --candidate data/takes/fitted.wav \
  --output data/listening/round-01
```

Play `listen.wav`. The reference plays first, followed by A and B in randomized order at matched active speech levels. Fill in `scorecard.txt` before opening `key.json`. Rate likeness and naturalness separately, and flag missing or changed words. Include a deadpan line and an animated one. Keep the baseline and test each new setting on fresh lines to see whether a change generalizes.

## Route your microphone into Discord

### Preview a custom voice locally

Use Python 3.11 or 3.12 and FFmpeg. Install the conversion dependencies and the pinned OpenVoice implementation, then download the two public models once. Model inference reads the cache locally. [OpenVoice implementation](https://github.com/myshell-ai/OpenVoice/tree/74a1d147b17a8c3092dd5430504bd83ef6c7eb23)

```sh
python -m pip install -e '.[vc,live]'
python -m pip install --no-deps 'git+https://github.com/myshell-ai/OpenVoice.git@74a1d147b17a8c3092dd5430504bd83ef6c7eb23'
voice-lab vc-fetch
voice-lab vc-studio --target data/references/speaker-a/reference.wav \
  --output-dir data/mic-previews --open
```

Use the local page to select a microphone, check the meter, and record a phrase. You can also choose a short existing speech file. The comparison plays your take followed by the same take converted with the streaming processor. Add `--device-rate 48000` to the studio or replay command to hear the input and output resampling used by the 48 kHz live monitor. Choose 44100 if you use that rate for monitoring. The folder you select holds the dry recording, converted recording, and comparison. Stop the local server with Ctrl+C.

For a terminal session, replace the example device ID with the physical microphone listed on your computer. These commands save the comparison without opening a playback device.

```sh
voice-lab live devices
voice-lab mic-record data/takes/my-mic.wav --mic 3 --seconds 8
voice-lab vc-replay data/takes/my-mic.wav data/takes/converted.wav \
  --target data/references/speaker-a/reference.wav
voice-lab mic-compare data/takes/my-mic.wav data/takes/converted.wav data/takes/listen.wav
```

For continuous local monitoring, use headphones and supply a clean recording of your mic voice as calibration. Choose the headphone **output** ID from `live devices`. On Apple Silicon, add `--device mps` to use Metal on `vc-studio`, `vc-replay`, and `vc-monitor`. The preflight checks device formats and times inference before you open a stream. The browser page uses the browser's microphone permission. Continuous monitoring also needs microphone access for the app running the command. On macOS, this is under **System Settings → Privacy & Security → Microphone**.

```sh
voice-lab vc-monitor --mic 3 --output-device 1 --target data/references/speaker-a/reference.wav \
  --source-calibration data/takes/my-mic.wav --check
```

Remove `--check` to start monitoring. Automatic mode measures the 160, 240 and 800 ms presets in order and starts the first one that meets its timing budget. The same worker keeps 93 ms of lookahead and sends audio to the driver every 10 ms. The app shows the selected block size and estimated software delay. If timing deteriorates for two consecutive seconds, automatic mode briefly pauses output to measure the next larger preset. You can also choose `--profile fast`, `balanced` or `quality` explicitly. Ctrl+C reports missed audio and the estimated delay including the device timestamps. Discord and the network add their own delay.

On one Apple Silicon Mac, the previous default took 643 ms from simulated input to aligned output. Automatic mode took 453 ms for both stock test voices (Kokoro Heart and Michael). Each version ran for 50 seconds per target at 48 kHz with zero dropped input blocks and zero late output packets. These timings include the simulated 10 ms device interval and exclude physical audio hardware and Discord.

To tune a quiet microphone, start with the same recorded phrase each time. The calibration defaults aim for a peak of −12 dBFS and cap gain at 18 dB. One fixed gain preserves the changes in loudness within the phrase. Set `--mic-peak-dbfs` and `--max-mic-gain-db` on replay, the browser studio, or the live monitor to repeat the settings you liked. Check a louder phrase for distortion before using a higher gain live. For a target with varied delivery, the reference WAV can include several clean excerpts from that speaker, up to 30 seconds in total.

For Discord, explicitly choose the virtual cable as the converter's output and Discord's input, with headphones selected for Discord playback.

### Use a stock effect

Install the microphone extra and choose a virtual audio cable for your system. On macOS, BlackHole 2ch is one option. Set Discord's **Input Device** to the cable and **Output Device** to your ordinary headphones. Keep the Mac's sound output on your headphones too. The route writes only to the selected cable. [BlackHole](https://github.com/ExistentialAudio/BlackHole) · [Discord voice settings](https://support.discord.com/hc/en-us/articles/33030151293079-Discord-Voice-Video-Streaming-Guide)

```sh
python -m pip install -e '.[live]'
voice-lab live devices
# Replace 3 and 7 with your physical mic input and virtual cable output IDs.
voice-lab live route --mic 3 --virtual-output 7 --effect radio --check
voice-lab live route --mic 3 --virtual-output 7 --effect radio
```

The `clean` effect passes your mic through, `radio` band-limits it, and `giant` lowers pitch and vowel resonances together while keeping your timing. For `giant`, install the Rubber Band 4 shared library (`brew install rubberband` on macOS). Preview it on a mono PCM16 file without opening a device with `voice-lab live preview input.wav giant.wav --effect giant`. Choose the same effect in `live route` to use it on your microphone. Custom F5 and Kokoro profiles render offline.

Use Discord's **Mic Test** before joining a call. Stop the route with Ctrl+C and reselect your ordinary microphone in Discord when finished. [Discord Mic Test](https://support.discord.com/hc/en-us/articles/360020641332-Mic-Testing)

## Make a Kokoro voice pack

```sh
python -m pip install -e '.[kokoro]'
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

The [official Kokoro release](https://huggingface.co/hexgrad/Kokoro-82M) provides inference weights without the original training pipeline or style encoder. This project makes a separate voice pack for the existing model. It doesn't retrain the 82M base weights.

For a fast baseline, Inno derives a voice pack from the reference audio. Resemblance varies by speaker. [Inno model card](https://huggingface.co/remsky/kokoro-inno-clone-tuner)

```sh
voice-lab enroll data/references/speaker-a/reference.wav data/models/speaker-a.pt

voice-lab speak --voice data/models/speaker-a.pt \
  --text 'This is a synthetic voice test. The train leaves at seven.' \
  --output data/artifacts/speaker-a-preview.wav
```

Use `voice-lab speak --speed 0.9` with the same voice, text, and output options to render a slower read. Compare several separate real turns to check whether that rate is a better match for the speaker.

For experimental fitting, KVoiceWalk searches for a closer voice tensor using the reference audio and its exact transcript. The wrapper runs a temporary copy of the pinned revision with small fixes for tempo parsing and undefined scores. Fit uses US English and can take a long time on CPU. The fitted pack works with the same `voice-lab speak` command. [Pinned KVoiceWalk source](https://github.com/RobViren/kvoicewalk/tree/3a38c6030cc4657df073c67ded37cdf7627c4969)

```sh
git clone https://github.com/RobViren/kvoicewalk.git third-party/kvoicewalk
git -C third-party/kvoicewalk checkout --detach 3a38c6030cc4657df073c67ded37cdf7627c4969
python3.12 -m venv third-party/kvoicewalk/.venv
third-party/kvoicewalk/.venv/bin/python -m pip install \
  'kokoro==0.9.4' 'faster-whisper==1.2.1' 'librosa>=0.11.0' \
  'numpy>=2.2.6,<2.4' 'onnxruntime<1.24' 'resemblyzer>=0.1.4' \
  'soundfile>=0.13.1' 'torch>=2.7.0' 'tqdm>=4.67.1' 'setuptools<81' \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

voice-lab fit \
  --trainer third-party/kvoicewalk --device cpu --steps 1000 \
  --reference data/references/speaker-a/reference.wav \
  --transcript data/references/speaker-a/transcript.txt \
  --output data/models/speaker-a-fitted.pt
```

Use `--device cuda` with a CUDA-enabled PyTorch install. If fitting doesn't improve on the starting voice, the command exits without a pack. Try a cleaner take or more steps, and compare the enrolled and fitted packs using sentences that weren't spoken in the reference.

## Offline smoke test

```sh
voice-lab fixture data/recordings/synthetic-test
voice-lab prepare data/recordings/synthetic-test data/datasets/synthetic-test --no-transcribe
python -m unittest discover -s tests -v
```

The fixture uses two separated synthetic tones and exercises preparation without models. If you administer a server and need the older per-user bot workflow, see [recorder setup](../recorder/README.md).

## Run the call studio from source

Install the preparation dependencies in [Setup](#setup) and the OpenVoice dependencies in [Preview a custom voice locally](#preview-a-custom-voice-locally). Build the capture helper with `./desktop/build.sh`, then run `voice-lab beta --open`. To build the downloadable app, use the [Mac packaging guide](../packaging/macos/README.md).

## How Voice Lab builds text voices

Each speaker has their own training clips, held-out clips, teaching examples and fitted profile. Qwen3-TTS uses clean speech and its transcript to generate teaching lines. With enough distinct training recordings, a full build also tests an updated reference and a fine-tuned Qwen candidate on a separate development set. Whisper checks the words and WeSpeaker checks the voice. Kokoro learns from the teaching examples that pass. A smaller per-speaker profile then works with the shared Kokoro model for typed speech. [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)

The final comparison uses the same genuine recordings for the current and candidate Kokoro voices. It checks likeness, word errors, timing, typical pitch and pitch variation when pitch can be measured. Close or conflicting results keep the current voice. A voice that failed its quality check can be replaced when the candidate restores likeness and typical pitch on every review line, preserves words and timing, and stays close to the recorded pitch range. You can listen to either version against the original in the studio. Closing the window allows background work to continue. Quitting suspends work until next launch. When you open an older library, existing text voices get the automatic improvement setting.

The reported speaker similarity is a cosine score on separate original recordings. The app also shows similarity between genuine recordings and the word error rate on the generated versions. These are beta checks. Listen for natural timing and timbre, especially before using an unfamiliar voice. Live mic conversion uses OpenVoice directly and follows the timing of your spoken performance.
