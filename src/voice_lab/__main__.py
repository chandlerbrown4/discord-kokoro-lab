"""Command-line entry point; model dependencies load only when needed."""

import argparse
import sys
import wave

from . import audition, beta, diarize, f5, f5_adapter, fit, live, local_preview, mic_preview
from . import models, prepare, reference, tracks, vc, vc_stream


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="voice-lab",
        description="Prepare speaker clips, synthesize speech locally, and preview microphone effects.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    tracks.configure(commands)
    reference.configure(commands)
    models.configure(commands)
    f5.configure(commands)
    f5_adapter.configure(commands)
    audition.configure(commands)
    live.configure(commands)
    mic_preview.configure(commands)
    vc.configure(commands)
    vc_stream.configure(commands)
    local_preview.configure(commands)
    beta.configure(commands)

    split = commands.add_parser("diarize", help="Group speech from a mixed recording by speaker")
    diarize.add_arguments(split)
    split.set_defaults(handler=diarize.run)

    prep = commands.add_parser("prepare", help="Clean, segment, and transcribe a session")
    prepare.add_arguments(prep)
    prep.set_defaults(handler=prepare.run)

    trainer = commands.add_parser("fit", help="Experimentally optimize a Kokoro voice pack")
    fit.add_arguments(trainer)
    trainer.set_defaults(handler=fit.run)

    args = parser.parse_args(argv)
    try:
        return args.handler(args) or 0
    except (ValueError, RuntimeError, OSError, wave.Error) as exc:
        print(f"voice-lab: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("voice-lab: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
