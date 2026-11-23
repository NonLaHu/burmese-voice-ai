#!/usr/bin/env python3
"""
Burmese Text-to-Speech CLI using YeeNyunn/nara-mini-v1 (Kokoro-based ONNX TTS).
"""

import argparse
import sys
from pathlib import Path

# Add nara-mini-v1 directory to python sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
NARA_DIR = SCRIPT_DIR / "nara-mini-v1"
if str(NARA_DIR) not in sys.path:
    sys.path.insert(0, str(NARA_DIR))

from synthesize import synthesize, wave


def main():
    parser = argparse.ArgumentParser(description="Burmese Text-to-Speech (Nara Mini v1)")
    parser.add_argument("text", nargs="?", default=None, help="Burmese or English text to speak")
    parser.add_argument("-t", "--text-arg", dest="text_flag", help="Alternative text flag")
    parser.add_argument("-v", "--voice", choices=["shoon", "thazin", "ko-banyar", "htet-myat"], default="shoon",
                        help="Voice character (shoon, thazin, ko-banyar, htet-myat; default: shoon)")
    parser.add_argument("-o", "--output", type=str, default="output.wav", help="Output WAV path (default: output.wav)")
    parser.add_argument("--threads", type=int, default=4, help="CPU inference threads (default: 4)")
    parser.add_argument("--no-normalize-numbers", action="store_true", help="Preserve numeric text without verbalizing")

    args = parser.parse_args()
    text = args.text or args.text_flag
    if not text:
        parser.error("Please provide text to synthesize, e.g. ./nara-tts 'မင်္ဂလာပါ။'")

    model_dir = NARA_DIR / "onnx" / args.voice
    if not model_dir.exists():
        parser.error(f"Voice model folder not found: {model_dir}")

    output_path = Path(args.output).resolve()
    print(f"Synthesizing using voice '{args.voice}'...", file=sys.stderr)
    samples = synthesize(
        model_folder=model_dir,
        text=text,
        threads=args.threads,
        normalize_numbers=not args.no_normalize_numbers,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as output:
        output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        output.writeframes((samples * 32767).astype("<i2").tobytes())

    duration = samples.size / 24000.0
    print(f"Saved {duration:.2f}s of audio to: {output_path}")


if __name__ == "__main__":
    main()
