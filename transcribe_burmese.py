#!/usr/bin/env python3
"""
Burmese Speech-to-Text CLI using BuzzASR/burmese (Whisper Large v3 fine-tune).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BitsAndBytesConfig,
)


def load_audio_16k(audio_path: str) -> np.ndarray:
    """Load audio file and convert to 16kHz mono float32 numpy array."""
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Use ffmpeg to convert any audio format (mp3, wav, m4a, flac, etc.) to 16kHz s16le PCM
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", str(audio_path),
        "-f", "s16le",
        "-ac", "1",
        "-ar", "16000",
        "-"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
        raw_pcm = proc.stdout
        audio = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    except subprocess.CalledProcessError as e:
        # Fallback to soundfile if ffmpeg fails
        audio, sr = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import scipy.signal
            num_samples = int(len(audio) * 16000 / sr)
            audio = scipy.signal.resample(audio, num_samples)
        return audio.astype(np.float32)


def transcribe(
    audio_path: str,
    device: str = "auto",
    use_8bit: bool = True,
    model_id: str = "BuzzASR/burmese",
) -> str:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading processor and model '{model_id}' on {device} (8-bit: {use_8bit if device == 'cuda' else False})...", file=sys.stderr)
    proc = WhisperProcessor.from_pretrained(model_id)

    if device == "cuda" and use_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
        )
    elif device == "cuda":
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to("cuda").eval()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        ).to("cpu").eval()

    print(f"Loading and preprocessing audio: {audio_path}...", file=sys.stderr)
    audio = load_audio_16k(audio_path)
    sample_rate = 16000
    chunk_size = 30 * sample_rate  # 30 seconds per Whisper chunk

    # Split into 30s chunks if needed
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
    transcriptions = []

    print(f"Transcribing {len(chunks)} chunk(s)...", file=sys.stderr)
    with torch.no_grad():
        for idx, chunk in enumerate(chunks, 1):
            if len(chunk) < sample_rate * 0.5:  # skip chunks under 0.5s
                continue
            feats = proc(chunk, sampling_rate=sample_rate, return_tensors="pt").input_features
            if device == "cuda":
                feats = feats.half().to("cuda")
            else:
                feats = feats.to("cpu")

            ids = model.generate(
                feats,
                num_beams=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
            )
            text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if text:
                transcriptions.append(text)

    full_text = " ".join(transcriptions).strip()
    return full_text


def main():
    parser = argparse.ArgumentParser(description="Burmese Speech-to-Text (BuzzASR)")
    parser.add_argument("audio", type=str, help="Path to input audio file (wav, mp3, m4a, flac, etc.)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Optional output text file path")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device (default: auto)")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit quantization on GPU (requires >= 4GB free VRAM)")

    args = parser.parse_args()

    result = transcribe(
        audio_path=args.audio,
        device=args.device,
        use_8bit=not args.no_8bit,
    )

    print("\n--- Transcription ---")
    print(result)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result, encoding="utf-8")
        print(f"\nSaved transcription to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
