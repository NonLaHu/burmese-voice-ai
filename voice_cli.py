#!/usr/bin/env python3
"""
Unified Voice AI CLI: English & Burmese Speech-to-Text, Text-to-Speech & MP3 Recorder.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NARA_DIR = SCRIPT_DIR / "nara-mini-v1"
if str(NARA_DIR) not in sys.path:
    sys.path.insert(0, str(NARA_DIR))


def load_audio_16k(audio_path: str):
    """Load audio file via ffmpeg and convert to 16kHz mono float32 numpy array."""
    import numpy as np
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

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
    except subprocess.CalledProcessError:
        import soundfile as sf
        audio, sr = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import scipy.signal
            num_samples = int(len(audio) * 16000 / sr)
            audio = scipy.signal.resample(audio, num_samples)
        return audio.astype(np.float32)


def run_stt(audio_path: str, lang: str = "auto", device: str = "auto", output_file: str = None) -> str:
    """Perform Speech-to-Text for Burmese or English (or auto-detected)."""
    import torch
    from transformers import (
        AutoModelForSpeechSeq2Seq,
        AutoProcessor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        BitsAndBytesConfig,
    )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    lang_lower = lang.lower().strip()
    is_burmese = lang_lower in ["my", "burmese", "myanmar"]
    is_english = lang_lower in ["en", "english"]
    is_auto = lang_lower in ["auto", "detect"]

    print(f"[*] Processing audio file: {audio_path}", file=sys.stderr)
    audio = load_audio_16k(audio_path)
    sample_rate = 16000

    # Auto language detection using whisper-large-v3-turbo if auto is specified
    if is_auto:
        print("[*] Detecting language using Whisper Turbo...", file=sys.stderr)
        proc_turbo = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        model_turbo = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-large-v3-turbo",
            dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device).eval()

        snippet = audio[:30 * sample_rate]
        feats = proc_turbo(snippet, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
        if device == "cuda":
            feats = feats.half()

        with torch.no_grad():
            detected_ids = model_turbo.generate(feats)
        detected_text = proc_turbo.batch_decode(detected_ids, skip_special_tokens=True)[0].strip()

        # Check if the output contains Burmese Unicode characters
        has_burmese_chars = any("\u1000" <= ch <= "\u109F" or "\uAA60" <= ch <= "\uAA7F" for ch in detected_text)
        if has_burmese_chars:
            print("[*] Detected Burmese speech. Switching to BuzzASR engine...", file=sys.stderr)
            is_burmese = True
        else:
            print("[*] Detected English/multilingual speech. Using Whisper Turbo...", file=sys.stderr)
            is_english = True

        del model_turbo, proc_turbo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    transcriptions = []
    chunk_size = 30 * sample_rate
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

    if is_burmese:
        model_id = "BuzzASR/burmese"
        print(f"[*] Engine: BuzzASR (Burmese) on {device}...", file=sys.stderr)
        proc = WhisperProcessor.from_pretrained(model_id)
        if device == "cuda":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="auto",
            )
        else:
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.float32,
            ).to("cpu").eval()

        with torch.no_grad():
            for chunk in chunks:
                if len(chunk) < sample_rate * 0.4:
                    continue
                feats = proc(chunk, sampling_rate=sample_rate, return_tensors="pt").input_features
                if device == "cuda":
                    feats = feats.half().to("cuda")
                else:
                    feats = feats.to("cpu")
                ids = model.generate(feats, num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2)
                text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
                if text:
                    transcriptions.append(text)

    else:
        model_id = "openai/whisper-large-v3-turbo"
        print(f"[*] Engine: Whisper Turbo (English/Multilingual) on {device}...", file=sys.stderr)
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device).eval()

        with torch.no_grad():
            for chunk in chunks:
                if len(chunk) < sample_rate * 0.4:
                    continue
                feats = proc(chunk, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
                if device == "cuda":
                    feats = feats.half()
                ids = model.generate(feats, language="en" if is_english else None)
                text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
                if text:
                    transcriptions.append(text)

    full_text = " ".join(transcriptions).strip()

    if output_file:
        out_p = Path(output_file).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(full_text, encoding="utf-8")
        print(f"[*] Transcription saved to: {out_p}", file=sys.stderr)

    return full_text


def run_tts(text: str, voice: str = "shoon", output_path: str = "output.mp3", threads: int = 4, play_audio: bool = False):
    """Perform Text-to-Speech for Burmese or English using Nara Mini v1."""
    from synthesize import synthesize, wave

    model_dir = NARA_DIR / "onnx" / voice
    if not model_dir.exists():
        raise FileNotFoundError(f"Voice '{voice}' not found in {NARA_DIR / 'onnx'}")

    out_file = Path(output_path).resolve()
    is_mp3 = out_file.suffix.lower() == ".mp3"
    wav_temp = out_file if not is_mp3 else out_file.with_suffix(".tmp.wav")

    print(f"[*] Synthesizing speech with voice '{voice}'...", file=sys.stderr)
    samples = synthesize(model_dir, text, threads=threads, normalize_numbers=True)

    wav_temp.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_temp), "wb") as output:
        output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        output.writeframes((samples * 32767).astype("<i2").tobytes())

    duration = samples.size / 24000.0

    if is_mp3:
        cmd = [
            "ffmpeg", "-y", "-nostdin", "-loglevel", "error",
            "-i", str(wav_temp),
            "-acodec", "libmp3lame", "-b:a", "192k",
            str(out_file)
        ]
        subprocess.run(cmd, check=True)
        wav_temp.unlink(missing_ok=True)

    print(f"[*] Saved {duration:.2f}s of audio to: {out_file}", file=sys.stderr)

    if play_audio:
        play_cmd = None
        for player in ["mpv", "ffplay", "pw-play", "paplay"]:
            if subprocess.run(["which", player], capture_output=True).returncode == 0:
                play_cmd = [player, str(out_file)]
                if player == "ffplay":
                    play_cmd.extend(["-nodisp", "-autoexit"])
                break
        if play_cmd:
            subprocess.run(play_cmd)
        else:
            print("[*] No audio player found to auto-play.", file=sys.stderr)

    return str(out_file)


def run_record(output_path: str = None, duration: int = None, transcribe: bool = False, lang: str = "auto"):
    """Record audio from microphone to MP3 and optionally transcribe."""
    from record_audio import record_mp3

    if not output_path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"recording_{timestamp}.mp3"
    elif not output_path.endswith(".mp3"):
        output_path += ".mp3"

    saved_file = record_mp3(output_path, duration=duration)

    if transcribe and Path(saved_file).exists():
        print("\n[*] Transcribing recorded audio...", file=sys.stderr)
        text = run_stt(saved_file, lang=lang)
        print("\n--- Transcription ---")
        print(text)

    return saved_file


def print_status():
    """Print system status, installed models, and capabilities."""
    import torch

    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "None"
    vram = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if has_cuda else "N/A"

    print("=" * 60)
    print(" Voice AI System Status & Installed Models")
    print("=" * 60)
    print(f"  CUDA Available: {has_cuda} ({gpu_name}, {vram})")
    print(f"  PyTorch:        {torch.__version__}")
    print()
    print("  Models Available:")
    print("  1. Burmese STT (ASR):    BuzzASR/burmese (Whisper Large v3 8-bit)")
    print("  2. English STT (ASR):    openai/whisper-large-v3-turbo (FP16)")
    print("  3. Burmese & English TTS: YeeNyunn/nara-mini-v1 (Kokoro ONNX)")
    print("     - Preset Voices:      shoon, thazin, ko-banyar, htet-myat")
    print("  4. MP3 Recording Tool:   PulseAudio/PipeWire + FFmpeg lame")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        prog="voice",
        description="Unified Voice AI CLI: English & Burmese STT, TTS, and MP3 Recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  voice stt audio.mp3 -l my                 # Transcribe Burmese
  voice stt lecture.mp3 -l en               # Transcribe English
  voice stt voice_memo.m4a -l auto          # Auto-detect language
  voice tts "မင်္ဂလာပါ။ Hello world." -o speech.mp3 --play
  voice tts "Good morning!" -v ko-banyar -o hello.wav
  voice record my_memo.mp3                  # Record MP3 from mic
  voice record -d 5 --transcribe -l en      # Record 5s and transcribe
  voice status                              # Show models and GPU status
        """
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # STT subcommand
    stt_parser = subparsers.add_parser("stt", help="Speech-to-Text (Transcribe audio)")
    stt_parser.add_argument("audio", type=str, help="Input audio file (mp3, wav, m4a, flac, etc.)")
    stt_parser.add_argument("-l", "--lang", choices=["my", "burmese", "en", "english", "auto"], default="auto",
                            help="Language: 'my' (Burmese), 'en' (English), or 'auto' (default: auto)")
    stt_parser.add_argument("-o", "--output", type=str, default=None, help="Save transcription text to file")
    stt_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device")

    # TTS subcommand
    tts_parser = subparsers.add_parser("tts", help="Text-to-Speech (Synthesize voice)")
    tts_parser.add_argument("text", type=str, help="Burmese or English text to speak")
    tts_parser.add_argument("-v", "--voice", choices=["shoon", "thazin", "ko-banyar", "htet-myat"], default="shoon",
                            help="Voice preset (shoon, thazin, ko-banyar, htet-myat; default: shoon)")
    tts_parser.add_argument("-o", "--output", type=str, default="output.mp3", help="Output file path (.mp3 or .wav)")
    tts_parser.add_argument("-p", "--play", action="store_true", help="Play audio immediately after synthesis")
    tts_parser.add_argument("--threads", type=int, default=4, help="CPU threads for ONNX")

    # Record subcommand
    rec_parser = subparsers.add_parser("record", help="Record audio from microphone to MP3")
    rec_parser.add_argument("output", nargs="?", default=None, help="Output MP3 filename")
    rec_parser.add_argument("-d", "--duration", type=int, default=None, help="Record duration in seconds")
    rec_parser.add_argument("-t", "--transcribe", action="store_true", help="Transcribe audio after recording finishes")
    rec_parser.add_argument("-l", "--lang", choices=["my", "burmese", "en", "english", "auto"], default="auto",
                            help="Language for transcription")

    # Status subcommand
    subparsers.add_parser("status", help="Show system status and installed models")

    args = parser.parse_args()

    if args.command == "stt":
        text = run_stt(args.audio, lang=args.lang, device=args.device, output_file=args.output)
        print("\n--- Transcription ---")
        print(text)

    elif args.command == "tts":
        run_tts(args.text, voice=args.voice, output_path=args.output, threads=args.threads, play_audio=args.play)

    elif args.command == "record":
        run_record(output_path=args.output, duration=args.duration, transcribe=args.transcribe, lang=args.lang)

    elif args.command == "status":
        print_status()


if __name__ == "__main__":
    main()
