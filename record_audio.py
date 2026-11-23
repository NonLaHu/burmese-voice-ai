#!/usr/bin/env python3
"""
Simple MP3 Audio Recorder using system PulseAudio / PipeWire through ffmpeg.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def record_mp3(output_path: str, duration: int = None, sample_rate: int = 44100):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel", "error",
        "-f", "pulse",
        "-i", "default",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "libmp3lame",
        "-b:a", "192k",
    ]
    if duration:
        cmd.extend(["-t", str(duration)])
    cmd.append(str(output_path))

    proc = subprocess.Popen(cmd)
    start_time = time.time()

    print(f"\n🎙️  Recording audio to: {output_path.name}")
    if duration:
        print(f"⏳ Recording for {duration} seconds... (Press Ctrl+C to stop earlier)")
    else:
        print("🔴 Recording in progress... Press [ENTER] or [Ctrl+C] to stop and save.")

    stopped = False

    def handle_sigint(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        import select
        while proc.poll() is None and not stopped:
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            sys.stdout.write(f"\r🔴 [{mins:02d}:{secs:02d}] Recording... (Press ENTER to finish) ")
            sys.stdout.flush()

            # Check if user pressed Enter
            rlist, _, _ = select.select([sys.stdin], [], [], 0.3)
            if rlist:
                sys.stdin.readline()
                break
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    total_time = int(time.time() - start_time)
    mins, secs = divmod(total_time, 60)
    print(f"\n\n✅ Recording finished! [{mins:02d}:{secs:02d}]")
    if output_path.exists():
        size_kb = output_path.stat().st_size / 1024.0
        print(f"📁 Saved: {output_path} ({size_kb:.1f} KB)")
    else:
        print("⚠️ Warning: Output file was not created.", file=sys.stderr)

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Record MP3 from default microphone")
    parser.add_argument("output", nargs="?", default=None, help="Output MP3 file path (default: recording_<timestamp>.mp3)")
    parser.add_argument("-d", "--duration", type=int, default=None, help="Record duration in seconds")
    parser.add_argument("-r", "--sample-rate", type=int, default=44100, help="Sample rate (default: 44100)")

    args = parser.parse_args()

    if not args.output:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output = f"recording_{timestamp}.mp3"
    else:
        output = args.output
        if not output.endswith(".mp3"):
            output += ".mp3"

    record_mp3(output, duration=args.duration, sample_rate=args.sample_rate)


if __name__ == "__main__":
    main()
