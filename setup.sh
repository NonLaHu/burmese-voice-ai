#!/usr/bin/env bash
# ==============================================================================
# Voice AI (Burmese & English) - Environment & Tool Setup
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up Burmese & English Voice AI Toolkit ==="

# 1. Create Python virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment (.venv)..."
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.12 .venv || uv venv .venv
    else
        python3 -m venv .venv
    fi
fi

# 2. Install requirements
echo "[*] Installing Python dependencies..."
if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv -r requirements.txt
else
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

# 3. Make scripts executable
echo "[*] Making CLI executables..."
chmod +x voice record-mp3 buzzasr nara-tts voice_cli.py record_audio.py transcribe_burmese.py synthesize_burmese.py

# 4. Create symlinks in ~/.local/bin
mkdir -p "$HOME/.local/bin"
echo "[*] Linking commands to $HOME/.local/bin/..."
ln -sf "$SCRIPT_DIR/voice" "$HOME/.local/bin/voice"
ln -sf "$SCRIPT_DIR/voice" "$HOME/.local/bin/speech"
ln -sf "$SCRIPT_DIR/record-mp3" "$HOME/.local/bin/record-mp3"
ln -sf "$SCRIPT_DIR/buzzasr" "$HOME/.local/bin/buzzasr"
ln -sf "$SCRIPT_DIR/nara-tts" "$HOME/.local/bin/nara-tts"

# 5. Download model weights if missing
echo "[*] Checking models..."
.venv/bin/python -c '
from huggingface_hub import snapshot_download
from pathlib import Path

# Check Whisper Turbo
print("Checking openai/whisper-large-v3-turbo...")
snapshot_download("openai/whisper-large-v3-turbo")

# Check BuzzASR Burmese
print("Checking BuzzASR/burmese...")
snapshot_download("BuzzASR/burmese")

# Check Nara Mini v1
nara_onnx = Path("nara-mini-v1/onnx")
if not nara_onnx.exists():
    print("Downloading YeeNyunn/nara-mini-v1 into nara-mini-v1/...")
    snapshot_download("YeeNyunn/nara-mini-v1", local_dir="nara-mini-v1")
'

echo "=== Setup complete! ==="
echo "You can now run:"
echo "  voice status"
echo "  voice stt <audio-file>"
echo "  voice tts '<text>'"
echo "  record-mp3 [filename.mp3]"
