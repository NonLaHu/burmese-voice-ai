# 🎙️ Burmese & English Voice AI Toolkit

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12%2F13-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20Apache--2.0-green.svg)](LICENSE)

An all-in-one local Voice AI suite supporting **Burmese (မြန်မာ)** and **English**:
- **Speech-to-Text (ASR):** Local high-accuracy transcription for Burmese and English.
- **Text-to-Speech (TTS):** Natural speech synthesis with multiple character voices.
- **Microphone MP3 Recorder:** Terminal audio recorder with real-time timers and instant speech-to-text dictation.
- **GPU Optimized:** Specially tuned for consumer GPUs (e.g. **NVIDIA GeForce RTX 2050 4GB VRAM**) with 8-bit quantization and native FP16 execution, plus full CPU fallback support.

---

## 🌟 Model Architecture & Capabilities

| Capability | Engine / Architecture | Parameters | VRAM Footprint | Description |
|---|---|---|---|---|
| **English STT** | [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | 809M | ~2.1 GB (Native FP16) | State-of-the-art English & multilingual transcription at >20× real-time speed. |
| **Burmese STT** | [`BuzzASR/burmese`](https://huggingface.co/BuzzASR/burmese) | 1.55B | ~1.6 GB (8-bit Quantized) | Whisper Large v3 fine-tune with specialized Burmese vocabulary and tokenizer. |
| **Bilingual TTS** | [`YeeNyunn/nara-mini-v1`](https://huggingface.co/YeeNyunn/nara-mini-v1) | 82M | CPU (ONNX Runtime) | Kokoro-based TTS adapted for Burmese & English with 4 preset voices. |
| **Audio Recorder** | PipeWire / PulseAudio + FFmpeg LAME | N/A | System RAM | 192 kbps MP3 recorder with live console timer and direct dictation pipeline. |

---

## 🚀 Quick Start

### 1. Prerequisites
Make sure `ffmpeg` is installed on your Linux machine:
```bash
# Arch Linux
sudo pacman -S ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

### 2. Automated Installation
Run the included setup script:
```bash
git clone <repo-url>
cd ai
./setup.sh
```
`setup.sh` will:
1. Create a Python 3.12 virtual environment (`.venv`).
2. Install all dependencies from `requirements.txt`.
3. Symlink `voice`, `speech`, and `record-mp3` to `~/.local/bin` so you can use them anywhere in your terminal.
4. Verify and cache model weights locally.

---

## 💻 Unified CLI: `voice` (or `speech`)

The toolkit provides a single binary `voice` (also aliased as `speech`) that handles all four voice modes:

```
Usage: voice <command> [options]

Commands:
  stt       Transcribe audio files to text (English or Burmese)
  tts       Synthesize text into speech (Burmese, English, or mixed)
  record    Record microphone input to MP3 (with optional instant transcription)
  status    Show GPU, VRAM, and model status
```

---

### 1. Speech-to-Text (`voice stt`)

Supports all major audio formats (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.webm`):

```bash
# Transcribe English audio
voice stt lecture.mp3 -l en

# Transcribe Burmese audio
voice stt burmese_speech.wav -l my

# Auto-detect language (automatically switches between BuzzASR and Whisper Turbo)
voice stt voice_memo.m4a -l auto

# Save transcription output to a text file
voice stt meeting.mp3 -l en -o meeting_notes.txt

# Force CPU inference (useful when GPU memory is busy)
voice stt audio.wav --device cpu
```

---

### 2. Text-to-Speech (`voice tts`)

Synthesizes Burmese, English, or mixed-language sentences directly to `.mp3` or `.wav`:

```bash
# Speak English to MP3
voice tts "Welcome to the all-in-one voice assistant." -o welcome.mp3

# Speak Burmese
voice tts "မင်္ဂလာပါ။ နေကောင်းပါသလားခင်ဗျာ။" -o burmese.mp3

# Mixed language sentence
voice tts "မင်္ဂလာပါ။ Hello world, welcome to our application." -o mixed.mp3

# Choose voice character: 'shoon', 'thazin', 'ko-banyar', 'htet-myat'
voice tts "Hello, this is Ko Banyar speaking." -v ko-banyar -o banyar.wav

# Auto-play audio immediately after synthesis
voice tts "Task finished successfully." --play
```

#### Available Voices
| Voice ID | Display Name | Gender | Language Support |
|---|---|---|---|
| `shoon` | Shoon (Default) | Female | Burmese & English |
| `thazin` | Thazin | Female | Burmese & English |
| `ko-banyar` | Ko Banyar | Male | Burmese & English |
| `htet-myat` | Htet Myat | Male | Burmese & English |

---

### 3. Microphone Recording (`record-mp3` & `voice record`)

#### Dedicated Interactive Recorder (`record-mp3`)
```bash
# Interactive recording: shows live timer, press ENTER or Ctrl+C to stop and save
record-mp3 memo.mp3

# Record for a fixed duration (e.g. 15 seconds)
record-mp3 -d 15 quick_note.mp3
```

#### Dictation Mode (`voice record --transcribe`)
Record your voice from the microphone and immediately transcribe what you said:
```bash
# English dictation
voice record dictation.mp3 --transcribe -l en

# Burmese dictation
voice record dictation.mp3 --transcribe -l my
```

---

### 4. Hardware & Models Status (`voice status`)

Displays your CUDA availability, GPU device name, VRAM, and model engines:
```bash
$ voice status
============================================================
 Voice AI System Status & Installed Models
============================================================
  CUDA Available: True (NVIDIA GeForce RTX 2050, 3.7 GB)
  PyTorch:        2.15.0+cu132

  Models Available:
  1. Burmese STT (ASR):    BuzzASR/burmese (Whisper Large v3 8-bit)
  2. English STT (ASR):    openai/whisper-large-v3-turbo (FP16)
  3. Burmese & English TTS: YeeNyunn/nara-mini-v1 (Kokoro ONNX)
     - Preset Voices:      shoon, thazin, ko-banyar, htet-myat
  4. MP3 Recording Tool:   PulseAudio/PipeWire + FFmpeg lame
============================================================
```

---

## 🐍 Python API Examples

### English Transcription (`whisper-large-v3-turbo`)
```python
import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3-turbo",
    torch_dtype=dtype,
    low_cpu_mem_usage=True
).to(device).eval()

audio, sr = sf.read("audio.mp3")
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
if device == "cuda":
    inputs = inputs.half()

with torch.no_grad():
    predicted_ids = model.generate(inputs, language="en")
text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Transcription:", text)
```

### Burmese Transcription (`BuzzASR/burmese` in 8-bit)
```python
import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig

# 8-bit quantization fits comfortably in 4GB VRAM
quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = WhisperForConditionalGeneration.from_pretrained(
    "BuzzASR/burmese",
    quantization_config=quant_config,
    device_map="auto"
)
processor = WhisperProcessor.from_pretrained("BuzzASR/burmese")

audio, sr = sf.read("burmese_audio.wav")
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.half().to("cuda")

with torch.no_grad():
    ids = model.generate(inputs, num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2)
text = processor.batch_decode(ids, skip_special_tokens=True)[0]
print("Burmese Transcription:", text)
```

---

## ⌨️ Typing Burmese on Linux (Myanmar3 Keyboard Reference)

If you are using the standard Linux Burmese keyboard (`mm` layout):

- **Switch layout shortcut:** `Alt` + `Shift` or `Super` + `Shift` + `Space`.
- **Consonants:** `u` = က, `c` = ခ, `C` = ဂ, `i` = င, `p` = စ, `z` = ဆ, `Z` = ဇ, `n` = ည, `w` = တ, `x` = ထ, `K` = ဒ, `L` = ဓ, `e` = န, `y` = ပ, `P` = ဖ, `b` = ဗ, `r` = မ, `B` = ယ, `&` = ရ, `v` = လ, `W` = ဝ, `o` = သ, `[` = ဟ, `t` = အ
- **Vowels & Signs:** `a` = ေ, `m` = ာ, `g` = ါ, `d` = ိ, `D` = ီ, `k` = ု, `l` = ူ, `h` = ့, `H` = ံ, `;` = း, `f` = ် (အသတ်)
- **Medials (ဗျည်းတွဲ):** `s` = ျ (ယပင့်), `j` = ြ (ရရစ်), `G` (`Shift+g`) = ွ (ဝဆွဲ), `S` (`Shift+s`) = ှ (ဟထိုး)
- **Consonant Stacking (ပါဌ်ဆင့်):** Type `Shift` + `f` (`္` virama). For example, `u` + `Shift+f` + `u` $\rightarrow$ `က္က`.

---

## 📄 License & Acknowledgments

- **BuzzASR:** [BuzzASR/burmese](https://huggingface.co/BuzzASR/burmese) (MIT License) - Swarm of 100+ Monolingual Speech Recognition Models ([arXiv:2609.09554](https://arxiv.org/abs/2609.09554)).
- **Whisper:** [OpenAI Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT License).
- **Nara Mini v1:** [YeeNyunn/nara-mini-v1](https://huggingface.co/YeeNyunn/nara-mini-v1) (Apache 2.0 License) - Kokoro-based Burmese and English speech synthesis.
- **Kokoro:** [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0 License).
