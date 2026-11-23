"""Run Nara Mini v1's released ONNX voices on CPU."""
import argparse
import json
from pathlib import Path
import re
import wave

import numpy as np
import onnxruntime as ort

from frontend.english import to_tokens
from frontend.my_numbers import verbalize
from frontend.nara_frontend import phonemize, prepare_names


def split_speech(text, limit=200):
    units = []
    for sentence in re.split(r'(?<=[။!?…\n])\s*', text):
        for clause in re.split(r'(?<=၊)\s*', sentence):
            while len(clause) > limit:
                cut = clause.rfind(' ', 0, limit)
                if cut <= 0:
                    cut = limit
                units.append(clause[:cut])
                clause = clause[cut:].strip()
            if clause.strip():
                units.append(clause.strip())
    pieces = []
    current = ''
    for unit in units:
        if current and len(current) + 1 + len(unit) > limit:
            pieces.append(current)
            current = unit
        else:
            current = f'{current} {unit}'.strip()
    if current:
        pieces.append(current)
    return pieces


def synthesize(model_folder, text, threads=4, normalize_numbers=True):
    if threads < 1:
        raise ValueError('Thread count must be positive')
    if not text.strip():
        raise ValueError('Text must contain speech')
    metadata = json.loads((model_folder / 'voice.json').read_text(encoding='utf-8'))
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    encoder = ort.InferenceSession(str(model_folder / 'encoder.onnx'), options,
                                  providers=['CPUExecutionProvider'])
    decoder = ort.InferenceSession(str(model_folder / 'decoder.onnx'), options,
                                  providers=['CPUExecutionProvider'])
    if normalize_numbers:
        text = verbalize(text)
    audio = []
    for piece in split_speech(text):
        chars = phonemize(to_tokens(prepare_names(piece)))
        tokens = [metadata['vocab'][char] for char in chars if char in metadata['vocab']]
        if not tokens:
            raise ValueError('A passage produced no phonemes')
        if len(tokens) > 508:
            raise ValueError('A passage exceeds the phoneme context limit')
        index = int(np.argmin(np.abs(np.asarray(metadata['centers']) - len(tokens))))
        style = np.asarray([metadata['styles'][index]], dtype=np.float32)
        durations, prosody, encoded_text = encoder.run(
            ['durations', 'prosody', 'text'],
            {'tokens': np.asarray([[0, *tokens, 0]], dtype=np.int64), 'style': style})
        if durations.min() < 1 or not 1 <= durations.sum() <= 20000:
            raise ValueError('Invalid predicted duration')
        samples = decoder.run(None, {
            'prosody': np.repeat(prosody, durations[0], axis=2),
            'text': np.repeat(encoded_text, durations[0], axis=2),
            'style': style,
        })[0].reshape(-1)
        if samples.size < 2400 or not np.isfinite(samples).all():
            raise ValueError('The model produced invalid audio')
        if audio:
            audio.append(np.zeros(4800, dtype=np.float32))
        audio.append(samples)
    return np.clip(np.concatenate(audio), -1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--voice', choices=['shoon', 'thazin', 'ko-banyar', 'htet-myat'], default='shoon')
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', type=Path, default=Path('speech.wav'))
    parser.add_argument('--model-dir', type=Path, default=Path(__file__).resolve().parent / 'onnx')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--no-normalize-numbers', action='store_true')
    args = parser.parse_args()
    samples = synthesize(args.model_dir / args.voice, args.text, args.threads,
                         normalize_numbers=not args.no_normalize_numbers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), 'wb') as output:
        output.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        output.writeframes((samples * 32767).astype('<i2').tobytes())
    print(f'Saved {samples.size / 24000:.2f}s of audio to {args.output}')


if __name__ == '__main__':
    main()
