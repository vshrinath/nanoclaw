#!/usr/bin/env python3
"""
Local audio transcription using faster-whisper
Usage: python transcribe-audio.py <audio_file> [--model medium] [--output transcript.txt]
"""

import argparse
import sys
import os
from pathlib import Path

# Use the whisper-env virtual environment
# WHISPER_ENV env var overrides the default (allows host vs container usage)
WHISPER_ENV = os.environ.get('WHISPER_ENV', '/opt/whisper-env')
WHISPER_PYTHON = os.path.join(WHISPER_ENV, "bin", "python3")

# If not running with the correct Python, restart with whisper-env Python
if sys.executable != WHISPER_PYTHON and os.path.exists(WHISPER_PYTHON):
    os.execv(WHISPER_PYTHON, [WHISPER_PYTHON] + sys.argv)

# Now we're running with whisper-env Python
# Add the site-packages directory to Python path
import glob
site_packages = glob.glob(os.path.join(WHISPER_ENV, "lib/python*/site-packages"))
if site_packages:
    sys.path.insert(0, site_packages[0])

from faster_whisper import WhisperModel

def transcribe(audio_path, model_size="medium", output_path=None, device="cpu"):
    print(f"Loading {model_size} model...")
    
    # CPU with int8 quantization for efficiency
    model = WhisperModel(model_size, device=device, compute_type="int8")
    
    print(f"Transcribing: {audio_path}...")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    transcript_lines = []
    for segment in segments:
        line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
        print(line)
        transcript_lines.append(segment.text)
    
    full_transcript = " ".join(transcript_lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(full_transcript)
        print(f"\nSaved to: {output_path}")
    
    return full_transcript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using faster-whisper")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--model", "-m", default="medium", help="Model size: tiny, base, small, medium, large")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--device", "-d", default="cpu", help="Device: cpu or cuda")
    
    args = parser.parse_args()
    transcribe(args.audio_file, args.model, args.output, args.device)