#!/usr/bin/env python3
"""
Slices raw audio recording (Testr2.wav) into LJSpeech dataset using Whisper.
Applies:
  1. Highpass filter (75 Hz) to eliminate sub-bass rumble
  2. Gentle FFT denoising to clean pauses
  3. EBU R128 studio loudness normalization (-20 LUFS)
  4. Whisper 'small' model transcription for precise Russian text & timestamps
Produces:
  dataset/wavs/0001.wav ... (22050 Hz, 16-bit mono)
  dataset/metadata.csv (ID|Transcript)
"""

import os
import re
import shutil
import subprocess

def clean_text(text: str) -> str:
    text = text.strip()
    # Normalize multiple punctuation or spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove unwanted symbols but keep Russian letters, digits, and basic punctuation
    text = re.sub(r'[^\w\s.,!?:;—\-]', '', text)
    return text.strip()

def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    raw_audio = os.path.join(repo_dir, "raw_audio", "Testr2.wav")
    dataset_dir = os.path.join(repo_dir, "dataset")
    wavs_dir = os.path.join(dataset_dir, "wavs")

    if not os.path.exists(raw_audio):
        raise FileNotFoundError(f"Missing raw audio: {raw_audio}")

    # Recreate wavs dir
    shutil.rmtree(wavs_dir, ignore_errors=True)
    os.makedirs(wavs_dir, exist_ok=True)

    print(f"=== Loading Whisper 'small' for Russian speech recognition ===")
    import whisper
    model = whisper.load_model("small")

    print(f"Transcribing {raw_audio}...")
    result = model.transcribe(raw_audio, language="ru", word_timestamps=False, verbose=False)
    segments = result.get("segments", [])
    print(f"Whisper detected {len(segments)} initial segments.")

    metadata = []
    counter = 1
    total_audio_time = 0.0

    for seg in segments:
        text = clean_text(seg.get("text", ""))
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        dur = end - start

        # Filter out too short (< 0.9s), too long (> 11.0s), or empty/hallucinated segments
        if dur < 0.9 or dur > 11.0:
            continue
        if len(text) < 4 or not any('\u0400' <= c <= '\u04FF' for c in text):
            continue

        # Add 0.05s padding safely
        start_pad = max(0.0, start - 0.05)
        dur_pad = dur + 0.10

        file_id = f"{counter:04d}"
        out_wav = os.path.join(wavs_dir, f"{file_id}.wav")

        # Studio audio filtering: High-pass (75Hz) + gentle FFT denoise + Loudnorm (-20 LUFS)
        filter_chain = "highpass=f=75,afftdn=nf=-25,loudnorm=I=-20:LRA=11:TP=-1.5"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_pad:.3f}",
            "-t", f"{dur_pad:.3f}",
            "-i", raw_audio,
            "-af", filter_chain,
            "-ar", "22050",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            out_wav
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        metadata.append(f"{file_id}|{text}")
        total_audio_time += dur_pad
        print(f"[{file_id}] {start_pad:6.2f}s -> {start_pad + dur_pad:6.2f}s ({dur_pad:4.2f}s): {text[:60]}")
        counter += 1

    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata) + "\n")

    print(f"\n✓ Dataset generated: {len(metadata)} phrases, total duration: {total_audio_time/60:.2f} minutes")
    print(f"✓ Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
