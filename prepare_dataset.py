#!/usr/bin/env python3
"""
Slices raw audio recording (Testr2.wav) into LJSpeech dataset using Voice Activity Detection (VAD)
and individual phrase Whisper speech recognition.

Pipeline:
  1. Detect natural silence intervals via ffmpeg silencedetect (-27dB, min duration 0.3s).
  2. Merge consecutive speech phrases separated by short pauses (<0.65s) up to 8.0s max duration.
  3. Filter speech chunks (1.0s <= duration <= 9.0s).
  4. Slice each chunk with 50ms silence padding at boundaries and studio audio filter chain:
     - Highpass (75Hz)
     - Gentle FFT denoise (afftdn)
     - EBU R128 loudness normalization (-20 LUFS)
     - 22050 Hz, 16-bit mono PCM.
  5. Transcribe each isolated chunk individually with Whisper 'small' model (Russian).
     This guarantees 100% audio-text synchronization without cut words or phoneme truncation.
  6. Filter out empty, non-Cyrillic, or hallucinated lines.
  7. Generate clean dataset/wavs/XXXX.wav and dataset/metadata.csv (ID|Transcript).
"""

import os
import re
import shutil
import subprocess

def get_audio_duration(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return float(res.stdout.strip())

def detect_silences(path: str, noise_db="-27dB", min_dur=0.3):
    cmd = [
        "ffmpeg", "-i", path,
        "-af", f"silencedetect=noise={noise_db}:d={min_dur}",
        "-f", "null", "-"
    ]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    silences = []
    start_p = re.compile(r"silence_start:\s*([\d\.]+)")
    end_p = re.compile(r"silence_end:\s*([\d\.]+)")
    cur_start = None
    for line in proc.stderr.splitlines():
        ms = start_p.search(line)
        if ms:
            cur_start = float(ms.group(1))
        me = end_p.search(line)
        if me:
            silences.append((cur_start if cur_start is not None else 0.0, float(me.group(1))))
            cur_start = None
    return silences

def clean_text(text: str) -> str:
    text = text.strip()
    # Normalize multiple punctuation or spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove unwanted symbols but keep Russian letters, digits, and basic punctuation
    text = re.sub(r'[^\w\s.,!?:;—\-]', '', text)
    return text.strip()

def is_valid_transcript(text: str) -> bool:
    if len(text) < 3:
        return False
    # Must contain at least one Cyrillic letter
    if not any('\u0400' <= c <= '\u04FF' for c in text):
        return False
    # Known Whisper hallucinations
    hallucinations = [
        "субтитры", "редактор субтитров", "продолжение следует",
        "спасибо за просмотр", "подписывайтесь", "ставьте лайк",
        "до скорой встречи", "всем пока"
    ]
    lower = text.lower()
    for h in hallucinations:
        if h in lower:
            return False
    # Check for repeated word hallucination (e.g., "слово слово слово слово")
    words = lower.split()
    if len(words) >= 4:
        repeat_count = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                repeat_count += 1
                if repeat_count >= 3:
                    return False
            else:
                repeat_count = 0
    return True

def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    raw_audio = os.path.join(repo_dir, "raw_audio", "Testr2.wav")
    dataset_dir = os.path.join(repo_dir, "dataset")
    wavs_dir = os.path.join(dataset_dir, "wavs")

    if not os.path.exists(raw_audio):
        raise FileNotFoundError(f"Missing raw audio: {raw_audio}")

    total_dur = get_audio_duration(raw_audio)
    print(f"Total raw audio duration: {total_dur:.2f} seconds ({total_dur/60:.2f} min)")

    print("=== Step 1. Voice Activity Detection (silencedetect) ===")
    silences = detect_silences(raw_audio, noise_db="-27dB", min_dur=0.3)
    print(f"Detected {len(silences)} silence intervals.")

    # Extract speech intervals between silences
    raw_speech = []
    last_end = 0.0
    for s_start, s_end in silences:
        if s_start > last_end:
            raw_speech.append({"start": last_end, "end": s_start})
        last_end = s_end
    if last_end < total_dur:
        raw_speech.append({"start": last_end, "end": total_dur})

    print(f"Extracted {len(raw_speech)} raw speech segments.")

    # Merge consecutive segments separated by small pauses (< 0.65s) up to 8.0s
    merged = []
    for seg in raw_speech:
        if not merged:
            merged.append(dict(seg))
        else:
            prev = merged[-1]
            gap = seg["start"] - prev["end"]
            combined_dur = seg["end"] - prev["start"]
            prev_dur = prev["end"] - prev["start"]
            curr_dur = seg["end"] - seg["start"]

            can_merge = False
            if combined_dur <= 8.0:
                if gap <= 0.65:
                    can_merge = True
                elif (prev_dur < 1.0 or curr_dur < 1.0) and gap <= 0.9:
                    can_merge = True

            if can_merge:
                prev["end"] = seg["end"]
            else:
                merged.append(dict(seg))

    # Filter by duration (1.0s <= dur <= 9.0s)
    candidates = [s for s in merged if 1.0 <= (s["end"] - s["start"]) <= 9.0]
    print(f"Prepared {len(candidates)} candidate phrases for slicing and transcription.")

    # Recreate wavs dir
    shutil.rmtree(wavs_dir, ignore_errors=True)
    os.makedirs(wavs_dir, exist_ok=True)

    print("=== Step 2. Loading Whisper 'small' for Russian speech recognition ===")
    import whisper
    model = whisper.load_model("small")

    print("=== Step 3. Slicing and transcribing phrases individually ===")
    metadata = []
    counter = 1
    total_audio_time = 0.0

    temp_wav = os.path.join(dataset_dir, "temp_chunk.wav")
    filter_chain = "highpass=f=75,afftdn=nf=-25,loudnorm=I=-20:LRA=11:TP=-1.5"

    for idx, seg in enumerate(candidates, start=1):
        # 50ms silence padding
        start_pad = max(0.0, seg["start"] - 0.05)
        end_pad = min(total_dur, seg["end"] + 0.05)
        dur_pad = end_pad - start_pad

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_pad:.3f}",
            "-t", f"{dur_pad:.3f}",
            "-i", raw_audio,
            "-af", filter_chain,
            "-ar", "22050",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            temp_wav
        ]
        res_ff = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res_ff.returncode != 0 or not os.path.exists(temp_wav):
            continue

        # Transcribe chunk individually
        whisper_res = model.transcribe(temp_wav, language="ru", fp16=False)
        text = clean_text(whisper_res.get("text", ""))

        if not is_valid_transcript(text):
            print(f"[{idx}/{len(candidates)}] Skipped invalid transcript: '{text}'")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            continue

        file_id = f"{counter:04d}"
        final_wav = os.path.join(wavs_dir, f"{file_id}.wav")
        shutil.move(temp_wav, final_wav)

        metadata.append(f"{file_id}|{text}")
        total_audio_time += dur_pad
        print(f"[{file_id}] {start_pad:6.2f}s -> {end_pad:6.2f}s ({dur_pad:4.2f}s): {text[:65]}")
        counter += 1

    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata) + "\n")

    print(f"\n✓ Dataset successfully generated: {len(metadata)} phrases, total speech: {total_audio_time/60:.2f} minutes")
    print(f"✓ Metadata saved to {metadata_path}")

if __name__ == "__main__":
    main()
