#!/usr/bin/env python3
"""
Slices raw audio recordings into LJSpeech format dataset using Whisper for alignment.
Produces:
  dataset/wavs/0001.wav ... 0050.wav (22050 Hz, 16-bit mono)
  dataset/metadata.csv (ID|Normalized Text)
"""

import json
import os
import re
import subprocess
import difflib
import wave

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())

def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    sentences_file = os.path.join(repo_dir, "sentences.json")
    dataset_dir = os.path.join(repo_dir, "dataset")
    wavs_dir = os.path.join(dataset_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    with open(sentences_file, "r", encoding="utf-8") as f:
        sentences_data = json.load(f)

    print("=== Loading Whisper model for precise audio alignment ===")
    import whisper
    model = whisper.load_model("base")

    metadata = []
    counter = 1

    for part_key in ["part0", "part1", "part2"]:
        audio_path = os.path.join(repo_dir, "raw_audio", f"{part_key}.wav")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Missing audio: {audio_path}")

        target_texts = sentences_data.get(part_key, [])
        print(f"\nProcessing {part_key} ({len(target_texts)} target phrases)...")

        # Get audio duration
        with wave.open(audio_path, "r") as w:
            total_dur = w.getnframes() / float(w.getframerate())
        print(f"Total audio duration: {total_dur:.2f}s")

        # Transcribe with Whisper
        result = model.transcribe(audio_path, language="ru", word_timestamps=True)
        segments = result.get("segments", [])

        # Flatten words
        all_words = []
        for seg in segments:
            for w in seg.get("words", []):
                cleaned = normalize_text(w["word"])
                if cleaned:
                    all_words.append({
                        "word": cleaned,
                        "raw_word": w["word"],
                        "start": w["start"],
                        "end": w["end"]
                    })

        print(f"Whisper recognized {len(segments)} segments and {len(all_words)} words.")

        # Sequential alignment
        word_idx = 0
        seg_idx = 0

        for t_idx, target_text in enumerate(target_texts):
            file_id = f"{counter:04d}"
            out_wav = os.path.join(wavs_dir, f"{file_id}.wav")
            target_norm = normalize_text(target_text)
            target_words = target_norm.split()
            n_target = len(target_words)

            start_time = None
            end_time = None
            matched_via = "word"

            # 1. Try word matching
            if word_idx < len(all_words) and n_target > 0:
                first_target_word = target_words[0]
                last_target_word = target_words[-1]

                # Look ahead up to 10 words for start
                best_start_offset = None
                for offset in range(min(12, len(all_words) - word_idx)):
                    sim = difflib.SequenceMatcher(None, first_target_word, all_words[word_idx + offset]["word"]).ratio()
                    if sim > 0.7:
                        best_start_offset = offset
                        break

                if best_start_offset is not None:
                    actual_start_idx = word_idx + best_start_offset
                    start_time = max(0.0, all_words[actual_start_idx]["start"] - 0.08)

                    # Look ahead for end
                    est_end_idx = min(len(all_words) - 1, actual_start_idx + n_target - 1)
                    best_end_offset = None
                    for offset in range(-3, 6):
                        check_idx = est_end_idx + offset
                        if 0 <= check_idx < len(all_words):
                            sim = difflib.SequenceMatcher(None, last_target_word, all_words[check_idx]["word"]).ratio()
                            if sim > 0.7:
                                best_end_offset = check_idx
                                break

                    if best_end_offset is not None:
                        end_time = all_words[best_end_offset]["end"] + 0.10
                        word_idx = best_end_offset + 1
                    else:
                        end_time = all_words[est_end_idx]["end"] + 0.12
                        word_idx = est_end_idx + 1

            # 2. Fallback to segment matching if word matching didn't yield times
            if start_time is None or end_time is None:
                matched_via = "segment"
                best_seg = None
                best_ratio = 0.0
                for s_i in range(seg_idx, min(seg_idx + 3, len(segments))):
                    seg_norm = normalize_text(segments[s_i]["text"])
                    ratio = difflib.SequenceMatcher(None, target_norm, seg_norm).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_seg = s_i

                if best_seg is not None and best_ratio > 0.4:
                    start_time = max(0.0, segments[best_seg]["start"] - 0.08)
                    end_time = segments[best_seg]["end"] + 0.10
                    seg_idx = best_seg + 1
                elif seg_idx < len(segments):
                    # Direct sequential index fallback
                    start_time = max(0.0, segments[seg_idx]["start"] - 0.08)
                    end_time = segments[seg_idx]["end"] + 0.10
                    seg_idx += 1
                else:
                    # Last resort proportional fallback
                    start_time = max(0.0, (t_idx / len(target_texts)) * total_dur)
                    end_time = min(total_dur, ((t_idx + 1) / len(target_texts)) * total_dur)

            dur = max(0.1, end_time - start_time)

            # Cut using ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_time:.3f}",
                "-t", f"{dur:.3f}",
                "-i", audio_path,
                "-ar", "22050",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                out_wav
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            print(f"[{file_id}] {start_time:6.2f}s -> {end_time:6.2f}s (dur {dur:4.2f}s) [{matched_via:7s}] : {target_text[:45]}...")
            metadata.append(f"{file_id}|{target_text}")
            counter += 1

    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata) + "\n")

    print(f"\n✓ Completed: {len(metadata)} samples written to {metadata_path}")

if __name__ == "__main__":
    main()
