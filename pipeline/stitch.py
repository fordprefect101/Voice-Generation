from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from pipeline.emotions import default_pause_after
from pipeline.models import EnrichedLine, load_lines_json

def clip_path(clips_dir: Path, line: EnrichedLine) -> Path:
    return clips_dir / f"{line.index:03d}_{line.speaker}.wav"

def trim_silence(
    audio: AudioSegment,
    silence_thresh: int = -40,
    min_silence_len: int = 250,
    padding_ms: int = 50,
) -> AudioSegment:
    if len(audio) == 0:
        return audio

    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )
    if not nonsilent:
        return audio

    start = max(nonsilent[0][0] - padding_ms, 0)
    end = min(nonsilent[-1][1] + padding_ms, len(audio))
    return audio[start:end]

def _pause_after_ms(line: EnrichedLine) -> int:
    if line.pause_after_ms:
        return line.pause_after_ms
    return default_pause_after(line.text, line.emotion)

def build_episode(
    lines: list[EnrichedLine],
    clips_dir: str | Path,
) -> AudioSegment:
    clips_dir = Path(clips_dir)
    episode = AudioSegment.empty()

    for line in lines:
        path = clip_path(clips_dir, line)
        if not path.exists():
            raise FileNotFoundError(f"Missing clip: {path}")

        clip = trim_silence(AudioSegment.from_wav(path))

        if line.pause_before_ms > 0:
            episode += AudioSegment.silent(duration=line.pause_before_ms)
        episode += clip
        episode += AudioSegment.silent(duration=_pause_after_ms(line))

    return episode

def normalize_loudness(input_path: str | Path, output_path: str | Path) -> str:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(output_path)

def stitch_episode(
    lines_json: str | Path,
    clips_dir: str | Path,
    output_mp3: str | Path,
) -> str:
    lines = load_lines_json(lines_json)
    episode = build_episode(lines, clips_dir)

    with tempfile.TemporaryDirectory() as tmp:
        temp_wav = Path(tmp) / "episode.wav"
        episode.export(temp_wav, format="wav")
        return normalize_loudness(temp_wav, output_mp3)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stitch clips into final MP3")
    parser.add_argument("--lines", default="output/lines.json")
    parser.add_argument("--clips", default="output/clips")
    parser.add_argument("--output", default="output/final.mp3")
    args = parser.parse_args()

    out = stitch_episode(args.lines, args.clips, args.output)
    print(f"Wrote {out}")