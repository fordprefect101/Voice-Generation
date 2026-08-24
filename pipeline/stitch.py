from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from pydub import AudioSegment

from pipeline.emotions import default_pause_after
from pipeline.mix import (
    append_with_crossfade,
    ensure_placeholder_beds,
    load_mix_config,
    observation_hold_ms,
    overlay_section_beds,
    trim_silence,
)
from pipeline.models import EnrichedLine, load_lines_json


def clip_path(clips_dir: Path, line: EnrichedLine) -> Path:
    return clips_dir / f"{line.index:03d}_{line.speaker}.wav"


def _pause_after_ms(line: EnrichedLine) -> int:
    base = line.pause_after_ms or default_pause_after(line.text, line.emotion)
    hold = observation_hold_ms(line.text)
    return max(int(base), int(hold))


def build_episode(
    lines: list[EnrichedLine],
    clips_dir: str | Path,
    *,
    with_beds: bool = True,
) -> AudioSegment:
    clips_dir = Path(clips_dir)
    ensure_placeholder_beds()
    cfg = load_mix_config()
    episode = AudioSegment.empty()
    # Track section spans for beds: (start_ms, end_ms, section)
    spans: list[tuple[int, int, str]] = []
    section = "default"

    for line in lines:
        path = clip_path(clips_dir, line)
        if not path.exists():
            raise FileNotFoundError(f"Missing clip: {path}")

        clip = trim_silence(AudioSegment.from_wav(path))
        start = len(episode)

        if line.pause_before_ms > 0:
            episode += AudioSegment.silent(duration=line.pause_before_ms)

        episode = append_with_crossfade(episode, clip, int(cfg["crossfade_ms"]))
        episode += AudioSegment.silent(duration=_pause_after_ms(line))

        # Heuristic section from emotion / text for CosyVoice path
        low = line.text.lower()
        if any(k in low for k in ("feast", "dal", "baati", "hungry", "thali", "food")):
            section = "food"
        elif any(k in low for k in ("dancer", "ghoomar", "kalbelia", "dhol", "puppet")):
            section = "performance"
        elif any(k in low for k in ("potter", "weaver", "henna", "artisan", "craft")):
            section = "craft"
        elif line.index <= 2:
            section = "arrival"
        elif "chokhi" in low or "hamlet" in low:
            section = "culture"
        spans.append((start, len(episode), section))

    if with_beds and spans:
        episode = overlay_section_beds(episode, spans)
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
    *,
    with_beds: bool = True,
) -> str:
    lines = load_lines_json(lines_json)
    episode = build_episode(lines, clips_dir, with_beds=with_beds)

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
    parser.add_argument("--no-beds", action="store_true")
    args = parser.parse_args()

    out = stitch_episode(
        args.lines, args.clips, args.output, with_beds=not args.no_beds
    )
    print(f"Wrote {out}")
