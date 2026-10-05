#!/usr/bin/env python3
"""Mix a MOSS one-pass episode: observation holds + subtle ambience beds.

Usage:
  python run_mix_moss.py --episode /path/to/episode.wav \\
      --cues output/chokhi_dhani_moss.cues.json -o output/episode_mixed.mp3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydub import AudioSegment

from pipeline.mix import (
    ensure_placeholder_beds,
    insert_holds_by_word_share,
    overlay_section_beds,
)
from pipeline.stitch import normalize_loudness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, help="MOSS episode.wav")
    parser.add_argument("--cues", required=True, help="*.cues.json from run_moss_director")
    parser.add_argument("-o", "--output", default="output/episode_mixed.mp3")
    parser.add_argument(
        "--no-holds",
        action="store_true",
        help="Skip approximate observation holds",
    )
    parser.add_argument(
        "--no-beds",
        action="store_true",
        help="Skip ambience underlay",
    )
    args = parser.parse_args()

    episode_path = Path(args.episode)
    cues = json.loads(Path(args.cues).read_text(encoding="utf-8"))
    ensure_placeholder_beds()

    audio = AudioSegment.from_file(episode_path)
    texts = [str(c.get("text") or "") for c in cues]
    holds = [0 if args.no_holds else int(c.get("hold_after_ms") or 0) for c in cues]

    if any(holds):
        print(f"Inserting {sum(1 for h in holds if h)} observation holds (approx)...")
        audio = insert_holds_by_word_share(audio, texts, holds)

    if not args.no_beds:
        # Prefer section beds if cues span the episode by word share
        words = [max(1, len(t.split())) for t in texts]
        total = sum(words) or 1
        segments: list[tuple[int, int, str]] = []
        cursor = 0
        for w, c in zip(words, cues):
            dur = int(round(len(audio) * (w / total)))
            section = str(c.get("section") or "default")
            segments.append((cursor, min(len(audio), cursor + dur), section))
            cursor += dur
        if segments:
            segments[-1] = (segments[-1][0], len(audio), segments[-1][2])
        print(f"Overlaying section beds ({len(segments)} segments)...")
        audio = overlay_section_beds(audio, segments)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.wav")
    audio.export(raw, format="wav")
    if out.suffix.lower() == ".wav":
        raw.replace(out)
        print(f"Wrote {out}")
    else:
        normalize_loudness(raw, out)
        raw.unlink(missing_ok=True)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
