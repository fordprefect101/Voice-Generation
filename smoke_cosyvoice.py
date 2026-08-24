#!/usr/bin/env python3
"""Smoke-test CosyVoice3 instruct2 with local identity refs."""
from __future__ import annotations

from pathlib import Path

from pipeline.emotions import build_cosyvoice_instruct
from pipeline.generate_cosyvoice import generate_line_audio, load_cosyvoice_model
from pipeline.models import EnrichedLine

OUT = Path("output/smoke_cosyvoice")
OUT.mkdir(parents=True, exist_ok=True)

jobs = [
    EnrichedLine(
        index=1,
        speaker="M",
        text="Just take a look around… This… is Chokhi Dhani.",
        emotion="wonder",
        pause_before_ms=0,
        pause_after_ms=400,
    ),
    EnrichedLine(
        index=2,
        speaker="F",
        text="It's not an ancient village — it's a living recreation.",
        emotion="explanatory",
        pause_before_ms=0,
        pause_after_ms=400,
    ),
]

print("Instruct previews:")
for line in jobs:
    print(
        f"  {line.speaker}/{line.emotion}: "
        f"{build_cosyvoice_instruct(line.emotion, speaker=line.speaker, emphasize=line.emphasize)}"
    )

load_cosyvoice_model()
for line in jobs:
    path = OUT / f"{line.index:03d}_{line.speaker}.wav"
    print(f"Generating {path}...")
    generate_line_audio(line, path)
    print(f"  -> {path}")

print("Done. Listen under output/smoke_cosyvoice/")
