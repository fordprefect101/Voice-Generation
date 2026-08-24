"""Punctuation + pause post-processing for actable TTS lines."""
from __future__ import annotations

import re

from pipeline.models import EnrichedLine


def normalize_pause_punctuation(text: str) -> str:
    """Make pause cues Qwen-friendly: hyphen beats -> ellipsis / em dash."""
    t = text
    t = re.sub(r"\s+-\s+", " — ", t)
    t = re.sub(r"\s+-(\s|$)", r"…\1", t)
    t = t.replace("...", "…")
    t = re.sub(r"[.]{4,}", "…", t)
    t = re.sub(r"[…]{2,}", "…", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def diversify_pauses(lines: list[EnrichedLine]) -> list[EnrichedLine]:
    """If pauses collapsed to one value, nudge by simple dialogue context."""
    afters = [L.pause_after_ms for L in lines]
    collapsed = len(set(afters)) <= 2 and len(lines) >= 4

    for i, L in enumerate(lines):
        text = L.text
        prev = lines[i - 1].text if i else ""
        pb = clamp(L.pause_before_ms, 0, 900)
        pa = clamp(L.pause_after_ms, 120, 900)

        if collapsed or pa == 350:
            if text.rstrip().endswith("?"):
                pa = 520
            elif text.rstrip().endswith("!"):
                pa = 260
            elif "…" in text or "—" in text:
                pa = 400
            else:
                pa = 240 + (i % 5) * 45

            if prev.rstrip().endswith("?") and i > 0:
                lines[i - 1].pause_after_ms = max(lines[i - 1].pause_after_ms, 500)

        if prev.rstrip().endswith("?") and pb == 0:
            pb = 120

        L.pause_before_ms = pb
        L.pause_after_ms = pa
        L.text = normalize_pause_punctuation(L.text)
    return lines
