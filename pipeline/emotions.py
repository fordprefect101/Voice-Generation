from __future__ import annotations
import re

EMOTION_INSTRUCT: dict[str, str] = {
    "neutral": "neutral, calm delivery",
    "curious": "curious, lightly questioning, discovery tone",
    "playful": "playful, amused, slight smile in the voice",
    "excited": "excited but not shouting, higher energy",
    "warm": "warm, reflective, soft smile",
}

def build_instruct(base_voice: str, emotion: str) -> str:
    style = EMOTION_INSTRUCT.get(emotion, EMOTION_INSTRUCT["neutral"])
    return f"{base_voice.rstrip()}, {style}"

def default_pause_after(text: str, emotion: str) -> int:
    stripped = text.rstrip()
    if stripped.endswith("?"):
        return 550
    if emotion == "playful" and len(stripped) < 100:
        return 300
    return 450

def text_for_tts(text: str) -> str:
    """Keep delivery intent; remove chars models sometimes speak aloud."""
    t = text
    # ellipsis → natural pause cue in prose (not "dot dot dot")
    t = t.replace("...", ",")   # or " — "
    t = t.replace("…", ",")
    # excitement/question → sentence end (prosody from emotion refs / wording)
    t = t.replace("!", ".")
    t = t.replace("?", ".")
    t = re.sub(r"\s+", " ", t).strip()
    return t