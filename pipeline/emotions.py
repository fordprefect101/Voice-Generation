from __future__ import annotations

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