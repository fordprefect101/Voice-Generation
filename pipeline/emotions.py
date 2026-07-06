from __future__ import annotations

EMOTION_TAGS: dict[str, str | None] = {
    "neutral": None,
    "curious": "<curious>",
    "playful": "<chuckle>",
    "excited": "<excited>",
    "warm": None,  # voice description carries warmth; avoid extra tag on closes
}

def apply_emotion_tag(text: str, emotion: str) -> str:
    tag = EMOTION_TAGS.get(emotion)
    if not tag:
        return text
    return f"{tag} {text}"

def default_pause_after(text: str, emotion: str) -> int:
    stripped = text.rstrip()

    if stripped.endswith("?"):
        return 550
    if emotion == "playful" and len(stripped) < 100:
        return 300
    return 450