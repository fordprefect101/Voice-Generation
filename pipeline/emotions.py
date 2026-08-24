from __future__ import annotations

import re

from pipeline.emotion_config import default_emotion, emotion_descriptions

# Style hints (VoiceDesign / docs). Clone path picks refs by emotion key instead.
EMOTION_INSTRUCT: dict[str, str] = {
    "neutral": (
        "warm living conversational delivery with natural pitch movement — "
        "friendly and present, never flat or monotone"
    ),
    "curious": "curious, lightly questioning, discovery tone",
    "playful": "playful, amused, slight smile in the voice",
    "excited": "excited but not shouting, higher energy",
    "warm": "warm, reflective, soft smile",
    "wonder": "quiet awe and soft wonder, unhurried, never shouty",
    "explanatory": "clear friendly teaching tone, patient and engaging, never lecture-y",
}


def _sync_instruct_from_config() -> None:
    for key, desc in emotion_descriptions().items():
        EMOTION_INSTRUCT.setdefault(key, desc)


_sync_instruct_from_config()


# Short CosyVoice delivery cues (keep tiny — long English prose gets spoken aloud).
# Include a mild pace nudge; "unhurried"/teaching cues make CosyVoice drag.
COSYVOICE_STYLE: dict[str, str] = {
    "neutral": "warm natural conversational tone, speak a bit faster",
    "curious": "curious questioning tone, speak a bit faster",
    "playful": "playful amused tone, lively pace",
    "excited": "excited but not shouting, lively pace",
    "warm": "warm friendly tone, conversational pace",
    "wonder": "soft wonder, still conversational — not slow",
    "explanatory": "clear friendly tone, conversational pace",
}

# CosyVoice inference_instruct2 speed (>1 = faster). Tuned for podcast English.
COSYVOICE_SPEED: dict[str, float] = {
    "wonder": 1.08,
    "warm": 1.10,
    "explanatory": 1.10,
    "curious": 1.10,
    "neutral": 1.10,
    "playful": 1.12,
    "excited": 1.15,
}


def cosyvoice_speed(emotion: str | None) -> float:
    emo = normalize_line_emotion(emotion)
    return float(COSYVOICE_SPEED.get(emo, 1.10))


def build_instruct(base_voice: str, emotion: str) -> str:
    style = EMOTION_INSTRUCT.get(emotion, EMOTION_INSTRUCT.get("neutral", ""))
    return f"{base_voice.rstrip()}, {style}"


def build_cosyvoice_instruct(
    emotion: str | None,
    speaker: str | None = None,
    emphasize: list[str] | None = None,
) -> str:
    """CosyVoice3 inference_instruct2 prompt (must end with <|endofprompt|>).

    Keep this SHORT. Long VoiceDesign-style prose gets read aloud by CosyVoice.
    Speaker identity comes from the reference wav, not from instruct text.
    """
    del speaker  # identity is the ref audio
    emo = normalize_line_emotion(emotion)
    style = COSYVOICE_STYLE.get(emo, COSYVOICE_STYLE["warm"])
    parts = [f"Speak in English with a {style}."]
    words = [w.strip() for w in (emphasize or []) if w and str(w).strip()]
    if words:
        quoted = ", ".join(f'"{w}"' for w in words[:2])
        parts.append(f"Emphasize {quoted}.")
    return "You are a helpful assistant. " + " ".join(parts) + "<|endofprompt|>"


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
    t = t.replace("...", ",")  # or " — "
    t = t.replace("…", ",")
    # excitement/question → sentence end (prosody from emotion refs / wording)
    t = t.replace("!", ".")
    t = t.replace("?", ".")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_line_emotion(emotion: str | None) -> str:
    """Map missing/unknown → config default. Keep 'neutral' as identity."""
    from pipeline.emotion_config import allowed_emotions

    emo = (emotion or "").strip().lower() or default_emotion()
    if emo not in allowed_emotions():
        return default_emotion()
    return emo
