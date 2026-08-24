"""Shared voice-ref paths and emotion-bank helpers.

Identity (neutral): voices/refs/male_ref.wav, female_ref.wav
  → living warm default voice, not emotionless.
Emotion banks:     voices/refs/male_curious.wav, female_warm.wav, ...
  → optional tilts; keys come from config/emotions.yaml.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.emotion_config import emotion_bank_keys, emotion_sample_texts

REF_DIR = Path("voices/refs")

# Bank keys from config (extensible). Identity "neutral" is separate files.
EMOTION_BANK = emotion_bank_keys()
EMOTION_SAMPLE_TEXTS = emotion_sample_texts()

SPEAKER_KEYS = {"M": "male", "F": "female", "male": "male", "female": "female"}


def _default_ref_texts() -> dict[str, str]:
    """Identity sample lines for VoiceDesign — prefer hosts.yaml samples."""
    try:
        from pipeline.hosts import host_ref_sample

        male = host_ref_sample("M")
        female = host_ref_sample("F")
        if male and female:
            return {"male": male, "female": female}
    except Exception:
        pass
    return {
        "male": (
            "You know… this place has a real pull to it. I've been around a few spots like this, "
            "but here you can feel why people stay. Let me show you something interesting — "
            "take it at an easy pace with me."
        ),
        "female": (
            "Wait… that's actually fascinating. So people weren't only coming here for one reason? "
            "I love how that changes how you look at the place. Okay — tell me more about that."
        ),
    }


# Identity sample (VoiceDesign) — host personality resting voice.
NEUTRAL_REF_TEXTS: dict[str, str] = _default_ref_texts()


def speaker_key(speaker: str) -> str:
    key = SPEAKER_KEYS.get(speaker)
    if key is None:
        raise KeyError(f"Unknown speaker {speaker!r}")
    return key


def identity_path(speaker: str, ref_dir: Path | None = None) -> Path:
    root = ref_dir or REF_DIR
    return root / f"{speaker_key(speaker)}_ref.wav"


def emotion_path(speaker: str, emotion: str, ref_dir: Path | None = None) -> Path:
    root = ref_dir or REF_DIR
    return root / f"{speaker_key(speaker)}_{emotion}.wav"


def load_ref_texts(path: Path | None = None) -> dict:
    p = path or (REF_DIR / "ref_texts.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_ref_texts(data: dict, path: Path | None = None) -> Path:
    p = path or (REF_DIR / "ref_texts.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def merge_identity_texts(male: str, female: str, path: Path | None = None) -> dict:
    """Update identity transcripts; keep any existing emotion bank texts."""
    data = load_ref_texts(path)
    data["male"] = male
    data["female"] = female
    data.setdefault("emotions", {})
    save_ref_texts(data, path)
    return data


def set_emotion_text(speaker: str, emotion: str, text: str, path: Path | None = None) -> dict:
    data = load_ref_texts(path)
    data.setdefault("emotions", {})
    sk = speaker_key(speaker)
    data["emotions"].setdefault(sk, {})
    data["emotions"][sk][emotion] = text
    save_ref_texts(data, path)
    return data


def resolve_ref(
    speaker: str,
    emotion: str | None = None,
    *,
    ref_dir: Path | None = None,
    ref_texts: dict | None = None,
) -> tuple[Path, str]:
    """Pick wav + transcript for (speaker, emotion). Falls back to identity."""
    root = ref_dir or REF_DIR
    texts = ref_texts if ref_texts is not None else load_ref_texts(root / "ref_texts.json")
    sk = speaker_key(speaker)
    emo = (emotion or "neutral").lower()

    identity = root / f"{sk}_ref.wav"
    identity_text = texts.get(sk) or texts.get("male" if sk == "male" else "female", "")

    if emo != "neutral":
        cand = root / f"{sk}_{emo}.wav"
        if cand.exists():
            bank = (texts.get("emotions") or {}).get(sk) or {}
            samples = emotion_sample_texts()
            text = bank.get(emo) or samples.get(emo, identity_text)
            return cand, text

    return identity, identity_text
