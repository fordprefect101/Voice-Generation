"""Shared voice-ref paths and identity transcripts.

Identity refs: voices/refs/male_ref.wav, female_ref.wav
  → living warm default voice that Fish clones for every line.
"""
from __future__ import annotations

import json
from pathlib import Path

REF_DIR = Path("voices/refs")


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
    """Update identity transcripts; keep any other keys already in the file."""
    data = load_ref_texts(path)
    data["male"] = male
    data["female"] = female
    save_ref_texts(data, path)
    return data
