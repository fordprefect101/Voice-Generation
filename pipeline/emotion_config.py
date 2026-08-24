"""Load extensible emotion bank from config/emotions.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path("config/emotions.yaml")


@lru_cache(maxsize=1)
def load_emotion_config(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return {
            "default_emotion": "warm",
            "identity_emotion": "neutral",
            "bank": {},
        }
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.setdefault("default_emotion", "warm")
    data.setdefault("identity_emotion", "neutral")
    data.setdefault("bank", {})
    return data


def default_emotion() -> str:
    return str(load_emotion_config()["default_emotion"])


def identity_emotion() -> str:
    return str(load_emotion_config()["identity_emotion"])


def emotion_bank_keys() -> tuple[str, ...]:
    bank = load_emotion_config().get("bank") or {}
    return tuple(bank.keys())


def allowed_emotions() -> frozenset[str]:
    """Identity key + all bank keys."""
    cfg = load_emotion_config()
    keys = {str(cfg["identity_emotion"]), str(cfg["default_emotion"])}
    keys.update(str(k) for k in (cfg.get("bank") or {}))
    return frozenset(keys)


def emotion_sample_texts() -> dict[str, str]:
    bank = load_emotion_config().get("bank") or {}
    out: dict[str, str] = {}
    for key, meta in bank.items():
        if isinstance(meta, dict) and meta.get("sample"):
            out[str(key)] = str(meta["sample"]).strip()
    return out


def emotion_descriptions() -> dict[str, str]:
    bank = load_emotion_config().get("bank") or {}
    out: dict[str, str] = {}
    for key, meta in bank.items():
        if isinstance(meta, dict) and meta.get("description"):
            out[str(key)] = str(meta["description"]).strip()
    return out


def emotion_list_for_prompt() -> str:
    """Human-readable list for Writer/Prosody system prompts."""
    cfg = load_emotion_config()
    ident = cfg["identity_emotion"]
    lines = [
        f"- {ident}: living default identity voice (warm, present, human — NOT flat)"
    ]
    for key, desc in emotion_descriptions().items():
        lines.append(f"- {key}: {desc}")
    return "\n".join(lines)
