"""Load locked host personalities from config/hosts.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path("config/hosts.yaml")


@lru_cache(maxsize=1)
def load_hosts_config(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _speaker_block(speaker: str) -> dict[str, Any]:
    cfg = load_hosts_config()
    key = "male" if speaker in ("M", "male") else "female" if speaker in ("F", "female") else None
    if key is None:
        raise KeyError(f"Unknown speaker {speaker!r}")
    block = cfg.get(key) or {}
    if not isinstance(block, dict):
        raise ValueError(f"Invalid hosts.yaml entry for {key}")
    return block


def host_one_line(speaker: str) -> str:
    return str(_speaker_block(speaker).get("one_line") or "").strip()


def host_personality(speaker: str) -> str:
    return str(_speaker_block(speaker).get("personality") or "").strip()


def host_role(speaker: str) -> str:
    return str(_speaker_block(speaker).get("role") or "").strip()


def host_qwen_instruct(speaker: str) -> str:
    """Acoustic NL brief for Qwen3-TTS VoiceDesign (and CosyVoice base)."""
    return str(_speaker_block(speaker).get("qwen_instruct") or "").strip()


def host_voice_brief(speaker: str) -> str:
    """Alias: VoiceDesign-aligned acoustic description."""
    return host_qwen_instruct(speaker)


def host_ref_sample(speaker: str) -> str:
    return str(_speaker_block(speaker).get("ref_sample") or "").strip()


def relationship_brief() -> str:
    return str(load_hosts_config().get("relationship") or "").strip()


def hosts_brief_for_prompt() -> str:
    """Compact cast sheet for Writer / Prosody system prompts."""
    m = _speaker_block("M")
    f = _speaker_block("F")
    rel = relationship_brief()
    return f"""HOST CAST (locked personalities — write in their voices):

M — guide / context
- {str(m.get('one_line') or '').strip()}
- Role: {str(m.get('personality') or '').strip()}
- Energy ceiling: curious teacher; animated on reveals; never YouTube-influencer hype.

F — curiosity / reaction
- {str(f.get('one_line') or '').strip()}
- Role: {str(f.get('personality') or '').strip()}
- Energy ceiling: genuine surprise/interest; occasionally playful; never bubbly presenter.

RELATIONSHIP:
{rel}

DIALOGUE PATTERN (prefer):
- M gives context / framing → F reacts or asks → M deepens (or reverse when natural).
- Do NOT make both hosts sound like the same friendly narrator.
- Contrast = personality + function, NOT "male serious / female bubbly."
"""
