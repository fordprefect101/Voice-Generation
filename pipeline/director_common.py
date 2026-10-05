"""Shared helpers for the Fish director: OpenAI client, JSON chat, cues, txt export."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from pipeline.mix import observation_hold_ms, text_wants_hold
from pipeline.models import ParsedTurn

DEFAULT_DIRECTOR_MODEL = "gpt-4o"

_SPEAKER_TO_NUM = {"M": "1", "F": "2"}
SECTIONS = ("arrival", "culture", "performance", "craft", "food", "close")


def client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not set. Add it to .env or export it in your shell."
        )
    return OpenAI()


def chat_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.7,
) -> Any:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    return json.loads(content)


def build_cues(
    turns: list[ParsedTurn],
    sections: list[str] | None = None,
) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    for i, t in enumerate(turns):
        section = "culture"
        if sections and i < len(sections):
            section = sections[i]
        cues.append(
            {
                "index": t.index,
                "speaker": t.speaker,
                "text": t.text,
                "section": section,
                "hold_after_ms": observation_hold_ms(t.text),
                "wants_hold": text_wants_hold(t.text),
            }
        )
    return cues


def turns_to_txt(turns: list[ParsedTurn], *, style: str = "speaker") -> str:
    lines: list[str] = []
    for t in turns:
        if style == "tag":
            lines.append(f"[{t.speaker}] {t.text}")
        else:
            lines.append(f"Speaker {_SPEAKER_TO_NUM[t.speaker]}: {t.text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_txt(path: str | Path, turns: list[ParsedTurn], *, style: str = "speaker") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(turns_to_txt(turns, style=style), encoding="utf-8")


def save_cues(path: str | Path, cues: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cues, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
