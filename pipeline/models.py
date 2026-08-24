from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Speaker = Literal["M", "F"]
# Emotion is an extensible string key (config/emotions.yaml + identity "neutral").
Emotion = str


@dataclass
class ParsedTurn:
    index: int
    speaker: Speaker
    text: str


@dataclass
class EnrichedLine:
    index: int
    speaker: Speaker
    text: str
    emotion: Emotion
    pause_before_ms: int
    pause_after_ms: int
    # Exact substrings of text for CosyVoice word stress (0–2 typical).
    emphasize: list[str] = field(default_factory=list)


def parsed_turn_to_dict(turn: ParsedTurn) -> dict:
    return asdict(turn)


def parsed_turn_from_dict(data: dict) -> ParsedTurn:
    return ParsedTurn(
        index=int(data["index"]),
        speaker=data["speaker"],
        text=str(data["text"]),
    )


def enriched_line_to_dict(line: EnrichedLine) -> dict:
    return asdict(line)


def normalize_emphasize(raw: object, text: str) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = [str(x) for x in raw]
    else:
        return []
    out: list[str] = []
    lower = text.lower()
    for item in items:
        span = item.strip()
        if not span:
            continue
        if span in text:
            resolved = span
        else:
            # Case-insensitive rescue → keep the spelling as it appears in text.
            idx = lower.find(span.lower())
            if idx < 0:
                continue
            resolved = text[idx : idx + len(span)]
        if resolved not in out:
            out.append(resolved)
        if len(out) >= 2:
            break
    return out


def enriched_line_from_dict(data: dict) -> EnrichedLine:
    text = str(data["text"])
    return EnrichedLine(
        index=int(data["index"]),
        speaker=data["speaker"],
        text=text,
        emotion=str(data.get("emotion") or "warm"),
        pause_before_ms=int(data["pause_before_ms"]),
        pause_after_ms=int(data["pause_after_ms"]),
        emphasize=normalize_emphasize(data.get("emphasize"), text),
    )


def load_lines_json(path: str | Path) -> list[EnrichedLine]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array in {path}")
    return [enriched_line_from_dict(item) for item in raw]


def save_lines_json(path: str | Path, lines: list[EnrichedLine]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [enriched_line_to_dict(line) for line in lines]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
