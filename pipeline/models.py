from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Speaker = Literal["M", "F"]
Emotion = Literal["neutral", "curious", "playful", "excited", "warm"]

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

def enriched_line_from_dict(data: dict) -> EnrichedLine:
    return EnrichedLine(
        index=int(data["index"]),
        speaker=data["speaker"],
        text=str(data["text"]),
        emotion=data["emotion"],
        pause_before_ms=int(data["pause_before_ms"]),
        pause_after_ms=int(data["pause_after_ms"]),
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