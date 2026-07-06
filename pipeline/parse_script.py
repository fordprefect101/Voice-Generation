from __future__ import annotations

import re
from pathlib import Path

from pipeline.models import ParsedTurn

_SPEAKER_LINE = re.compile(
    r"^\s*Speaker\s+(?P<num>1|2)\s*:\s*(?P<text>.+?)\s*$",
    re.IGNORECASE,
)
_TAG_LINE = re.compile(r"^\s*\[(?P<speaker>M|F)\]\s*(?P<text>.+?)\s*$", re.IGNORECASE)

_SPEAKER_NUM_TO_ID = {"1": "M", "2": "F"}

def _detect_speaker(line: str) -> tuple[str, str] | None:
  """Return (speaker_id, dialogue_text) or None if line is not a dialogue line."""
  match = _SPEAKER_LINE.match(line)
  if match:
    speaker = _SPEAKER_NUM_TO_ID[match.group("num")]
    return speaker, match.group("text").strip()

  match = _TAG_LINE.match(line)
  if match:
    return match.group("speaker").upper(), match.group("text").strip()

  return None

def parse_script_text(text: str) -> list[ParsedTurn]:
  turns: list[ParsedTurn] = []
  index = 1

  for line in text.splitlines():
    if not line.strip():
      continue

    detected = _detect_speaker(line)
    if detected is None:
      continue

    speaker, dialogue = detected
    turns.append(ParsedTurn(index=index, speaker=speaker, text=dialogue))
    index += 1

  return turns

def parse_script_file(path: str | Path) -> list[ParsedTurn]:
  path = Path(path)
  return parse_script_text(path.read_text(encoding="utf-8"))