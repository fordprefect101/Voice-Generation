from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Speaker = Literal["M", "F"]


@dataclass
class ParsedTurn:
    index: int
    speaker: Speaker
    text: str
