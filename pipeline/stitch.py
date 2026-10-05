from __future__ import annotations

import subprocess
from pathlib import Path


def normalize_loudness(input_path: str | Path, output_path: str | Path) -> str:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(output_path)
