"""Per-clip speaking-rate analysis + pitch-preserving time stretch.

Flow: generate wav → measure WPM vs emotion target → ffmpeg atempo if off → stitch.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from pipeline.emotions import normalize_line_emotion
from pipeline.models import EnrichedLine

CONFIG_PATH = Path("config/pace.yaml")


def clip_path(clips_dir: Path, line: EnrichedLine) -> Path:
    return Path(clips_dir) / f"{line.index:03d}_{line.speaker}.wav"


@lru_cache(maxsize=1)
def load_pace_config(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return {
            "default_wpm": 150,
            "tolerance": 0.10,
            "min_factor": 0.85,
            "max_factor": 1.15,
            "min_words": 4,
            "min_duration_sec": 0.4,
            "by_emotion": {},
        }
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.setdefault("default_wpm", 150)
    data.setdefault("tolerance", 0.10)
    data.setdefault("min_factor", 0.85)
    data.setdefault("max_factor", 1.15)
    data.setdefault("min_words", 4)
    data.setdefault("min_duration_sec", 0.4)
    data.setdefault("by_emotion", {})
    return data


def word_count(text: str) -> int:
    # Count word-ish tokens; ignore lone ellipsis / dashes.
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", text)
    return len(tokens)


def target_wpm(emotion: str | None) -> float:
    cfg = load_pace_config()
    emo = normalize_line_emotion(emotion)
    by = cfg.get("by_emotion") or {}
    if emo in by:
        return float(by[emo])
    return float(cfg["default_wpm"])


def nonsilent_duration_sec(
    path: str | Path,
    *,
    silence_thresh: int = -40,
    min_silence_len: int = 250,
) -> float:
    audio = AudioSegment.from_wav(path)
    if len(audio) == 0:
        return 0.0
    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )
    if not ranges:
        return len(audio) / 1000.0
    return sum((end - start) for start, end in ranges) / 1000.0


def actual_wpm(text: str, duration_sec: float) -> float | None:
    words = word_count(text)
    if words <= 0 or duration_sec <= 0:
        return None
    return words / (duration_sec / 60.0)


def stretch_factor(actual: float, target: float) -> float | None:
    """Return atempo factor ( >1 = faster ) or None if within tolerance."""
    cfg = load_pace_config()
    if target <= 0 or actual <= 0:
        return None
    ratio = actual / target
    tol = float(cfg["tolerance"])
    if abs(ratio - 1.0) <= tol:
        return None
    # actual slow → ratio < 1 → need faster playback → factor > 1
    factor = target / actual
    lo = float(cfg["min_factor"])
    hi = float(cfg["max_factor"])
    return max(lo, min(hi, factor))


def apply_atempo(path: str | Path, factor: float) -> Path:
    """Pitch-preserving speed change via ffmpeg atempo. Overwrites path."""
    path = Path(path)
    if abs(factor - 1.0) < 1e-3:
        return path
    factor = max(0.5, min(2.0, float(factor)))
    tmp_path = path.with_suffix(".atempo_tmp.wav")
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-filter:a",
            f"atempo={factor:.4f}",
            str(tmp_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return path


@dataclass
class PaceReport:
    index: int
    speaker: str
    emotion: str
    words: int
    duration_sec: float
    actual_wpm: float | None
    target_wpm: float
    factor: float
    adjusted: bool
    path: str
    skipped_reason: str | None = None


def adjust_clip_pace(
    path: str | Path,
    line: EnrichedLine,
    *,
    dry_run: bool = False,
) -> PaceReport:
    path = Path(path)
    cfg = load_pace_config()
    emotion = normalize_line_emotion(line.emotion)
    words = word_count(line.text)
    target = target_wpm(emotion)
    duration = nonsilent_duration_sec(path) if path.exists() else 0.0
    wpm = actual_wpm(line.text, duration)

    base = PaceReport(
        index=line.index,
        speaker=line.speaker,
        emotion=emotion,
        words=words,
        duration_sec=round(duration, 3),
        actual_wpm=round(wpm, 1) if wpm is not None else None,
        target_wpm=target,
        factor=1.0,
        adjusted=False,
        path=str(path),
    )

    if words < int(cfg["min_words"]):
        base.skipped_reason = "too_few_words"
        return base
    if duration < float(cfg["min_duration_sec"]):
        base.skipped_reason = "too_short"
        return base
    if wpm is None:
        base.skipped_reason = "no_wpm"
        return base

    factor = stretch_factor(wpm, target)
    if factor is None:
        base.skipped_reason = "within_tolerance"
        return base

    base.factor = round(factor, 4)
    if dry_run:
        base.adjusted = False
        base.skipped_reason = "dry_run"
        return base

    apply_atempo(path, factor)
    # Re-measure after stretch for the report.
    new_dur = nonsilent_duration_sec(path)
    new_wpm = actual_wpm(line.text, new_dur)
    base.duration_sec = round(new_dur, 3)
    base.actual_wpm = round(new_wpm, 1) if new_wpm is not None else None
    base.adjusted = True
    base.skipped_reason = None
    return base


def adjust_all_clips(
    lines: list[EnrichedLine],
    clips_dir: str | Path,
    *,
    dry_run: bool = False,
    report_path: str | Path | None = None,
) -> list[PaceReport]:
    clips_dir = Path(clips_dir)
    reports: list[PaceReport] = []
    for line in lines:
        path = clip_path(clips_dir, line)
        if not path.exists():
            reports.append(
                PaceReport(
                    index=line.index,
                    speaker=line.speaker,
                    emotion=normalize_line_emotion(line.emotion),
                    words=word_count(line.text),
                    duration_sec=0.0,
                    actual_wpm=None,
                    target_wpm=target_wpm(line.emotion),
                    factor=1.0,
                    adjusted=False,
                    path=str(path),
                    skipped_reason="missing_clip",
                )
            )
            continue
        report = adjust_clip_pace(path, line, dry_run=dry_run)
        reports.append(report)
        tag = "ADJUST" if report.adjusted else ("WOULD" if dry_run and report.factor != 1.0 and report.skipped_reason == "dry_run" else "keep")
        print(
            f"  pace [{line.index}] {tag} "
            f"wpm={report.actual_wpm}->{report.target_wpm} "
            f"x{report.factor}"
            + (f" ({report.skipped_reason})" if report.skipped_reason else "")
        )

    out = Path(report_path) if report_path else clips_dir / "pace_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([asdict(r) for r in reports], indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Pace report -> {out}")
    return reports


if __name__ == "__main__":
    import argparse

    from pipeline.models import load_lines_json

    parser = argparse.ArgumentParser(description="Adjust clip speaking rates")
    parser.add_argument("--lines", default="output/lines.json")
    parser.add_argument("--clips", default="output/clips")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    adjust_all_clips(load_lines_json(args.lines), args.clips, dry_run=args.dry_run)
