"""Travel-audio mix: observation holds, crossfades, subtle ambience beds."""
from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

AMBIENCE_DIR = Path("assets/ambience")
CONFIG_PATH = Path("config/mix.yaml")


@lru_cache(maxsize=1)
def load_mix_config(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return {
            "observation_hold_ms": 2200,
            "crossfade_ms": 60,
            "bed_gain_db": -34,
            "bed_fade_ms": 1200,
            "beds": {},
            "hold_patterns": [],
        }
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.setdefault("observation_hold_ms", 2200)
    data.setdefault("crossfade_ms", 60)
    data.setdefault("bed_gain_db", -34)
    data.setdefault("bed_fade_ms", 1200)
    data.setdefault("beds", {})
    data.setdefault("hold_patterns", [])
    return data


def text_wants_hold(text: str, patterns: list[str] | None = None) -> bool:
    cfg_patterns = patterns if patterns is not None else load_mix_config().get("hold_patterns") or []
    t = text.lower()
    for pat in cfg_patterns:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False


def observation_hold_ms(text: str, explicit: int | None = None) -> int:
    if explicit is not None and explicit > 0:
        return int(explicit)
    if text_wants_hold(text):
        return int(load_mix_config()["observation_hold_ms"])
    return 0


def ensure_placeholder_beds(ambience_dir: str | Path = AMBIENCE_DIR) -> Path:
    """Create soft loopable placeholder beds via ffmpeg (no external SFX pack)."""
    ambience_dir = Path(ambience_dir)
    ambience_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "fair_soft.wav": "anoisesrc=d=20:c=pink:r=44100,highpass=f=200,lowpass=f=4000,volume=0.15",
        "performance_soft.wav": "anoisesrc=d=20:c=brown:r=44100,bandpass=f=180:width_type=h:width=120,volume=0.12",
        "market_soft.wav": "anoisesrc=d=20:c=pink:r=44100,highpass=f=400,lowpass=f=6000,volume=0.14",
        "dining_soft.wav": "anoisesrc=d=20:c=brown:r=44100,lowpass=f=1200,volume=0.11",
    }
    for name, filtr in specs.items():
        path = ambience_dir / name
        if path.exists() and path.stat().st_size > 1000:
            continue
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            filtr,
            "-t",
            "20",
            str(path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    return ambience_dir


def _loop_to_length(bed: AudioSegment, length_ms: int) -> AudioSegment:
    if length_ms <= 0:
        return AudioSegment.silent(duration=0)
    if len(bed) <= 0:
        return AudioSegment.silent(duration=length_ms)
    out = AudioSegment.empty()
    while len(out) < length_ms:
        out += bed
    return out[:length_ms]


def load_bed(section: str, ambience_dir: Path | None = None) -> AudioSegment:
    cfg = load_mix_config()
    beds = cfg.get("beds") or {}
    name = beds.get(section) or beds.get("default") or "fair_soft.wav"
    root = Path(ambience_dir) if ambience_dir else AMBIENCE_DIR
    ensure_placeholder_beds(root)
    path = root / name
    if not path.exists():
        return AudioSegment.silent(duration=1000)
    bed = AudioSegment.from_file(path)
    gain = float(cfg["bed_gain_db"])
    return bed + gain


def overlay_bed(dialogue: AudioSegment, section: str = "default") -> AudioSegment:
    """Full-length subtle bed under a finished dialogue wav (MOSS episode)."""
    if len(dialogue) == 0:
        return dialogue
    cfg = load_mix_config()
    bed = _loop_to_length(load_bed(section), len(dialogue))
    fade = int(cfg["bed_fade_ms"])
    if fade > 0 and len(bed) > 2 * fade:
        bed = bed.fade_in(fade).fade_out(fade)
    return dialogue.overlay(bed)


def overlay_section_beds(
    dialogue: AudioSegment,
    segments: list[tuple[int, int, str]],
) -> AudioSegment:
    """segments: list of (start_ms, end_ms, section_name)."""
    if not segments:
        return overlay_bed(dialogue, "default")
    mixed = dialogue
    cfg = load_mix_config()
    fade = int(cfg["bed_fade_ms"])
    for start, end, section in segments:
        start = max(0, start)
        end = min(len(dialogue), end)
        if end <= start:
            continue
        bed = _loop_to_length(load_bed(section), end - start)
        if fade > 0 and len(bed) > 2 * fade:
            bed = bed.fade_in(min(fade, len(bed) // 3)).fade_out(min(fade, len(bed) // 3))
        mixed = mixed.overlay(bed, position=start)
    return mixed


def trim_silence(
    audio: AudioSegment,
    silence_thresh: int = -40,
    min_silence_len: int = 250,
    padding_ms: int = 50,
) -> AudioSegment:
    if len(audio) == 0:
        return audio
    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )
    if not nonsilent:
        return audio
    start = max(nonsilent[0][0] - padding_ms, 0)
    end = min(nonsilent[-1][1] + padding_ms, len(audio))
    return audio[start:end]


def append_with_crossfade(
    episode: AudioSegment,
    clip: AudioSegment,
    crossfade_ms: int | None = None,
) -> AudioSegment:
    cfg = load_mix_config()
    xf = int(cfg["crossfade_ms"] if crossfade_ms is None else crossfade_ms)
    if len(episode) == 0:
        return clip
    if xf <= 0 or len(episode) < xf or len(clip) < xf:
        return episode + clip
    return episode.append(clip, crossfade=xf)


def insert_holds_by_word_share(
    episode: AudioSegment,
    texts: list[str],
    hold_after_ms: list[int],
) -> AudioSegment:
    """Approx line boundaries by word share of nonsilent audio; insert holds.

    Used for MOSS one-pass wav when we have the directed script texts + holds.
    """
    if not texts or len(texts) != len(hold_after_ms):
        return episode
    words = [max(1, len(re.findall(r"[A-Za-z0-9']+", t))) for t in texts]
    total_w = sum(words) or 1
    # Use nonsilent span as 'speech' region
    ranges = detect_nonsilent(episode, min_silence_len=200, silence_thresh=-42)
    if not ranges:
        return episode
    speech_start, speech_end = ranges[0][0], ranges[-1][1]
    speech = episode[speech_start:speech_end]
    head = episode[:speech_start]
    tail = episode[speech_end:]

    out = head
    cursor = 0
    for i, (w, hold) in enumerate(zip(words, hold_after_ms)):
        seg_len = int(round(len(speech) * (w / total_w)))
        if i == len(words) - 1:
            piece = speech[cursor:]
        else:
            piece = speech[cursor : cursor + seg_len]
            cursor += seg_len
        out += piece
        if hold and hold > 0:
            out += AudioSegment.silent(duration=int(hold))
    out += tail
    return out
