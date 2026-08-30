"""Fish dialogue timeline: turn gaps + utterance grouping + clip assembly.

Fish speaker tokens / [tags] do not create inter-turn timing. Gaps live here:
  cues.pause_after_ms → silence after each generated utterance clip.
Same-speaker lines are merged into one TTS call so prosody stays continuous
inside a beat; silence is only inserted between utterances.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from pipeline.mix import (
    append_with_crossfade,
    ensure_placeholder_beds,
    load_mix_config,
    observation_hold_ms,
    overlay_section_beds,
    trim_silence,
)
from pipeline.models import ParsedTurn

# Soft caps so same-speaker merges stay one conversational beat, not a monologue.
_MAX_UTTERANCE_WORDS = 55
_MAX_UTTERANCE_TURNS = 3


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def conversational_pause_ms(
    text: str,
    *,
    next_speaker: str | None,
    speaker: str,
    index: int,
) -> int:
    """Varied turn gap — not a flat +700ms stamp."""
    stripped = text.rstrip()
    speaker_change = next_speaker is not None and next_speaker != speaker

    if stripped.endswith("?"):
        base = 520 if speaker_change else 380
    elif stripped.endswith("!"):
        base = 280 if speaker_change else 200
    elif "…" in text or "..." in text or "—" in text:
        base = 420 if speaker_change else 300
    elif speaker_change:
        base = 360 + (index % 4) * 40  # ~360–480
    else:
        base = 160 + (index % 3) * 40  # same-speaker bridge if not merged

    hold = observation_hold_ms(text)
    return clamp(max(base, hold), 120, 2500)


def assign_pause_after_ms(
    turns: list[ParsedTurn],
) -> list[int]:
    pauses: list[int] = []
    for i, t in enumerate(turns):
        nxt = turns[i + 1].speaker if i + 1 < len(turns) else None
        pauses.append(
            conversational_pause_ms(
                t.text,
                next_speaker=nxt,
                speaker=t.speaker,
                index=i,
            )
        )
    if pauses:
        pauses[-1] = max(pauses[-1], 600)  # breathe before end
    return pauses


def enrich_cues_with_pauses(cues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach pause_after_ms to existing cue dicts (mutates copies)."""
    turns = [
        ParsedTurn(
            index=int(c.get("index") or i + 1),
            speaker=str(c["speaker"]),
            text=str(c.get("text") or ""),
        )
        for i, c in enumerate(cues)
    ]
    pauses = assign_pause_after_ms(turns)
    out: list[dict[str, Any]] = []
    for c, pa in zip(cues, pauses):
        row = dict(c)
        # Observation hold already folded into conversational_pause_ms via max().
        hold = int(row.get("hold_after_ms") or 0)
        row["pause_after_ms"] = max(pa, hold)
        out.append(row)
    return out


def group_utterances(
    cues: list[dict[str, Any]],
    *,
    max_words: int = _MAX_UTTERANCE_WORDS,
    max_turns: int = _MAX_UTTERANCE_TURNS,
) -> list[dict[str, Any]]:
    """Merge consecutive same-speaker cues into TTS utterances.

    Each utterance:
      speaker, texts[], tagged_texts[], pause_after_ms, section, cue_indexes[],
      fish_text (<|speaker:N|> joined lines for one Fish call)
    """
    if not cues:
        return []

    def fish_sid(speaker: str) -> str:
        sp = speaker.strip().upper()
        if sp in ("1", "M", "SPEAKER 1"):
            return "0"
        return "1"

    groups: list[dict[str, Any]] = []
    buf: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        speaker = str(buf[0]["speaker"])
        tagged = [str(c.get("tagged_text") or c.get("text") or "") for c in buf]
        spoken = [str(c.get("text") or "") for c in buf]
        sid = fish_sid(speaker)
        fish_text = "\n".join(f"<|speaker:{sid}|> {t}" for t in tagged if t.strip())
        pause = int(buf[-1].get("pause_after_ms") or 350)
        # Prefer the strongest observation-style pause inside the group.
        for c in buf:
            pause = max(pause, int(c.get("pause_after_ms") or 0))
            pause = max(pause, int(c.get("hold_after_ms") or 0))
        groups.append(
            {
                "utterance_index": len(groups) + 1,
                "speaker": speaker,
                "section": str(buf[0].get("section") or "culture"),
                "cue_indexes": [int(c.get("index") or 0) for c in buf],
                "texts": spoken,
                "tagged_texts": tagged,
                "fish_text": fish_text,
                "pause_after_ms": pause,
                "words": sum(len(t.split()) for t in spoken),
            }
        )
        buf = []

    for c in cues:
        if not buf:
            buf = [c]
            continue
        same = str(c["speaker"]) == str(buf[0]["speaker"])
        words = sum(len(str(x.get("text") or "").split()) for x in buf) + len(
            str(c.get("text") or "").split()
        )
        if same and len(buf) < max_turns and words <= max_words:
            buf.append(c)
        else:
            flush()
            buf = [c]
    flush()
    return groups


def utterance_clip_name(utterance_index: int, speaker: str) -> str:
    sp = "M" if str(speaker).upper() in ("M", "1", "SPEAKER 1") else "F"
    return f"utt_{utterance_index:03d}_{sp}.wav"


def assemble_utterance_clips(
    utterances: list[dict[str, Any]],
    clips_dir: str | Path,
    *,
    with_beds: bool = True,
    crossfade_ms: int | None = None,
) -> AudioSegment:
    """Concat utterance wavs with pause_after_ms silence (deterministic gaps)."""
    clips_dir = Path(clips_dir)
    ensure_placeholder_beds()
    cfg = load_mix_config()
    xf = int(cfg["crossfade_ms"] if crossfade_ms is None else crossfade_ms)
    episode = AudioSegment.empty()
    spans: list[tuple[int, int, str]] = []

    for u in utterances:
        name = utterance_clip_name(int(u["utterance_index"]), str(u["speaker"]))
        path = clips_dir / name
        if not path.exists():
            # fallback: turn_###.wav if generated 1:1 with cues
            alt = clips_dir / f"turn_{int(u['utterance_index']):03d}.wav"
            path = alt if alt.exists() else path
        if not path.exists():
            raise FileNotFoundError(f"Missing utterance clip: {name} (in {clips_dir})")

        clip = trim_silence(AudioSegment.from_wav(path))
        start = len(episode)
        episode = append_with_crossfade(episode, clip, xf)
        gap = clamp(int(u.get("pause_after_ms") or 0), 0, 3000)
        if gap:
            episode += AudioSegment.silent(duration=gap)
        spans.append((start, len(episode), str(u.get("section") or "default")))

    if with_beds and spans:
        episode = overlay_section_beds(episode, spans)
    return episode


def load_cues(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Cues must be a JSON list: {path}")
    return enrich_cues_with_pauses(data)


def save_utterances(path: str | Path, utterances: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(utterances, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cues_from_parsed_turns(
    turns: list[tuple[str, str]],
    *,
    pauses: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Build minimal cues from notebook (sid, text) pairs when cues.json missing."""
    # sid is Fish "0"/"1" (or M/F)
    parsed: list[ParsedTurn] = []
    for i, (sid, text) in enumerate(turns, start=1):
        s = str(sid).strip().upper()
        speaker = "M" if s in ("0", "M", "SPEAKER 1") else "F"
        parsed.append(ParsedTurn(index=i, speaker=speaker, text=text))
    pause_list = pauses if pauses is not None else assign_pause_after_ms(parsed)
    cues: list[dict[str, Any]] = []
    for t, pa in zip(parsed, pause_list):
        hold = observation_hold_ms(t.text)
        spoken = re.sub(r"\[[^\]]+\]", "", t.text)
        spoken = re.sub(r"\s+", " ", spoken).strip()
        cues.append(
            {
                "index": t.index,
                "speaker": t.speaker,
                "text": spoken,
                "tagged_text": t.text,
                "section": "culture",
                "hold_after_ms": hold,
                "pause_after_ms": max(pa, hold),
                "wants_hold": hold > 0,
            }
        )
    return cues
