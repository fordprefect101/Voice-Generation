"""MOSS-TTSD travel dialogue pipeline: editor → director → pronunciations → export.

Researched against OpenMOSS/MOSS-TTSD demos + normalize_text behavior.
Does NOT aggressively cut content density (product choice); focuses on
situation-safe phrasing, arc/ending, companion delivery, and speak-forms.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from pipeline.hosts import hosts_brief_for_prompt
from pipeline.mix import observation_hold_ms, text_wants_hold
from pipeline.models import ParsedTurn
from pipeline.parse_script import parse_script_text
from pipeline.pronounce import apply_pronunciations_to_turns

DEFAULT_MOSS_DIRECTOR_MODEL = "gpt-4o"

_SPEAKER_TO_NUM = {"M": "1", "F": "2"}
_SECTIONS = ("arrival", "culture", "performance", "craft", "food", "close")


def _client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not set. Add it to .env or export it in your shell."
        )
    return OpenAI()


def _chat_json(
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


def _travel_editor_system() -> str:
    cast = hosts_brief_for_prompt()
    return f"""You are a TRAVEL AUDIO GUIDE EDITOR for on-site listening.
You rewrite two-host dialogue so it feels like companions on site — not a studio
promo, not a voice-assistant checklist.

{cast}

## FIX THESE (do not invent new attractions)
1) SITUATION-SAFE PHRASING — the listener may be elsewhere in the venue, daytime,
   or when no show is on. NEVER assert unseen scenes:
   Ban / rewrite: "Look over there.", "Listen to that.", "Those are the dancers.",
   "As the sun sets…" as hard facts.
   Prefer: "If you can see dancers nearby…", "You may hear a dhol later this
   evening…", "Around sunset the lanterns usually come on…", "When a performance
   is on…".
2) FEMALE VOICE (F) — avoid list-reader fragments. F should mostly react, ask,
   and land one feeling per turn in a natural sentence — not five tiny items.
   Vary openings (not always "And…" / "It's…"). Give list/enumeration work to M.
   Prefer coherent F lines with emotional shift inside the turn when useful.
3) EMOTIONAL ARC — keep the same facts but shape progression via section tags:
   arrival → culture → performance → craft → food → close.
   Soften hype early; warm curiosity mid; food section appetizing; close calm.
4) ENDING — ban tourism slogans ("sweep you off your feet", "pure magic",
   "unforgettable journey"). End companion-like with a useful next step or a
   quiet observation (e.g. head toward the thali seating if hungry).
5) OBSERVATION BEATS — occasionally invite the traveller to notice something
   with conditional language, then keep that turn short so a hold can follow.

## DO NOT
- Aggressively cut content just to shorten (keep cultural terms and key facts).
- Add stage directions, ALL CAPS, SSML, or [laugh].
- Output pause_ms / emphasize fields.

## OUTPUT
Return ONLY valid JSON:
{{"lines":[{{"speaker":"M","text":"...","section":"arrival"}}, ...]}}
section must be one of: {", ".join(_SECTIONS)}.
Speakers: only "M" or "F". You MAY split turns when intentions differ.
"""


def _moss_director_system() -> str:
    cast = hosts_brief_for_prompt()
    return f"""You are a MOSS-TTSD PERFORMANCE DIRECTOR for a two-host travel AUDIO GUIDE.
You reshape already-edited travel dialogue for MOSS-TTSD (one-pass multi-speaker TTS).

{cast}

## HOW MOSS WORKS
- Prosody from words + punctuation + cross-turn context. No emphasize[] instruct.
- Prefer "." "?" and short punch lines. Ellipsis/em-dash often become commas.
- Echo-reactions: "Living recreation." / "You can feel it, can't you?"
- Keep section meaning; preserve situation-safe conditionals from the editor.
- F: coherent reactions, not staccato checklist fragments. Vary rhythm; lists to M.
- M: guide/context; can carry lists as short period-separated beats.
- Preserve section field on each line when present.
- Ban slogan endings and brochure stacked hype.
- Keep cultural names (will be pronunciation-normalized later).

## OUTPUT
Return ONLY valid JSON:
{{"lines":[{{"speaker":"M","text":"...","section":"arrival"}}, ...]}}
"""


def _light_normalize_for_moss(text: str) -> str:
    t = text.strip()
    t = t.replace("…", ". ")
    t = t.replace("...", ". ")
    t = t.replace("—", ". ")
    t = t.replace("–", ". ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = re.sub(r"([.!?]){2,}", r"\1", t)
    t = re.sub(r"\.\s*\.", ".", t)
    t = re.sub(
        r"([.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        t,
    )
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t.strip()


def _parse_agent_lines(raw_lines: Any) -> list[dict[str, str]]:
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError("Agent returned no lines")
    out: list[dict[str, str]] = []
    for raw in raw_lines:
        sp = str(raw.get("speaker") or "").strip().upper()
        if sp in ("1", "SPEAKER 1", "S1"):
            sp = "M"
        elif sp in ("2", "SPEAKER 2", "S2"):
            sp = "F"
        if sp not in ("M", "F"):
            raise ValueError(f"Invalid speaker: {raw!r}")
        text = _light_normalize_for_moss(str(raw.get("text") or ""))
        if not text:
            continue
        section = str(raw.get("section") or "culture").strip().lower()
        if section not in _SECTIONS:
            section = "culture"
        out.append({"speaker": sp, "text": text, "section": section})
    if not out:
        raise ValueError("Agent produced only empty lines")
    return out


def run_travel_editor(
    turns: list[ParsedTurn],
    *,
    model: str = DEFAULT_MOSS_DIRECTOR_MODEL,
    client: OpenAI | None = None,
) -> list[dict[str, str]]:
    client = client or _client()
    payload = {
        "source_lines": [
            {"index": t.index, "speaker": t.speaker, "text": t.text} for t in turns
        ]
    }
    data = _chat_json(
        client,
        model,
        _travel_editor_system(),
        "Edit this on-site travel dialogue. Fix deixis, F list-reading, arc, and ending. "
        "Do not slash content density just to shorten.\n\n"
        + json.dumps(payload, ensure_ascii=False),
    )
    return _parse_agent_lines(data.get("lines") if isinstance(data, dict) else data)


def run_moss_director(
    turns: list[ParsedTurn] | list[dict[str, str]],
    *,
    model: str = DEFAULT_MOSS_DIRECTOR_MODEL,
    client: OpenAI | None = None,
) -> tuple[list[ParsedTurn], list[str]]:
    """Director rewrite → MOSS-ready turns + section labels."""
    client = client or _client()
    source = []
    for i, t in enumerate(turns, start=1):
        if isinstance(t, dict):
            source.append(
                {
                    "index": i,
                    "speaker": t["speaker"],
                    "text": t["text"],
                    "section": t.get("section", "culture"),
                }
            )
        else:
            source.append({"index": t.index, "speaker": t.speaker, "text": t.text})
    data = _chat_json(
        client,
        model,
        _moss_director_system(),
        "Reshape for MOSS-TTSD delivery. Keep situation-safe wording and sections.\n\n"
        + json.dumps({"source_lines": source}, ensure_ascii=False),
    )
    parsed = _parse_agent_lines(data.get("lines") if isinstance(data, dict) else data)
    out: list[ParsedTurn] = []
    sections: list[str] = []
    for i, row in enumerate(parsed, start=1):
        out.append(ParsedTurn(index=i, speaker=row["speaker"], text=row["text"]))  # type: ignore[arg-type]
        sections.append(row["section"])
    return out, sections


def build_cues(
    turns: list[ParsedTurn],
    sections: list[str] | None = None,
) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    for i, t in enumerate(turns):
        section = "culture"
        if sections and i < len(sections):
            section = sections[i]
        hold = observation_hold_ms(t.text)
        cues.append(
            {
                "index": t.index,
                "speaker": t.speaker,
                "text": t.text,
                "section": section,
                "hold_after_ms": hold,
                "wants_hold": text_wants_hold(t.text),
            }
        )
    return cues


def turns_to_moss_txt(turns: list[ParsedTurn], *, style: str = "speaker") -> str:
    lines: list[str] = []
    for t in turns:
        if style == "tag":
            lines.append(f"[{t.speaker}] {t.text}")
        else:
            num = _SPEAKER_TO_NUM[t.speaker]
            lines.append(f"Speaker {num}: {t.text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_moss_txt(path: str | Path, turns: list[ParsedTurn], *, style: str = "speaker") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(turns_to_moss_txt(turns, style=style), encoding="utf-8")


def save_moss_cues(path: str | Path, cues: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cues, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_moss_travel_pipeline(
    turns: list[ParsedTurn],
    *,
    model: str = DEFAULT_MOSS_DIRECTOR_MODEL,
    skip_editor: bool = False,
) -> tuple[list[ParsedTurn], list[dict[str, Any]]]:
    """Editor → Director → pronunciations → cues."""
    client = _client()
    if skip_editor:
        directed_source: list[ParsedTurn] | list[dict[str, str]] = turns
        print("Skipping travel editor.")
    else:
        print("Travel editor (situation / arc / ending / F delivery)...")
        directed_source = run_travel_editor(turns, model=model, client=client)
    print("MOSS director...")
    directed, sections = run_moss_director(directed_source, model=model, client=client)
    print("Applying pronunciations (lexicon speak-forms → script text)...")
    if os.getenv("PRONUNCIATION_LLM", "").strip() in ("1", "true", "yes"):
        from pipeline.pronounce import (
            extract_candidate_terms,
            merge_speak_forms_into_yaml,
            resolve_speak_forms,
        )

        terms = extract_candidate_terms([t.text for t in directed])
        # Also catch unknown Title-Case tokens later via LLM when enabled for missing
        forms = resolve_speak_forms(terms, use_llm=True, model=model)
        merge_speak_forms_into_yaml(forms)
        print(f"  pronunciation LLM refreshed {len(forms)} lexicon entries")
    directed = apply_pronunciations_to_turns(directed)
    cues = build_cues(directed, sections)
    return directed, cues


def load_and_direct_script(
    script_text: str,
    *,
    model: str = DEFAULT_MOSS_DIRECTOR_MODEL,
) -> list[ParsedTurn]:
    turns = parse_script_text(script_text)
    if not turns:
        raise ValueError("No dialogue lines found in script")
    directed, _ = run_moss_travel_pipeline(turns, model=model)
    return directed