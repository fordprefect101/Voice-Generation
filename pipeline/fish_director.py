"""Fish S2 travel pipeline: Fish-shaped editor → Fish writer → pronunciations → export.

The script is a score Fish can read: words, where a [tag] sits, who is speaking.
Do not decorate a MOSS export. Do not assume CosyVoice emphasize[] or MOSS
punctuation-only prosody.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from pipeline.hosts import hosts_brief_for_prompt
from pipeline.models import ParsedTurn
from pipeline.moss_director import (
    DEFAULT_MOSS_DIRECTOR_MODEL,
    _SECTIONS,
    _client,
    _chat_json,
    build_cues,
)
from pipeline.pronounce import apply_pronunciations_to_turns

# Canonical inline tags Fish should see. Placement is part of the score.
_CANONICAL = {
    "emphasis": "[emphasis]",
    "curious": "[curious]",
    "chuckle": "[chuckle]",
    "inhale": "[inhale]",
    "low voice": "[low voice]",
    "soft awe": "[soft awe]",
    "excited": "[excited]",
    "short pause": "[short pause]",
}
_ALIASES: dict[str, str | None] = {
    "emphasis": "emphasis",
    "emphasize": "emphasis",
    "stress": "emphasis",
    "curious": "curious",
    "chuckle": "chuckle",
    "chuckling": "chuckle",
    "light laugh": "chuckle",
    "inhale": "inhale",
    "breath": "inhale",
    "low voice": "low voice",
    "soft awe": "soft awe",
    "wonder": "soft awe",
    "excited": "excited",
    "short pause": "short pause",
    "pause": "short pause",
    # Identity is already in the clone — do not emit these.
    "warm": None,
    "explanatory": None,
    "neutral": None,
    "with strong accent": None,
    "strong accent": None,
    # Banned (demo-reel / ceiling breakers).
    "laugh": None,
    "laughing": None,
    "laughing tone": None,
    "shouting": None,
    "screaming": None,
    "whisper": None,
    "whispering": None,
    "super happy": None,
    "angry": None,
}

_DELIVERY = frozenset({"curious", "chuckle", "low voice", "soft awe", "excited"})
_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_LEXICAL_LAUGH = re.compile(
    r"^\s*(?:Hah+|Ha|Haha|Heh)(?:[,.]|\b)\s*",
    re.IGNORECASE,
)

_MAX_EXCITED = 2
_MAX_CHUCKLE = 3
_MAX_INHALE = 2
_MAX_SHORT_PAUSE = 5
_MAX_DELIVERY_PER_TURN = 2
_MAX_EMPHASIS_PER_TURN = 4
_MAX_SHORT_PAUSE_PER_TURN = 2


def strip_fish_tags(text: str) -> str:
    t = _BRACKET.sub("", text)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    return t


def _canon_key(raw: str) -> str | None:
    key = re.sub(r"\s+", " ", raw.strip().lower())
    if key in _ALIASES:
        return _ALIASES[key]
    if key in _CANONICAL:
        return key
    return None


_SLOGAN = re.compile(
    r"(sweep you off your feet|pure magic|unforgettable(?: culinary)? journey|"
    r"let (?:the )?magic of .{0,40}(?:unfold|sweep)|symphony of flavou?rs)",
    re.IGNORECASE,
)


def _strip_slogan_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [p for p in parts if p and not _SLOGAN.search(p)]
    return " ".join(kept).strip()


def _light_clean(text: str) -> str:
    t = text.strip().replace("\n", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    return t.strip()


def _capitalize_spoken_start(text: str) -> str:
    i = 0
    while i < len(text):
        if text[i] == "[":
            close = text.find("]", i)
            if close < 0:
                break
            i = close + 1
            while i < len(text) and text[i].isspace():
                i += 1
            continue
        if text[i].islower():
            return text[:i] + text[i].upper() + text[i + 1 :]
        break
    return text


def _drop_hanging_emphasis(text: str) -> str:
    """[emphasis] must sit immediately before a spoken word."""
    return re.sub(
        r"\[emphasis\]\s*(?=(?:\[[^\[\]]+\]|\s*[.!,?]|\s*$))",
        "",
        text,
    )


def _demote_comma_pauses(text: str) -> str:
    """[short pause] is not a comma. Keep it only as a notice beat."""
    t = re.sub(r"\s*\[short pause\]\s+(not\b)", r" — \1", text, flags=re.IGNORECASE)
    t = re.sub(
        r"\s*\[short pause\]\s+((?:and|or|it's|it is)\b)",
        r", \1",
        t,
        flags=re.IGNORECASE,
    )
    return t


def _repair_episode_tags(
    turns: list[ParsedTurn],
    sections: list[str] | None = None,
) -> tuple[list[ParsedTurn], list[str] | None]:
    """One [chuckle], on 'Why not both' if rides are nearby; questions are not chuckles."""
    why_idx = next(
        (i for i, t in enumerate(turns) if re.search(r"\bwhy not both\b", t.text, re.I)),
        None,
    )
    out: list[ParsedTurn] = []
    for i, t in enumerate(turns):
        text = t.text
        if "[chuckle]" in text.lower():
            if why_idx is not None and i != why_idx:
                text = re.sub(r"\[chuckle\]\s*", "", text, flags=re.I)
                if text.rstrip().endswith("?"):
                    text = f"[curious] {text.lstrip()}"
                text = _normalize_tag_spacing(text)
                text = _capitalize_spoken_start(text)
            elif why_idx is None:
                pass
        if why_idx is not None and i == why_idx and "[chuckle]" not in text.lower():
            spoken = strip_fish_tags(text)
            if len(spoken.split()) <= 8:
                text = f"[chuckle] {re.sub(r'^\[(?:curious|excited)\]\s*', '', text)}"
                text = _normalize_tag_spacing(text)
        out.append(ParsedTurn(index=t.index, speaker=t.speaker, text=text))

    secs = list(sections) if sections is not None else None
    why_idx = next(
        (i for i, t in enumerate(out) if re.search(r"\bwhy not both\b", t.text, re.I)),
        None,
    )
    if why_idx is not None:
        window = " ".join(
            strip_fish_tags(t.text)
            for t in out[max(0, why_idx - 4) : why_idx + 3]
        )
        if not re.search(r"\b(camel|elephant|bullock)\b", window, re.I):
            out = [t for i, t in enumerate(out) if i != why_idx]
            if secs is not None and why_idx < len(secs):
                secs = [s for i, s in enumerate(secs) if i != why_idx]
    return out, secs


def _ensure_ride_beats(
    turns: list[ParsedTurn],
    sections: list[str],
) -> tuple[list[ParsedTurn], list[str]]:
    """Camel/elephant can vanish if they only lived on a dropped 'Why not both' line."""
    blob = _spoken_blob(turns)
    need_camel = not re.search(r"\bcamel", blob, re.I)
    need_elephant = not re.search(r"elephant", blob, re.I)
    if not need_camel and not need_elephant:
        return turns, sections
    extras: list[tuple[ParsedTurn, str]] = []
    if need_camel:
        extras.append(
            (
                ParsedTurn(
                    index=0,
                    speaker="M",
                    text="If you fancy a ride, a camel's slow sway gives you a view of the fairground.",
                ),
                "rides",
            )
        )
    if need_elephant:
        extras.append(
            (
                ParsedTurn(
                    index=0,
                    speaker="F",
                    text="[curious] Or an elephant?",
                ),
                "rides",
            )
        )
    if not re.search(r"\bwhy not both\b", blob, re.I):
        extras.append(
            (
                ParsedTurn(index=0, speaker="M", text="[chuckle] Why not both?"),
                "rides",
            )
        )
    insert_at = next(
        (i for i, t in enumerate(turns) if re.search(r"bullock", t.text, re.I)),
        None,
    )
    if insert_at is None:
        insert_at = next(
            (i for i, t in enumerate(turns) if re.search(r"kathputl", t.text, re.I)),
            len(turns) - 1,
        ) + 1
    new_turns = list(turns)
    new_secs = list(sections) if sections else ["culture"] * len(turns)
    for offset, (turn, sec) in enumerate(extras):
        new_turns.insert(insert_at + offset, turn)
        new_secs.insert(insert_at + offset, sec)
    print(f"  restored {len(extras)} missing ride beat(s)")
    return new_turns, new_secs


def _fix_emphasis_quotes(text: str) -> str:
    """Don't let quotes sit between [emphasis] and the word Fish should stress."""
    t = re.sub(r"\[emphasis\]\s*['\"“”‘’]+\s*", "[emphasis] ", text)
    t = re.sub(
        r"\[emphasis\] ([^'\"“”‘’\s]+?)['\"“”‘’]+([.,!?]*)",
        r"[emphasis] \1\2",
        t,
    )
    return t


def _normalize_tag_spacing(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"\[([^\[\]]+)\](?=\S)", r"[\1] ", t)
    t = re.sub(r"\s+\[", " [", t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    return t.strip()


def _strip_lexical_laugh_if_chuckle(text: str) -> str:
    if "[chuckle]" not in text.lower():
        return text
    pieces: list[str] = []
    last = 0
    stripped_one = False
    for m in _BRACKET.finditer(text):
        seg = text[last : m.start()]
        if not stripped_one and seg.strip():
            seg = _LEXICAL_LAUGH.sub("", seg, count=1)
            stripped_one = True
        pieces.append(seg)
        pieces.append(m.group(0))
        last = m.end()
    tail = text[last:]
    if not stripped_one and tail.strip():
        tail = _LEXICAL_LAUGH.sub("", tail, count=1)
    pieces.append(tail)
    return "".join(pieces)


class _TagBudget:
    def __init__(self) -> None:
        self.excited = _MAX_EXCITED
        self.chuckle = _MAX_CHUCKLE
        self.inhale = _MAX_INHALE
        self.pause = _MAX_SHORT_PAUSE


def sanitize_fish_text(
    text: str,
    *,
    budget: _TagBudget | None = None,
    allow_excited: bool | None = None,
) -> str:
    """Keep allowed tags in place. Do not hoist a single tag to the front."""
    budget = budget or _TagBudget()
    if allow_excited is False:
        budget.excited = 0

    delivery_here = 0
    emphasis_here = 0
    pause_here = 0
    out: list[str] = []
    last = 0

    for m in _BRACKET.finditer(text):
        out.append(text[last : m.start()])
        last = m.end()
        mapped = _canon_key(m.group(1))
        if mapped is None:
            continue
        if mapped == "emphasis":
            if emphasis_here >= _MAX_EMPHASIS_PER_TURN:
                continue
            emphasis_here += 1
            out.append(_CANONICAL[mapped])
        elif mapped == "short pause":
            if pause_here >= _MAX_SHORT_PAUSE_PER_TURN or budget.pause <= 0:
                continue
            pause_here += 1
            budget.pause -= 1
            out.append(_CANONICAL[mapped])
        elif mapped == "inhale":
            if budget.inhale <= 0:
                continue
            budget.inhale -= 1
            out.append(_CANONICAL[mapped])
        elif mapped == "excited":
            if budget.excited <= 0 or delivery_here >= _MAX_DELIVERY_PER_TURN:
                continue
            budget.excited -= 1
            delivery_here += 1
            out.append(_CANONICAL[mapped])
        elif mapped == "chuckle":
            if budget.chuckle <= 0 or delivery_here >= _MAX_DELIVERY_PER_TURN:
                continue
            budget.chuckle -= 1
            delivery_here += 1
            out.append(_CANONICAL[mapped])
        elif mapped in _DELIVERY:
            if delivery_here >= _MAX_DELIVERY_PER_TURN:
                continue
            delivery_here += 1
            out.append(_CANONICAL[mapped])
        else:
            out.append(_CANONICAL[mapped])

    out.append(text[last:])
    spoken_check = strip_fish_tags("".join(out))
    if not spoken_check:
        return ""
    result = _normalize_tag_spacing("".join(out))
    result = _fix_emphasis_quotes(result)
    result = _drop_hanging_emphasis(result)
    result = _demote_comma_pauses(result)
    result = _strip_lexical_laugh_if_chuckle(result)
    result = _strip_slogan_sentences(result)
    result = _normalize_tag_spacing(result)
    result = _capitalize_spoken_start(result)
    return result


def _fish_editor_system() -> str:
    cast = hosts_brief_for_prompt()
    return f"""You are a TRAVEL AUDIO GUIDE EDITOR writing for Fish Audio S2.
You reshape two-host on-site dialogue so a later Fish writer can place inline
[tags]. You do NOT add [tags] yourself. You do NOT write for MOSS or CosyVoice.

{cast}

## THIS IS NOT A MOSS OR COSYVOICE SCRIPT
MOSS only had punctuation. CosyVoice had emphasize[] on the side. Fish needs the
LINE itself to be placeable. Do not write punch fragments ("Living recreation."),
brochure F paragraphs, or "time portal / breathtaking / symphony of flavours".

## YOUR JOB (reshape in place — do NOT densify)
The source_lines ARE the full episode already. Rewrite phrasing and split further
when intentions differ. Do NOT outline, tighten, trim, or "pace down" the script.
Spoken word count must stay at or above the source. Turn count may go UP (further
splits) but must not collapse into a short highlight reel.
Ban: "concise", "tighten", "trim for pacing", merging many beats into one turn,
dropping craft/food/ride detail to reach a short runtime.

## SHAPE (one intention per turn)
Turn length is short. Episode length is NOT.
1) NAMES as their own beat: "That's Chokhi Dhani." not buried in a definition.
2) REAL QUESTIONS as the whole F turn when asking. Ban "isn't it?" / "doesn't it?".
3) SHORT OBSERVATION, then STOP (at least two in the episode) — insert without
   deleting surrounding facts:
   "If you catch a dhol later, take a quiet second."
   "If you can see dancers nearby, stay with it a moment."
4) LISTS → M. F does not read Kalbelia + Ghoomar + ghagras in one turn.
5) One light beat, once: "Why not both?" after camel vs elephant — not after the dancers.
6) Close: replace slogan endings with a useful next step. Do not delete the food
   or craft beats to make room. Example: "If you're hungry, head toward the
   thali seating and take it slow."

## MUST COVER (fail if any are missing)
Keep every source beat below. Situation-safe wording is required; deleting the
fact is not allowed. Preserve roughly the source mass (same order of magnitude
of turns and spoken words as source_lines — typically 24+ turns and hundreds of
words for a full venue script).
- Chokhi Dhani name; Fine Hamlet / Special Village; living recreation, not an
  ancient village
- Folk music; dhol; clay-pot aromas / food cooking nearby
- Around sunset, lanterns and diyas usually come on (conditional, not "as the sun sets")
- Village-fair atmosphere; when a performance is on: Kalbelia (fluid movement),
  Ghoomar in colourful ghagras
- Kathputli: ancient tales, not just for kids, passed down generations
- Camel ride (view), elephant ride, "Why not both?", bullock cart (bumpy),
  participating not only watching
- Mehendi/henna on hands; artisans: potters, weavers, miniature painters;
  living museum; unique souvenir
- Feast as the famous thing; low stools; leaf platter; hospitality event
- Dal, Baati, Churma with what they are (hard wheat rolls, ghee, lentil dal,
  sweet churma)
- Gatte ki sabzi, bajra roti, chutneys; manuhar (insistence to eat more)
- Close: celebration / soul of Rajasthan without slogans; optional next step
  toward thali seating

## SITUATION-SAFE
Ban: "Look over there.", "Listen to that.", "Those are the dancers.",
"As the sun sets…" as hard facts, "look around" as a command.
Prefer: "If you can see dancers nearby…", "You may hear a dhol later…",
"Around sunset the lanterns usually come on…".

## BAD (do not emit this shape)
F: "It's breathtaking! Did you know Chokhi Dhani means Fine Hamlet? It's a living
recreation designed to give a taste of Rajasthan's rich culture."
F: "If you see dancers they might be Kalbelia or Ghoomar swirling in ghagras."
F: "Explore every corner and let the magic sweep you off your feet."
A 10–15 line "highlights" pass that skips artisans, baati detail, or manuhar.

## TURN SHAPE (examples of length — not a complete episode)
M: "Here we are. That's Chokhi Dhani."
F: "Fine Hamlet — that's the name?"
M: "If you catch a dhol later, take a quiet second."
If you output only these beats, you have failed. Continue through Kathputli,
rides, mehendi, artisans, the full feast, and a quiet close.

Keep cultural terms and facts. Do not invent attractions.
Do not add [tags]. Do not output pause_ms / emphasize fields.

## OUTPUT
Return ONLY valid JSON:
{{"lines":[{{"speaker":"M","text":"...","section":"arrival"}}, ...]}}
section must be one of: {", ".join(_SECTIONS)}.
Speakers: only "M" or "F". Prefer one edited line per source index; split further
when intentions differ. Never return a short outline of the source.
"""


def _fish_writer_system() -> str:
    cast = hosts_brief_for_prompt()
    return f"""You are a FISH AUDIO S2 WRITER for a two-host travel AUDIO GUIDE.
You rewrite already-edited companion dialogue into a Fish S2 score: wording AND
inline [tags]. Do not sprinkle tags on MOSS/CosyVoice-shaped paragraphs.

{cast}

## HOW FISH S2 ACTUALLY WORKS
- Speakers later become <|speaker:0|> (M) and <|speaker:1|> (F). You still output M/F.
- Identity (timbre, Indian-English, resting warmth) comes from reference wavs.
  Tags do not invent identity. Do not describe the voice. Do not use [warm].
- Tags are free-form [natural language] at the WORD. Placement is the control:
  • [emphasis] immediately BEFORE the stressed word: That's [emphasis] baati.
  • [short pause] BETWEEN words (or after a notice invite), not "..."
  • Turn-level delivery BEFORE the line: [curious] So this isn't only for show?
  • Intra-turn shift: two clauses, tag at each beat:
    [curious] Wait — Fine Hamlet? [soft awe] That changes it.
  • Events: [chuckle] or [inhale] as their own beat. One channel only — never
    "Hah, [chuckle] …" and never [laugh] / [laughing].
- Most turns have NO delivery tag. Clone is already warm. Sparse > demo reel.
- Do not use CosyVoice emphasize: ["word"]. Do not use SSML.

## ALLOWED TAGS
  [emphasis]     — names, food, the one word that should land
  [short pause]  — observation / notice beats
  [curious]      — real questions, noticing
  [chuckle]      — one light banter beat (NOT [laugh])
  [inhale]       — rare, before quiet wonder
  [soft awe] / [low voice] — quiet wonder
  [excited]      — at most 1–2 peaks in the whole episode, never shouty
Never: [laugh], [laughing], [shouting], [whisper], [warm], [with strong accent].

## REWRITE RULES (curation, not decoration)
- Split/merge so each tag has a hang-point. One intention per turn unless you
  write an explicit two-beat shift (max two delivery tags).
- [emphasis] immediately before the word. Never before a quote:
  GOOD: the [emphasis] Kathputli.  BAD: the [emphasis] 'Kathputli.'
- [short pause] ONLY after a notice invite, then the turn stops.
  GOOD: If you catch a [emphasis] dhol later [short pause] take a quiet second.
  BAD: living recreation [short pause] not an ancient village
  BAD: Kalbeliya or [emphasis] Ghumar [short pause] in the ghagras
  Do not use [short pause] as a comma.
- One [chuckle] in the whole episode, on the camel/elephant beat:
  [chuckle] Why not both?
  Do not repeat "Why not both" after the dancers.
- [curious] only if the turn IS mainly a question, not a tagged paragraph.
  BAD: It's breathtaking! [curious] Did you know … heritage.
  GOOD: [curious] Fine Hamlet — that's the name?
- F does not enumerate dancers or dishes. M can; one [emphasis] per name max,
  and not every name needs it.
- Most turns: zero delivery tags. A few [emphasis] on names/food is enough.
- Keep situation-safe conditionals. Ban slogan endings.
- Keep every cultural name and dish from the source lines you were given.
  Do not drop camel, elephant, bullock, Kathputli, lanterns/diyas, baati/ghee,
  or manuhar. Do not collapse the episode to a 12-line outline.
- Do not densify: spoken word count must stay near the source. Light situation-safe
  rewrites only — not a shorter highlight reel.
- Source lines are ALREADY split. Return one output line per source_line index.
  You may lightly rewrite a line (situation-safe, tags). Do not merge two
  indexes into one paragraph. Do not omit an index.
- Do not write "Hah". Do not invent sites.

## BAD
"It's breathtaking! [curious] Did you know Chokhi Dhani means Fine Hamlet? It's a living recreation…"
"Prepare for [emphasis] Dal [short pause] Baati, and Churma."
"Why not both? A camel ride gives a fantastic view [short pause] and there's the bullock cart…"

## GOOD
"Here we are. That's [emphasis] Chokhi Dhani."
"[curious] Fine Hamlet — that's the name?"
"If you catch a [emphasis] dhol later [short pause] take a quiet second."
"[chuckle] Why not both?"
"The stars are [emphasis] baati, daal, and choorma."

## OUTPUT
Return ONLY valid JSON:
{{"lines":[{{"speaker":"M","text":"...words with optional [tags] in place...","section":"arrival"}}, ...]}}
section must be one of: {", ".join(_SECTIONS)}.
Speakers: only "M" or "F".
"""


def _parse_fish_lines(
    raw_lines: Any,
    *,
    tagged: bool,
) -> list[dict[str, str]]:
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError("Agent returned no lines")
    out: list[dict[str, str]] = []
    budget = _TagBudget()
    for raw in raw_lines:
        if not isinstance(raw, dict):
            continue
        sp = str(raw.get("speaker") or "").strip().upper()
        if sp in ("1", "SPEAKER 1", "S1"):
            sp = "M"
        elif sp in ("2", "SPEAKER 2", "S2"):
            sp = "F"
        if sp not in ("M", "F"):
            raise ValueError(f"Invalid speaker: {raw!r}")
        raw_text = str(raw.get("text") or "")
        if tagged:
            text = sanitize_fish_text(raw_text, budget=budget)
        else:
            text = _light_clean(raw_text)
            text = _BRACKET.sub("", text)
            text = _light_clean(text)
            if text and text[0].islower():
                text = text[0].upper() + text[1:]
        text = _strip_slogan_sentences(text)
        if text and not tagged and text[0].islower():
            text = text[0].upper() + text[1:]
        if tagged and text:
            text = _capitalize_spoken_start(text)
        if not text:
            continue
        section = str(raw.get("section") or "culture").strip().lower()
        if section not in _SECTIONS:
            section = "culture"
        out.append({"speaker": sp, "text": text, "section": section})
    if not out:
        raise ValueError("Agent produced only empty lines")
    return out


def _source_payload(
    turns: list[ParsedTurn] | list[dict[str, str]],
) -> list[dict[str, Any]]:
    source: list[dict[str, Any]] = []
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
    return source


def split_source_for_fish(
    turns: list[ParsedTurn],
    *,
    target_words: int = 20,
) -> list[ParsedTurn]:
    """Split long source paragraphs into Fish-sized turns without dropping facts."""
    out: list[ParsedTurn] = []
    for t in turns:
        sents = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", t.text.strip())
            if s.strip()
        ]
        buf: list[str] = []
        n = 0
        for sent in sents:
            w = len(sent.split())
            if buf and n + w > target_words:
                out.append(
                    ParsedTurn(
                        index=len(out) + 1,
                        speaker=t.speaker,
                        text=" ".join(buf),
                    )
                )
                buf, n = [sent], w
            else:
                buf.append(sent)
                n += w
        if buf:
            out.append(
                ParsedTurn(index=len(out) + 1, speaker=t.speaker, text=" ".join(buf))
            )
    return out


_REQUIRED_BEATS: list[tuple[str, re.Pattern[str]]] = [
    ("Chokhi Dhani", re.compile(r"chokh[ei]", re.I)),
    ("Fine Hamlet / Special Village", re.compile(r"fine hamlet|special village", re.I)),
    ("living recreation / not ancient", re.compile(r"recreation|not an ancient", re.I)),
    ("dhol", re.compile(r"dhohl|\bdhol\b", re.I)),
    ("folk music", re.compile(r"folk music", re.I)),
    ("clay", re.compile(r"\bclay\b", re.I)),
    ("lanterns / diyas", re.compile(r"lantern|diya", re.I)),
    ("Kalbelia", re.compile(r"kalbeliya|kalbelia", re.I)),
    ("Ghoomar", re.compile(r"ghumar|ghoomar", re.I)),
    ("ghagra", re.compile(r"ghaagra|ghagra", re.I)),
    ("Kathputli", re.compile(r"kathputl", re.I)),
    ("camel", re.compile(r"\bcamel", re.I)),
    ("elephant", re.compile(r"elephant", re.I)),
    ("bullock", re.compile(r"bullock", re.I)),
    ("mehendi / henna", re.compile(r"mehendi|henna", re.I)),
    ("potter", re.compile(r"potter", re.I)),
    ("weaver", re.compile(r"weaver", re.I)),
    ("miniature", re.compile(r"miniature", re.I)),
    ("baati", re.compile(r"\bbaati", re.I)),
    ("dal", re.compile(r"\bdaal\b|\bdal\b", re.I)),
    ("churma", re.compile(r"choorma|churma", re.I)),
    ("ghee", re.compile(r"ghee", re.I)),
    ("gatte", re.compile(r"gutte|gatte", re.I)),
    ("bajra", re.compile(r"bajr", re.I)),
    ("manuhar", re.compile(r"manuhaar|manuhar", re.I)),
    ("leaf platter or low stools", re.compile(r"leaf platter|low stool", re.I)),
]
_MIN_TURNS = 22
_MIN_SPOKEN_WORDS = 400
# Reject drafts that shrink the mechanical split / source mass.
_SOURCE_WORD_RATIO = 0.90
_SOURCE_TURN_RATIO = 0.90


def _spoken_blob(rows: list[dict[str, str]] | list[ParsedTurn]) -> str:
    parts: list[str] = []
    for row in rows:
        text = row["text"] if isinstance(row, dict) else row.text
        parts.append(strip_fish_tags(text))
    return "\n".join(parts)


def _spoken_stats(rows: list[dict[str, str]] | list[ParsedTurn]) -> tuple[int, int]:
    """Return (turn_count, spoken_word_count)."""
    return len(rows), len(_spoken_blob(rows).split())


def _min_turns_for_source(source_turns: int | None) -> int:
    if source_turns is None:
        return _MIN_TURNS
    return max(_MIN_TURNS, int(source_turns * _SOURCE_TURN_RATIO))


def _min_words_for_source(source_words: int | None) -> int:
    if source_words is None:
        return _MIN_SPOKEN_WORDS
    return max(_MIN_SPOKEN_WORDS, int(source_words * _SOURCE_WORD_RATIO))


def _coverage_gaps(
    rows: list[dict[str, str]] | list[ParsedTurn],
    *,
    source_words: int | None = None,
    source_turns: int | None = None,
) -> list[str]:
    blob = _spoken_blob(rows)
    missing = [name for name, pat in _REQUIRED_BEATS if not pat.search(blob)]
    n = len(rows)
    words = len(blob.split())
    min_turns = _min_turns_for_source(source_turns)
    min_words = _min_words_for_source(source_words)
    if n < min_turns:
        if source_turns is not None:
            missing.append(
                f"too few turns ({n}; need ≥{min_turns} vs {source_turns} source)"
            )
        else:
            missing.append(f"too few turns ({n}; need ≥{min_turns})")
    if words < min_words:
        if source_words is not None:
            missing.append(
                f"too thin vs source ({words} vs {source_words} words; "
                f"need ≥{min_words})"
            )
        else:
            missing.append(f"too few words ({words}; need ≥{min_words})")
    return missing


def run_fish_editor(
    turns: list[ParsedTurn],
    *,
    model: str = DEFAULT_MOSS_DIRECTOR_MODEL,
    client: OpenAI | None = None,
) -> list[dict[str, str]]:
    """Situation-safe companion edit, shaped for Fish hang-points. No [tags]."""
    client = client or _client()
    src_turns, src_words = _spoken_stats(turns)
    min_turns = _min_turns_for_source(src_turns)
    min_words = _min_words_for_source(src_words)
    payload = {
        "source_lines": [
            {"index": t.index, "speaker": t.speaker, "text": t.text} for t in turns
        ]
    }
    user = (
        "Edit this on-site travel dialogue for Fish S2. Rewrite in place / split "
        "further — do NOT summarise or densify. Cover every MUST COVER beat. "
        f"Keep ≥{min_words} spoken words and ≥{min_turns} turns "
        f"(source has {src_words} words, {src_turns} turns). "
        "Do not add [tags]. Do not write a MOSS or CosyVoice script.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    parsed: list[dict[str, str]] = []
    for attempt in range(3):
        data = _chat_json(
            client,
            model,
            _fish_editor_system(),
            user,
            temperature=0.45 if attempt == 0 else 0.55,
        )
        parsed = _parse_fish_lines(
            data.get("lines") if isinstance(data, dict) else data,
            tagged=False,
        )
        gaps = _coverage_gaps(
            parsed, source_words=src_words, source_turns=src_turns
        )
        words = len(_spoken_blob(parsed).split())
        print(
            f"  editor attempt {attempt + 1}: {len(parsed)} turns, {words} words "
            f"(source {src_turns} turns / {src_words} words)"
        )
        if not gaps:
            return parsed
        print("  missing:", "; ".join(gaps))
        user = (
            "Your previous draft DROPPED required content or shrank the episode "
            f"({words} spoken words / {len(parsed)} turns vs source "
            f"{src_words} words / {src_turns} turns). "
            "Rewrite the FULL episode again in place. Do not outline or tighten. "
            f"Keep ≥{min_words} spoken words and ≥{min_turns} turns. "
            "Explain baati (hard wheat rolls, ghee), dal, churma, camel view, "
            "elephant, Kathputli (kings and queens). Restore:\n- "
            + "\n- ".join(gaps)
            + "\nKeep all other source facts. Do not add [tags].\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )
    print("  editor still missing beats; using last draft")
    return parsed


def run_fish_director(
    turns: list[ParsedTurn] | list[dict[str, str]],
    *,
    model: str = DEFAULT_MOSS_DIRECTOR_MODEL,
    client: OpenAI | None = None,
) -> tuple[list[ParsedTurn], list[str]]:
    """Rewrite as a Fish score (wording + in-place tags)."""
    client = client or _client()
    source = _source_payload(turns)
    n_src = len(source)
    src_turns, src_words = _spoken_stats(source)
    min_words = _min_words_for_source(src_words)
    user = (
        "Rewrite as a Fish S2 score. Each source_line is already one turn — "
        f"return {n_src} lines (one per index). Reshape wording so tags have "
        "hang-points, then place tags. Situation-safe: no 'look over there', "
        "'listen to that', 'time portal', or 'as the sun sets' as hard facts. "
        f"Do not merge turns. Do not drop facts. Keep ≥{min_words} spoken words "
        f"(source has {src_words}).\n\n"
        + json.dumps({"source_lines": source}, ensure_ascii=False)
    )
    parsed: list[dict[str, str]] = []
    for attempt in range(3):
        data = _chat_json(
            client,
            model,
            _fish_writer_system(),
            user,
            temperature=0.5 if attempt == 0 else 0.55,
        )
        parsed = _parse_fish_lines(
            data.get("lines") if isinstance(data, dict) else data,
            tagged=True,
        )
        gaps = _coverage_gaps(
            parsed, source_words=src_words, source_turns=src_turns
        )
        if len(parsed) < n_src - 2:
            gaps.append(f"merged turns ({len(parsed)} vs {n_src} source lines)")
        words = len(_spoken_blob(parsed).split())
        print(
            f"  writer attempt {attempt + 1}: {len(parsed)} turns, {words} words "
            f"(source {src_turns} turns / {src_words} words)"
        )
        if not gaps:
            break
        print("  missing:", "; ".join(gaps))
        user = (
            f"Return exactly {n_src} lines, one per source_line index. "
            "Do not merge turns into long paragraphs. Do not densify. "
            f"Keep ≥{min_words} spoken words (source {src_words}). Restore:\n- "
            + "\n- ".join(gaps)
            + "\nKeep tags in place. Situation-safe wording.\n\n"
            + json.dumps({"source_lines": source}, ensure_ascii=False)
        )
    else:
        print("  writer still missing beats; using last draft")
    out: list[ParsedTurn] = []
    sections: list[str] = []
    for i, row in enumerate(parsed, start=1):
        out.append(ParsedTurn(index=i, speaker=row["speaker"], text=row["text"]))  # type: ignore[arg-type]
        sections.append(row["section"])
    return out, sections


def build_fish_cues(
    turns: list[ParsedTurn],
    sections: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Cues use spoken text (tags stripped) so mix word-share still works."""
    spoken = [
        ParsedTurn(index=t.index, speaker=t.speaker, text=strip_fish_tags(t.text))
        for t in turns
    ]
    cues = build_cues(spoken, sections)
    for c, t in zip(cues, turns):
        c["tagged_text"] = t.text
    return cues


def run_fish_travel_pipeline(
    turns: list[ParsedTurn],
    *,
    model: str = DEFAULT_MOSS_DIRECTOR_MODEL,
    skip_editor: bool = False,
) -> tuple[list[ParsedTurn], list[dict[str, Any]]]:
    """Fish editor (optional) → Fish writer → pronunciations → cues."""
    client = _client()
    split = split_source_for_fish(turns)
    print(
        f"Split source into {len(split)} short turns "
        f"({len(_spoken_blob(split).split())} words)"
    )
    if skip_editor:
        directed_source: list[ParsedTurn] | list[dict[str, str]] = split
        print("Skipping Fish editor; writer will tag the split source.")
    else:
        print("Fish editor (situation-safe, Fish-shaped turns, no tags)...")
        directed_source = run_fish_editor(split, model=model, client=client)
        split_turns, split_words = _spoken_stats(split)
        if _coverage_gaps(
            directed_source,
            source_words=split_words,
            source_turns=split_turns,
        ):
            print(
                "Editor still thin or missing beats; "
                "writer will use the split source instead."
            )
            directed_source = split
    print("Fish writer (reshape wording + in-place tags)...")
    directed, sections = run_fish_director(directed_source, model=model, client=client)
    print("Applying pronunciations (lexicon speak-forms; tags kept in place)...")
    if os.getenv("PRONUNCIATION_LLM", "").strip() in ("1", "true", "yes"):
        from pipeline.pronounce import (
            extract_candidate_terms,
            merge_speak_forms_into_yaml,
            resolve_speak_forms,
        )

        terms = extract_candidate_terms([strip_fish_tags(t.text) for t in directed])
        forms = resolve_speak_forms(terms, use_llm=True, model=model)
        merge_speak_forms_into_yaml(forms)
        print(f"  pronunciation LLM refreshed {len(forms)} lexicon entries")
    directed = apply_pronunciations_to_turns(directed)
    budget = _TagBudget()
    directed = [
        ParsedTurn(
            index=t.index,
            speaker=t.speaker,
            text=sanitize_fish_text(t.text, budget=budget),
        )
        for t in directed
    ]
    kept: list[ParsedTurn] = []
    kept_sections: list[str] = []
    for t, sec in zip(directed, sections):
        if t.text:
            kept.append(t)
            kept_sections.append(sec)
    directed, sections = _repair_episode_tags(kept, kept_sections)
    sections = sections or []
    directed, sections = _ensure_ride_beats(directed, sections)
    directed = [
        ParsedTurn(index=i, speaker=t.speaker, text=t.text)
        for i, t in enumerate(directed, start=1)
        if t.text
    ]
    gaps = _coverage_gaps(directed)
    words = len(_spoken_blob(directed).split())
    print(f"Final score: {len(directed)} turns, {words} spoken words")
    if gaps:
        print("  remaining coverage gaps:", "; ".join(gaps))
    cues = build_fish_cues(directed, sections[: len(directed)])
    return directed, cues
