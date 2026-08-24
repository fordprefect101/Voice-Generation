"""Multi-agent script pipeline: raw dialogue -> actable lines.json for TTS.

Agent 1 (Writer): rewrite in locked M/F host personalities + soft emotion keys.
Agent 2 (Prosody): refine punctuation beats + inter-turn pauses from full context.
Agent 3 (Emphasis): context-aware word stress targets for CosyVoice instruct.
Code: normalize punctuation + diversify pauses if the model collapses.
"""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from pipeline.emotion_config import (
    allowed_emotions,
    default_emotion,
    emotion_list_for_prompt,
)
from pipeline.emotions import normalize_line_emotion
from pipeline.hosts import hosts_brief_for_prompt
from pipeline.models import EnrichedLine, ParsedTurn, normalize_emphasize
from pipeline.script_postprocess import diversify_pauses, normalize_pause_punctuation

DEFAULT_SCRIPT_MODEL = "gpt-4o"


def _writer_system() -> str:
    emo_list = emotion_list_for_prompt()
    default = default_emotion()
    cast = hosts_brief_for_prompt()
    return f"""You are a podcast performance WRITER for a two-host travel AUDIO GUIDE.
Each line will be spoken by TTS (CosyVoice). The WORDS must carry acting, timing,
WORD EMPHASIS, and each host's locked personality. Write for the ear, not the page.

{cast}

TONE — friendly LEARNING companions, not cheerleaders:
- Share discoveries and teach lightly. Curious, warm, clear.
- Explain WHY something matters. React like friends walking together.
- Prefer wonder + clarity over hype. Smile in the voice, never manic brochure energy.
- Ban (unless truly earned once): spectacular, incredible, pure magic, legendary,
  unforgettable, infectious, breathtaking, and stacked !! hype.
- Stay in a calm-friendly band. Do NOT force a new mood every turn.
- SITUATION-SAFE: never assert unseen scenes ("Look over there", "Listen to that",
  "Those are the dancers", "As the sun sets" as fact). Prefer "If you can see…",
  "You may hear…", "Around evening…".
- ENDING: no tourism slogans; leave a useful next step or quiet observation.
- F lines: reactions and questions in full sentences — not staccato checklist items.

HARD CONSTRAINTS:
- Same number of turns, same index, same speaker M/F as input. Do not merge/split.
- Keep factual content (names, places, food); you MAY reshape HOW it is said.
- Write M lines in M's voice (guide/context). Write F lines in F's voice (curiosity/reaction).
- Prefer explain → react/question → deepen. Occasional role reverse is fine when natural.

EMOTION KEYS (delivery tilt only — not a full personality rewrite):
- Prefer "{default}" or "neutral" for most lines.
  "neutral" = living warm identity (NOT flat).
  "{default}" = soft friendly hospitality tilt.
- Use other keys ONLY when the line clearly needs that tilt.
- If unsure, use "{default}".
- Allowed keys:
{emo_list}

WRITE LIKE SPEECH:
- Short breath groups. Do not dump six exclamations in one turn.
- Use ! very sparingly (~1 per 2–3 turns across the episode). Prefer ? and .
- Do NOT end every clause with !. Do NOT use ASCII hyphen "-" as a pause — use … or —.
- React to what was just said so it feels like conversation between these two hosts.

WORD EMPHASIS (critical for TTS — no SSML):
- Put a held beat BEFORE the stressed word/phrase with … or —:
  e.g. "This… is Chokhi Dhani." / "The name means… a 'fine hamlet'."
- Isolate punch lines: "Not just a meal. An event."
- Use contrast: "It's not an ancient village — it's a living recreation."
- Name → define → friendly take: "Manuhar. That loving insistence to eat more."
- Break lists of key terms: "Dal… Baati… and Churma."
- Aim for 1–2 intentional stress beats on informative turns. Do not stress every noun.
- Do NOT use ALL CAPS for emphasis.

Output ONLY valid JSON:
{{"lines":[{{"index":1,"speaker":"M","text":"...","emotion":"{default}"}}, ...]}}
Do not include pause fields yet.
"""


def _prosody_system() -> str:
    emo_list = emotion_list_for_prompt()
    default = default_emotion()
    cast = hosts_brief_for_prompt()
    return f"""You are a podcast PROSODY / TIMING director for TTS.
You receive already-written spoken lines in locked host voices. Improve ONLY:
1) punctuation for in-line beats and WORD EMPHASIS (… — , ? sparse !)
2) pause_before_ms / pause_after_ms between turns using FULL conversation context

{cast}

Do not change speaker or index. Keep meaning and each host's personality.
You may lightly tweak wording only to improve spoken rhythm / emphasis
(e.g. insert … before a key name), not to add facts or turn teaching into hype.

Do NOT reassign emotions for variety. Keep the writer's emotion unless it is
invalid — then use "{default}". Emotion is only a delivery tilt.

PUNCTUATION / EMPHASIS:
- commas = short breath; … or — = held beat before a stressed word/phrase
- ? = question; ! rare (do not add more ! than already present unless needed)
- never use "-" as a pause cue; never ALL CAPS for stress
- preserve teaching rhythm: define beats, list breaks, contrast dashes
- after F questions, give M a little pause_before room to answer

INTER-TURN PAUSES (ms) — MUST VARY; never identical pause_after on every line:
- quick banter: 150–280 after
- normal / explanatory: 280–420 after
- after a question / reveal / definition: 450–700 after
- topic shift: 400–650 after
- pause_before usually 0–100; 80–180 when answering a charged question

Allowed emotion keys:
{emo_list}

Output ONLY valid JSON:
{{"lines":[{{"index":1,"speaker":"M","text":"...","emotion":"{default}","pause_before_ms":0,"pause_after_ms":320}}, ...]}}
"""


def _emphasis_system() -> str:
    cast = hosts_brief_for_prompt()
    return f"""You are an EMPHASIS / WORD-STRESS director for CosyVoice TTS.
You receive the full episode of spoken lines. Decide which WORDS need spoken STRESS
(stronger/clearer delivery on that word — NOT a pause, NOT elongated spelling like "mooore").

{cast}

RULES:
- Read FULL context: what was just said, what this line is doing (guide vs react).
- Prefer contrast words, key names/places, reveals, punch words (e.g. "more", "living", "Chokhi Dhani").
- Skip filler: the, a, and, to, of, just, really, like, you, we, it, is, are…
- 0–2 spans per line. Many lines should have [] (empty) — do not force stress.
- Each emphasize span MUST be an exact substring of that line's text (same spelling).
- Prefer single words; short multi-word names/phrases allowed (e.g. "Chokhi Dhani").
- Do NOT change text, speaker, index, emotion, or pauses.
- Do NOT use ALL CAPS or invented spellings.

Output ONLY valid JSON:
{{"lines":[{{"index":1,"emphasize":["more"]}}, {{"index":2,"emphasize":[]}}, ...]}}
Include every input index exactly once.
"""


def _client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not set. Add it to .env or export it in your shell."
        )
    return OpenAI()


def _chat_json(client: OpenAI, model: str, system: str, user: str) -> Any:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def _validate_alignment(raw_lines: list[dict], turns: list[ParsedTurn]) -> None:
    if len(raw_lines) != len(turns):
        raise ValueError(f"got {len(raw_lines)} lines, expected {len(turns)}")
    for raw, orig in zip(raw_lines, turns):
        if int(raw["index"]) != orig.index:
            raise ValueError(f"index mismatch {raw.get('index')} vs {orig.index}")
        if raw.get("speaker", orig.speaker) != orig.speaker:
            raise ValueError(f"speaker changed on {orig.index}")


def _emotion(raw: dict, speaker: str) -> str:
    del speaker  # personality is in text + identity ref; emotion is delivery tilt only
    return normalize_line_emotion(raw.get("emotion"))


def run_writer(
    turns: list[ParsedTurn],
    *,
    model: str,
    client: OpenAI | None = None,
) -> list[dict]:
    client = client or _client()
    payload = {
        "turns": [
            {"index": t.index, "speaker": t.speaker, "text": t.text} for t in turns
        ],
        "allowed_emotions": sorted(allowed_emotions()),
        "default_emotion": default_emotion(),
    }
    data = _chat_json(
        client,
        model,
        _writer_system(),
        "Rewrite these turns as natural two-host travel-companion speech for TTS. "
        "Keep M as guide/context and F as curiosity/reaction. "
        "Put expressiveness in the words; use emotion keys sparingly as delivery tilts.\n\n"
        + json.dumps(payload, ensure_ascii=False),
    )
    lines = data["lines"] if isinstance(data, dict) else data
    _validate_alignment(lines, turns)
    out = []
    for raw, orig in zip(lines, turns):
        out.append(
            {
                "index": orig.index,
                "speaker": orig.speaker,
                "text": normalize_pause_punctuation(str(raw["text"]).strip()),
                "emotion": _emotion(raw, orig.speaker),
            }
        )
    return out


def run_prosody(
    draft_lines: list[dict],
    turns: list[ParsedTurn],
    *,
    model: str,
    client: OpenAI | None = None,
) -> list[EnrichedLine]:
    client = client or _client()
    data = _chat_json(
        client,
        model,
        _prosody_system(),
        "Refine punctuation and assign varied pauses for this episode. "
        "Preserve each host's personality.\n\n"
        + json.dumps({"lines": draft_lines}, ensure_ascii=False),
    )
    lines = data["lines"] if isinstance(data, dict) else data
    _validate_alignment(lines, turns)
    enriched: list[EnrichedLine] = []
    for raw, orig in zip(lines, turns):
        enriched.append(
            EnrichedLine(
                index=orig.index,
                speaker=orig.speaker,
                text=normalize_pause_punctuation(str(raw["text"]).strip()),
                emotion=_emotion(raw, orig.speaker),
                pause_before_ms=int(raw.get("pause_before_ms", 0)),
                pause_after_ms=int(raw.get("pause_after_ms", 350)),
            )
        )
    return diversify_pauses(enriched)


def run_emphasis(
    lines: list[EnrichedLine],
    turns: list[ParsedTurn],
    *,
    model: str,
    client: OpenAI | None = None,
) -> list[EnrichedLine]:
    """Agent 3: context-aware word stress targets for CosyVoice."""
    client = client or _client()
    payload = {
        "lines": [
            {
                "index": L.index,
                "speaker": L.speaker,
                "text": L.text,
                "emotion": L.emotion,
            }
            for L in lines
        ]
    }
    data = _chat_json(
        client,
        model,
        _emphasis_system(),
        "Choose word-stress targets from full episode context. "
        "Return emphasize spans that appear exactly in each line's text.\n\n"
        + json.dumps(payload, ensure_ascii=False),
    )
    raw_lines = data["lines"] if isinstance(data, dict) else data
    by_index: dict[int, list[str]] = {}
    for raw in raw_lines:
        idx = int(raw["index"])
        by_index[idx] = list(raw.get("emphasize") or [])

    expected = sorted(L.index for L in lines)
    if sorted(by_index.keys()) != expected:
        print("Warning: emphasis agent index set mismatch; filling gaps with []")

    out: list[EnrichedLine] = []
    for L, orig in zip(lines, turns):
        if L.index != orig.index or L.speaker != orig.speaker:
            raise ValueError("emphasis pass corrupted speaker/index alignment")
        L.emphasize = normalize_emphasize(by_index.get(L.index, []), L.text)
        out.append(L)
    return out


def run_script_agents(
    turns: list[ParsedTurn],
    *,
    model: str = DEFAULT_SCRIPT_MODEL,
) -> list[EnrichedLine]:
    """Full Phase A: Writer -> Prosody -> Emphasis -> code postprocess."""
    client = _client()
    print(f"Agent 1/3 Writer ({model})...")
    draft = run_writer(turns, model=model, client=client)
    print(f"Agent 2/3 Prosody ({model})...")
    lines = run_prosody(draft, turns, model=model, client=client)
    print(f"Agent 3/3 Emphasis ({model})...")
    lines = run_emphasis(lines, turns, model=model, client=client)
    afters = [L.pause_after_ms for L in lines]
    print(f"pause_after_ms: {afters}")
    print(f"unique pause_after: {len(set(afters))}")
    emotions = [L.emotion for L in lines]
    print(f"emotions: {emotions}")
    print(f"emphasize: {[L.emphasize for L in lines]}")
    return lines
