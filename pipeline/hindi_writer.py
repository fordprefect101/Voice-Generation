"""Hindi writer: English Fish score → Hindi Fish score, one line per line.

Input is the finished English Fish script (tags, turn shape and cues already set).
The writer only re-voices the words; [tags], speakers, sections and pauses carry over.

Three written forms, because Fish gives no guidance on Hindi input — listen and pick:
  devanagari  everything in Devanagari, English loanwords included
  mixed       Hindi in Devanagari, English loanwords in Latin
  roman       everything in Latin (typed Hinglish)
"""
from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from pipeline.director_common import DEFAULT_DIRECTOR_MODEL, chat_json
from pipeline.fish_director import _BRACKET, strip_fish_tags
from pipeline.fish_timeline import group_utterances, utterance_clip_name
from pipeline.hosts import hosts_brief_for_prompt
from pipeline.models import ParsedTurn
from pipeline.pronounce import load_pronunciations

FORMS = ("devanagari", "mixed", "roman")

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")

_FORM_RULES = {
    "devanagari": """WRITTEN FORM — devanagari:
- Every spoken word in Devanagari, including English loanwords, spelled the way
  Hindi speakers say them: कैमल राइड, परफ़ॉर्मेंस, म्यूज़ियम.
- No Latin letters anywhere outside [tags].
- Sentence end: । (or ? / !).
GOOD: "[emphasis] कैमल राइड से पूरा मेला दिखता है।"
""",
    "mixed": """WRITTEN FORM — mixed:
- Hindi words in Devanagari. English loanwords stay in Latin letters, spelled as
  normal English: camel ride, performance, museum.
- Rajasthani / Hindi names and dishes are Hindi words → Devanagari.
- Sentence end: । (or ? / !).
GOOD: "[emphasis] Camel ride से पूरा मेला दिखता है।"
""",
    "roman": """WRITTEN FORM — roman:
- Everything in Latin letters, the way urban Indians type Hinglish.
- No Devanagari anywhere.
- Keep names and dishes exactly as spelled in the source line (they are already
  speak-forms: Chokhee Dhaanee, Kalbeliya, daal baati choorma).
- Sentence end: . (or ? / !).
GOOD: "[emphasis] Camel ride se poora mela dikhta hai."
""",
}


def _names_block() -> str:
    """Source lines carry English speak-forms; show the real spellings behind them."""
    rows = load_pronunciations()
    if not rows:
        return ""
    pairs = "\n".join(f'- "{r["to"]}" is the respelling of "{r["from"]}"' for r in rows)
    return f"""## NAMES IN THE SOURCE ARE RESPELLED FOR AN ENGLISH VOICE
{pairs}
In devanagari and mixed form, write the TRUE Hindi spelling instead
(चोखी ढाणी, कालबेलिया, घूमर, कठपुतली, दाल बाटी चूरमा, गट्टे की सब्ज़ी, बाजरे की रोटी, मनुहार, ढोल).
"""


def _hindi_writer_system(form: str) -> str:
    cast = hosts_brief_for_prompt()
    return f"""You are a HINDI WRITER for a two-host travel AUDIO GUIDE spoken by Fish Audio S2.
You receive the finished English score. You re-voice each line in natural spoken
Hindi. You do not restructure the episode.

{cast}

## YOUR JOB
- Transcreate, do not translate word for word. Write what these two hosts would
  actually SAY to a friend on site — everyday conversational Hindi, the register
  urban Indians speak, with the English loanwords they naturally use
  (ride, performance, museum, souvenir). Never stiff textbook or news Hindi
  (no शुद्ध words like "अविस्मरणीय", "दर्शनीय", "कृपया ध्यान दें").
- Keep each host's personality: M guides and gives context, F is curious and reacts.
- Keep every fact, name and dish from the line. Do not add or drop information.
- Keep it situation-safe: conditionals stay conditional ("अगर आगे कहीं ढोल सुनाई दे…").
- Numbers and years as spoken words, never digits.

## ONE LINE IN → ONE LINE OUT
- Return exactly one line per source index, same index, same speaker.
- Do not merge, split, reorder or skip lines.

## [TAGS] ARE PART OF THE SCORE
- Tags stay in English, exactly as written: [emphasis], [curious], [short pause], …
- Keep every tag from the source line, in the same order. Do not add new tags.
- Hindi word order differs from English. Move the tag so it still sits
  immediately BEFORE the same word or beat it marked:
  EN: "That's [emphasis] baati."   →   "यही है [emphasis] बाटी।"
- A turn-level tag stays at the start of the line:
  EN: "[curious] Fine Hamlet — that's the name?"

{_names_block()}
{_FORM_RULES[form]}
## OUTPUT
Return ONLY valid JSON:
{{"lines":[{{"index":1,"speaker":"M","text":"...Hindi with [tags] in place..."}}, ...]}}
"""


def tag_keys(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", t.strip().lower()) for t in _BRACKET.findall(text)]


def _form_problem(text: str, form: str) -> str | None:
    spoken = strip_fish_tags(text)
    if form == "devanagari" and _LATIN.search(spoken):
        return "has Latin letters outside [tags]"
    if form == "roman" and _DEVANAGARI.search(spoken):
        return "has Devanagari letters"
    if form == "devanagari" and not _DEVANAGARI.search(spoken):
        return "has no Devanagari"
    return None


def check_lines(source: list[ParsedTurn], raw_lines: Any, form: str) -> list[str]:
    """Return human-readable problems; empty list means the Hindi matches the score."""
    if not isinstance(raw_lines, list):
        return ["writer returned no lines"]
    if len(raw_lines) != len(source):
        return [f"expected {len(source)} lines, got {len(raw_lines)}"]
    problems: list[str] = []
    for src, raw in zip(source, raw_lines):
        row = raw if isinstance(raw, dict) else {}
        text = str(row.get("text") or "").strip()
        speaker = str(row.get("speaker") or "").strip().upper()
        if not strip_fish_tags(text):
            problems.append(f"line {src.index}: empty")
            continue
        if speaker != src.speaker:
            problems.append(f"line {src.index}: speaker {speaker!r}, expected {src.speaker!r}")
        if tag_keys(text) != tag_keys(src.text):
            problems.append(
                f"line {src.index}: tags {tag_keys(text)}, expected {tag_keys(src.text)}"
            )
        if issue := _form_problem(text, form):
            problems.append(f"line {src.index}: {issue}")
    if form == "mixed" and not problems:
        if not any(_DEVANAGARI.search(str(r.get("text") or "")) for r in raw_lines):
            problems.append("mixed form has no Devanagari at all")
    return problems


def run_hindi_writer(
    turns: list[ParsedTurn],
    *,
    form: str,
    model: str = DEFAULT_DIRECTOR_MODEL,
    client: OpenAI,
) -> list[ParsedTurn]:
    """English Fish turns → Hindi Fish turns in the given written form."""
    if form not in FORMS:
        raise ValueError(f"Unknown form {form!r}; expected one of {FORMS}")
    payload = json.dumps(
        {
            "source_lines": [
                {"index": t.index, "speaker": t.speaker, "text": t.text} for t in turns
            ]
        },
        ensure_ascii=False,
        indent=2,
    )
    system = _hindi_writer_system(form)
    user = payload
    problems: list[str] = []
    for attempt in (1, 2):
        data = chat_json(client, model, system, user)
        raw_lines = data.get("lines") if isinstance(data, dict) else None
        problems = check_lines(turns, raw_lines, form)
        if not problems:
            return [
                ParsedTurn(index=t.index, speaker=t.speaker, text=str(r["text"]).strip())
                for t, r in zip(turns, raw_lines)
            ]
        print(f"  {form}: {len(problems)} problem(s) on attempt {attempt}")
        user = (
            payload
            + "\n\nYour previous answer broke these rules. Fix them and return ALL lines again:\n"
            + "\n".join(f"- {p}" for p in problems)
        )
    raise SystemExit(
        f"Hindi writer ({form}) still breaks the score after a retry:\n"
        + "\n".join(f"  {p}" for p in problems)
    )


def hindi_cues(
    cues: list[dict[str, Any]],
    hindi: list[ParsedTurn],
) -> list[dict[str, Any]]:
    """Copy the English cues; swap only the spoken text. Timing stays English-derived."""
    if len(cues) != len(hindi):
        raise ValueError(f"{len(cues)} cues but {len(hindi)} Hindi lines")
    out: list[dict[str, Any]] = []
    for c, t in zip(cues, hindi):
        row = dict(c)
        row["text"] = strip_fish_tags(t.text)
        row["tagged_text"] = t.text
        out.append(row)
    return out


def bakeoff_slice(turns: list[ParsedTurn], n: int) -> list[ParsedTurn]:
    """First n turns, trimmed so the sample ends on the other speaker.

    Forms are laid back to back; ending on a different speaker than the sample
    starts with keeps utterance grouping from merging two forms into one clip.
    """
    sample = turns[: max(1, n)]
    while len(sample) > 1 and sample[-1].speaker == sample[0].speaker:
        sample = sample[:-1]
    return sample


def build_bakeoff(
    turns: list[ParsedTurn],
    cues: list[dict[str, Any]],
    *,
    n: int,
    model: str = DEFAULT_DIRECTOR_MODEL,
    client: OpenAI,
) -> tuple[list[ParsedTurn], list[dict[str, Any]]]:
    """Same opening lines in every form, back to back, as one short Fish script."""
    sample = bakeoff_slice(turns, n)
    sample_cues = cues[: len(sample)]
    all_turns: list[ParsedTurn] = []
    all_cues: list[dict[str, Any]] = []
    for form in FORMS:
        print(f"Hindi writer ({form}, {len(sample)} lines)...")
        hindi = run_hindi_writer(sample, form=form, model=model, client=client)
        for t, c in zip(hindi, hindi_cues(sample_cues, hindi)):
            index = len(all_turns) + 1
            all_turns.append(ParsedTurn(index=index, speaker=t.speaker, text=t.text))
            all_cues.append({**c, "index": index, "form": form})
    return all_turns, all_cues


def bakeoff_key(cues: list[dict[str, Any]]) -> str:
    """Which generated clip belongs to which written form (same grouping as the VM)."""
    form_by_index = {int(c["index"]): str(c["form"]) for c in cues}
    lines = ["# Hindi bake-off key", "", "Clip names are what the Fish notebook writes to `turns/`.", ""]
    current = None
    for u in group_utterances(cues):
        form = form_by_index[int(u["cue_indexes"][0])]
        if form != current:
            lines += ["", f"## {form}", ""]
            current = form
        clip = utterance_clip_name(int(u["utterance_index"]), str(u["speaker"]))
        lines.append(f"- `{clip}` — {' '.join(u['texts'])}")
    return "\n".join(lines) + "\n"
