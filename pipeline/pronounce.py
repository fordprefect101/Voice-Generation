"""Pronunciation speak-forms for TTS (written into script text).

Today: lexicon in config/pronunciations.yaml.
Future: LLM pronunciation writer fills/overrides speak-forms; keep using
apply_pronunciations() / apply_pronunciations_to_turns() as the single apply step.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path("config/pronunciations.yaml")


@lru_cache(maxsize=1)
def load_pronunciations(path: str | None = None) -> list[dict[str, str]]:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    items = data.get("replacements") or []
    out: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        frm = str(item.get("from") or "").strip()
        to = str(item.get("to") or "").strip()
        if frm and to:
            out.append({"from": frm, "to": to})
    out.sort(key=lambda x: len(x["from"]), reverse=True)
    return out


def reload_pronunciations() -> None:
    """Call after editing pronunciations.yaml or after an LLM writer updates it."""
    load_pronunciations.cache_clear()


def lexicon_map(path: str | None = None) -> dict[str, str]:
    return {item["from"]: item["to"] for item in load_pronunciations(path)}


_FISH_TAG = re.compile(r"\[[^\[\]]*\]")


def _apply_lexicon_span(text: str, *, path: str | None = None) -> str:
    result = text
    for item in load_pronunciations(path):
        frm, to = item["from"], item["to"]
        if frm.lower() == to.lower() and frm == to:
            continue
        pattern = re.compile(re.escape(frm), re.IGNORECASE)

        def _sub(_m: re.Match[str], replacement: str = to) -> str:
            return replacement

        result = pattern.sub(_sub, result)
    return result


def apply_pronunciations(text: str, *, path: str | None = None) -> str:
    """Replace cultural terms with speak-forms (case-insensitive) in script text.

    Leaves Fish [tags] untouched so [emphasis] / [short pause] stay intact.
    """
    if "[" not in text:
        return _apply_lexicon_span(text, path=path)
    pieces: list[str] = []
    last = 0
    for m in _FISH_TAG.finditer(text):
        pieces.append(_apply_lexicon_span(text[last : m.start()], path=path))
        pieces.append(m.group(0))
        last = m.end()
    pieces.append(_apply_lexicon_span(text[last:], path=path))
    return "".join(pieces)


def apply_pronunciations_to_turns(turns: list[Any], *, path: str | None = None) -> list[Any]:
    from pipeline.models import ParsedTurn

    out: list[ParsedTurn] = []
    for i, t in enumerate(turns, start=1):
        text = apply_pronunciations(str(t.text), path=path)
        out.append(
            ParsedTurn(
                index=getattr(t, "index", i),
                speaker=t.speaker,
                text=text,
            )
        )
    return out


def extract_candidate_terms(texts: list[str], *, path: str | None = None) -> list[str]:
    """Terms from the lexicon that appear in texts (for future LLM review)."""
    found: list[str] = []
    blob = "\n".join(texts)
    lower = blob.lower()
    for item in load_pronunciations(path):
        frm = item["from"]
        if frm.lower() in lower and frm not in found:
            found.append(frm)
    return found


def resolve_speak_forms(
    terms: list[str],
    *,
    use_llm: bool = False,
    model: str | None = None,
    path: str | None = None,
) -> dict[str, str]:
    """Map display terms → speak-forms.

    Current: lexicon only.
    Future: if use_llm=True, an LLM proposes spellings for unknown/override terms,
    then merge into the returned map (and optionally write back to yaml).
    """
    lex = lexicon_map(path)
    # Case-insensitive lexicon lookup
    lower_lex = {k.lower(): v for k, v in lex.items()}
    out: dict[str, str] = {}
    missing: list[str] = []
    for term in terms:
        t = term.strip()
        if not t:
            continue
        if t in lex:
            out[t] = lex[t]
        elif t.lower() in lower_lex:
            out[t] = lower_lex[t.lower()]
        else:
            missing.append(t)
            out[t] = t

    if use_llm and missing:
        out.update(_llm_propose_speak_forms(missing, model=model))
    return out


def _llm_propose_speak_forms(
    terms: list[str],
    *,
    model: str | None = None,
) -> dict[str, str]:
    """Future pronunciation writer.

    Asks an LLM for Latin speak-forms optimized for English TTS (no hyphens).
    Requires OPENAI_API_KEY. Safe no-op fallback: identity map.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        print("pronounce LLM: OPENAI_API_KEY missing — leaving terms unchanged")
        return {t: t for t in terms}

    try:
        from openai import OpenAI
    except ImportError:
        return {t: t for t in terms}

    client = OpenAI()
    model = model or os.getenv("PRONUNCIATION_MODEL", "gpt-4o")
    system = """You write Latin-alphabet speak-forms for Indian/Rajasthani terms
so English TTS (MOSS/CosyVoice) pronounces them closer to local speech.

Rules:
- Output ONLY JSON: {"replacements":[{"from":"...","to":"..."}, ...]}
- Prefer spellings without hyphens (TTS normalizers strip "-").
- Keep meaning; do not translate into English words.
- Optimize for Indian-English TTS reading, not academic IPA.
"""
    user = "Propose speak-forms for:\n" + json.dumps(terms, ensure_ascii=False)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    data = json.loads(content)
    items = data.get("replacements") if isinstance(data, dict) else data
    out: dict[str, str] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            frm = str(item.get("from") or "").strip()
            to = str(item.get("to") or "").strip()
            if frm and to:
                out[frm] = to
    # Ensure every requested term has an entry
    for t in terms:
        out.setdefault(t, t)
    return out


def merge_speak_forms_into_yaml(
    updates: dict[str, str],
    *,
    path: str | Path | None = None,
) -> Path:
    """Merge LLM/manual speak-forms into pronunciations.yaml and reload cache."""
    p = Path(path) if path else CONFIG_PATH
    data: dict[str, Any] = {}
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    items = list(data.get("replacements") or [])
    by_from = {
        str(it.get("from")).strip(): it
        for it in items
        if isinstance(it, dict) and it.get("from")
    }
    for frm, to in updates.items():
        frm, to = frm.strip(), to.strip()
        if not frm or not to:
            continue
        if frm in by_from:
            by_from[frm]["to"] = to
        else:
            by_from[frm] = {"from": frm, "to": to}
    data["replacements"] = sorted(
        by_from.values(),
        key=lambda x: len(str(x.get("from") or "")),
        reverse=True,
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    reload_pronunciations()
    return p
