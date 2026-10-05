#!/usr/bin/env python3
"""Fish S2 travel pipeline: Fish-shaped editor → writer → pronunciations → .txt + cues.

Input is the raw dialogue (e.g. chokhi_dhani.txt).
The writer curates wording for Fish S2, then places inline [tags].
Cues include pause_after_ms for per-utterance assemble (see run_fish_assemble.py).

Usage:
  python run_fish_director.py chokhi_dhani.txt
  python run_fish_director.py chokhi_dhani.txt -o output/chokhi_dhani_fish.txt

SCP the .txt AND .cues.json to the VM. Notebook generates one wav per utterance,
then inserts pause_after_ms gaps. On Mac, mix beds with --no-holds.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline.fish_director import run_fish_travel_pipeline
from pipeline.director_common import DEFAULT_DIRECTOR_MODEL, save_cues, save_txt
from pipeline.parse_script import parse_script_file


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", help="Raw dialogue .txt")
    parser.add_argument(
        "-o",
        "--output",
        default="output/chokhi_dhani_fish.txt",
        help="Fish-curated dialogue .txt",
    )
    parser.add_argument(
        "--cues",
        default=None,
        help="Cues JSON path (default: <output>.cues.json)",
    )
    parser.add_argument("--model", default=DEFAULT_DIRECTOR_MODEL)
    parser.add_argument(
        "--style",
        choices=("speaker", "tag"),
        default="speaker",
        help="Export as 'Speaker N:' or '[M]/[F]'",
    )
    parser.add_argument(
        "--skip-editor",
        action="store_true",
        help="Skip Fish editor; writer still reshapes + tags",
    )
    parser.add_argument(
        "--llm-pronounce",
        action="store_true",
        help="Ask LLM to refresh speak-forms",
    )
    args = parser.parse_args()

    if args.llm_pronounce:
        os.environ["PRONUNCIATION_LLM"] = "1"

    script_path = Path(args.script)
    turns = parse_script_file(script_path)
    if not turns:
        raise SystemExit(f"No dialogue lines found in {script_path}")

    print(f"Parsed {len(turns)} source turns from {script_path}")
    directed, cues = run_fish_travel_pipeline(
        turns, model=args.model, skip_editor=args.skip_editor
    )
    out = Path(args.output)
    cues_path = Path(args.cues) if args.cues else out.with_suffix(".cues.json")
    save_txt(out, directed, style=args.style)
    save_cues(cues_path, cues)

    tagged = sum(1 for t in directed if "[" in t.text)
    holds = sum(1 for c in cues if c.get("hold_after_ms"))
    pauses = [int(c.get("pause_after_ms") or 0) for c in cues]
    print(f"\nWrote {out} ({len(directed)} turns; {tagged} with tags)")
    print(f"Wrote {cues_path} ({holds} observation holds)")
    if pauses:
        print(
            f"  pause_after_ms min/avg/max = "
            f"{min(pauses)}/{sum(pauses)//len(pauses)}/{max(pauses)}"
        )
    print("Preview:")
    for t, c in list(zip(directed, cues))[:8]:
        label = f"Speaker {_num(t.speaker)}" if args.style == "speaker" else f"[{t.speaker}]"
        print(f"  [{c['section']}] {label}: {t.text[:110]}{'...' if len(t.text) > 110 else ''}")
    if len(directed) > 8:
        print(f"  ... ({len(directed) - 8} more)")
    print(
        "\nSCP .txt + .cues.json to VM, generate per-utterance, then on Mac:\n"
        f"  gcloud compute scp {out} {cues_path} "
        f"voices/refs/*.wav voices/refs/ref_texts.json "
        f"$VM:~/fish-s2/inputs/ --zone=$ZONE\n"
        f"  # after VM assemble:\n"
        f"  python run_mix.py --episode episode.wav --cues {cues_path} --no-holds\n"
        f"  # or re-gap from clips:\n"
        f"  python run_fish_assemble.py --clips fish_turns --cues {cues_path}"
    )


def _num(speaker: str) -> str:
    return "1" if speaker == "M" else "2"


if __name__ == "__main__":
    main()
