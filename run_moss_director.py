#!/usr/bin/env python3
"""MOSS travel pipeline: editor → director → pronunciations → .txt + cues.json

Usage:
  python run_moss_director.py chokhi_dhani.txt
  python run_moss_director.py chokhi_dhani.txt -o output/chokhi_dhani_moss.txt
  python run_moss_director.py chokhi_dhani.txt --llm-pronounce

After hosts.yaml ref_sample changes, regenerate refs:
  python make_voices_refs.py --speaker female --takes 12
  python make_voices_refs.py --speaker male --takes 8

Upload .txt + refs to moss_ttsd_colab.ipynb (MOSS-TTSD v1.0 on Colab GPU), then:
  python run_mix_moss.py --episode episode.wav --cues output/chokhi_dhani_moss.cues.json
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from pipeline.moss_director import (
    DEFAULT_MOSS_DIRECTOR_MODEL,
    run_moss_travel_pipeline,
    save_moss_cues,
    save_moss_txt,
)
from pipeline.parse_script import parse_script_file


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", help="Raw dialogue .txt (Speaker 1:/2: or [M]/[F])")
    parser.add_argument(
        "-o",
        "--output",
        default="output/chokhi_dhani_moss.txt",
        help="MOSS-ready dialogue .txt",
    )
    parser.add_argument(
        "--cues",
        default=None,
        help="Cues JSON path (default: <output>.cues.json)",
    )
    parser.add_argument("--model", default=DEFAULT_MOSS_DIRECTOR_MODEL)
    parser.add_argument(
        "--style",
        choices=("speaker", "tag"),
        default="speaker",
        help="Export as 'Speaker N:' or '[M]/[F]'",
    )
    parser.add_argument(
        "--skip-editor",
        action="store_true",
        help="Only run MOSS director + pronunciations (no travel editor)",
    )
    parser.add_argument(
        "--llm-pronounce",
        action="store_true",
        help="Ask LLM to refresh speak-forms (future pronunciation writer hook)",
    )
    args = parser.parse_args()

    if args.llm_pronounce:
        os.environ["PRONUNCIATION_LLM"] = "1"

    script_path = Path(args.script)
    turns = parse_script_file(script_path)
    if not turns:
        raise SystemExit(f"No dialogue lines found in {script_path}")

    print(f"Parsed {len(turns)} source turns from {script_path}")
    directed, cues = run_moss_travel_pipeline(
        turns, model=args.model, skip_editor=args.skip_editor
    )
    out = Path(args.output)
    cues_path = Path(args.cues) if args.cues else out.with_suffix(".cues.json")
    save_moss_txt(out, directed, style=args.style)
    save_moss_cues(cues_path, cues)

    holds = sum(1 for c in cues if c.get("hold_after_ms"))
    print(f"\nWrote {out} ({len(directed)} turns; was {len(turns)})")
    print(f"Wrote {cues_path} ({holds} observation holds)")
    print("Preview:")
    for t, c in list(zip(directed, cues))[:6]:
        label = f"Speaker {_num(t.speaker)}" if args.style == "speaker" else f"[{t.speaker}]"
        hold = f" hold={c['hold_after_ms']}" if c.get("hold_after_ms") else ""
        print(
            f"  [{c['section']}] {label}: {t.text[:100]}"
            f"{'...' if len(t.text) > 100 else ''}{hold}"
        )
    if len(directed) > 6:
        print(f"  ... ({len(directed) - 6} more)")
    print(
        "\nNext: upload .txt + refs to moss_ttsd_colab.ipynb (v1.0 GPU),\n"
        "then mix on Mac:\n"
        f"  python run_mix_moss.py --episode episode.wav --cues {cues_path}"
    )


def _num(speaker: str) -> str:
    return "1" if speaker == "M" else "2"


if __name__ == "__main__":
    main()
