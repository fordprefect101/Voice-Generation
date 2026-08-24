#!/usr/bin/env python3
"""Phase A: raw script -> actable lines.json (Writer + Prosody + Emphasis).

Usage:
  export OPENAI_API_KEY=sk-...   # or put it in .env
  python run_script_agents.py chokhi_dhani.txt
  python run_script_agents.py chokhi_dhani.txt -o output/lines.json
  python run_script_agents.py chokhi_dhani.txt --model gpt-4o-mini

Host personalities: config/hosts.yaml
Default model: gpt-4o
emphasize[] → CosyVoice word-stress instruct

Upload lines.json + identity refs to Colab (Phase B).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from pipeline.models import save_lines_json
from pipeline.parse_script import parse_script_file
from pipeline.script_agents import DEFAULT_SCRIPT_MODEL, run_script_agents


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", help="Raw dialogue .txt (Speaker 1:/Speaker 2: or [M]/[F])")
    parser.add_argument(
        "-o",
        "--output",
        default="output/lines.json",
        help="Output lines.json path (default: output/lines.json)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_SCRIPT_MODEL,
        help=f"OpenAI model for all agents (default: {DEFAULT_SCRIPT_MODEL})",
    )
    args = parser.parse_args()

    script_path = Path(args.script)
    turns = parse_script_file(script_path)
    if not turns:
        raise SystemExit(f"No dialogue lines found in {script_path}")

    print(f"Parsed {len(turns)} turns from {script_path}")
    lines = run_script_agents(turns, model=args.model)
    from pipeline.pronounce import apply_pronunciations

    for L in lines:
        L.text = apply_pronunciations(L.text)
    out = Path(args.output)
    save_lines_json(out, lines)
    print(f"\nWrote {out} ({len(lines)} lines)")
    print("Preview:")
    for L in lines[:3]:
        emph = f" emph={L.emphasize}" if L.emphasize else ""
        print(f"  [{L.index}] {L.speaker} ({L.emotion}) after={L.pause_after_ms}{emph}")
        print(f"      {L.text[:100]}...")
    print("\nPhase A done. Next (Phase B): cosyvoice_dialogue_colab.ipynb + identity refs.")


if __name__ == "__main__":
    main()
