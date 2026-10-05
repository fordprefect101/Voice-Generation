#!/usr/bin/env python3
"""Hindi writer: English Fish script + cues → Hindi Fish script + cues.

Input is the output of run_fish_director.py. Tags, speakers, sections and
pause_after_ms carry over; only the spoken words change.

Usage:
  # Listening test: first 12 lines in all three written forms, one short script
  python run_hindi_writer.py --bakeoff 12

  # Full episode in the form that won the listening test
  python run_hindi_writer.py --form devanagari

Forms: devanagari (all Devanagari) | mixed (English words in Latin) | roman (typed Hinglish)

The VM notebook picks the first *fish*.txt and *.cues.json in ~/fish-s2/inputs/.
Keep only one script + one cues file there per run, plus the Hindi refs from
voices/refs/hi/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from pipeline.director_common import DEFAULT_DIRECTOR_MODEL, client, save_cues, save_txt
from pipeline.fish_timeline import load_cues
from pipeline.hindi_writer import (
    FORMS,
    bakeoff_key,
    build_bakeoff,
    hindi_cues,
    run_hindi_writer,
)
from pipeline.parse_script import parse_script_file

BAKEOFF_TXT = Path("output/hindi_fish_bakeoff.txt")
# Not .txt: the VM notebook globs *fish*.txt for the script.
BAKEOFF_KEY = Path("output/hindi_bakeoff_key.md")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "script",
        nargs="?",
        default="output/chokhi_dhani_fish.txt",
        help="English Fish script from run_fish_director.py",
    )
    parser.add_argument(
        "--cues",
        default=None,
        help="English cues JSON (default: <script>.cues.json)",
    )
    parser.add_argument("--form", choices=FORMS, help="Written form for the full episode")
    parser.add_argument(
        "--bakeoff",
        type=int,
        metavar="N",
        help="Write the first N lines in every form as one short test script",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Hindi script path (default: <script>_hi.txt)",
    )
    parser.add_argument("--model", default=DEFAULT_DIRECTOR_MODEL)
    args = parser.parse_args()

    if bool(args.form) == bool(args.bakeoff):
        parser.error("Pass exactly one of --form or --bakeoff N")

    script_path = Path(args.script)
    cues_path = Path(args.cues) if args.cues else script_path.with_suffix(".cues.json")
    turns = parse_script_file(script_path)
    if not turns:
        raise SystemExit(f"No dialogue lines found in {script_path}")
    cues = load_cues(cues_path)
    if len(cues) != len(turns):
        raise SystemExit(
            f"{script_path} has {len(turns)} lines but {cues_path} has {len(cues)} cues. "
            "Re-run run_fish_director.py so they match."
        )
    print(f"Parsed {len(turns)} English Fish turns from {script_path}")

    if args.bakeoff:
        hindi, out_cues = build_bakeoff(
            turns, cues, n=args.bakeoff, model=args.model, client=client()
        )
        out = Path(args.output) if args.output else BAKEOFF_TXT
        out_cues_path = out.with_suffix(".cues.json")
        save_txt(out, hindi)
        save_cues(out_cues_path, out_cues)
        BAKEOFF_KEY.write_text(bakeoff_key(out_cues), encoding="utf-8")
        print(f"\nWrote {out} ({len(hindi)} lines = {len(hindi) // len(FORMS)} per form)")
        print(f"Wrote {out_cues_path}")
        print(f"Wrote {BAKEOFF_KEY} (which clip is which form)")
    else:
        print(f"Hindi writer ({args.form}, {len(turns)} lines)...")
        hindi = run_hindi_writer(turns, form=args.form, model=args.model, client=client())
        out_cues = hindi_cues(cues, hindi)
        out = (
            Path(args.output)
            if args.output
            else script_path.with_name(f"{script_path.stem}_hi.txt")
        )
        out_cues_path = out.with_suffix(".cues.json")
        save_txt(out, hindi)
        save_cues(out_cues_path, out_cues)
        print(f"\nWrote {out} ({len(hindi)} lines, {args.form})")
        print(f"Wrote {out_cues_path}")

    print("Preview:")
    for t in hindi[:6]:
        print(f"  {t.speaker}: {t.text[:110]}{'...' if len(t.text) > 110 else ''}")
    print(
        "\nSCP to the VM with the Hindi refs. Keep only this script + cues in inputs/:\n"
        f"  gcloud compute scp {out} {out_cues_path} "
        f"voices/refs/hi/male_ref.wav voices/refs/hi/female_ref.wav "
        f"voices/refs/hi/ref_texts.json $VM:~/fish-s2/inputs/ --zone=$ZONE"
    )


if __name__ == "__main__":
    main()
