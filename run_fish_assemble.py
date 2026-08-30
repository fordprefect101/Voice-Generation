#!/usr/bin/env python3
"""Assemble Fish utterance clips with deterministic turn gaps (+ optional beds).

Expects clips named utt_001_M.wav … from the Fish VM notebook, and cues JSON
from run_fish_director.py (with pause_after_ms).

Usage:
  python run_fish_assemble.py \\
    --clips outputs/turns \\
    --cues output/chokhi_dhani_fish.cues.json \\
    -o output/episode_fish.wav

  # Or rebuild gaps on Mac after pulling the turns/ folder from the VM:
  gcloud compute scp --recurse $VM:~/fish-s2/outputs/turns ./fish_turns --zone=$ZONE
  python run_fish_assemble.py --clips fish_turns --cues output/chokhi_dhani_fish.cues.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.fish_timeline import (
    assemble_utterance_clips,
    group_utterances,
    load_cues,
    save_utterances,
)
from pipeline.stitch import normalize_loudness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips",
        required=True,
        help="Directory of utt_XXX_M/F.wav from Fish per-utterance generate",
    )
    parser.add_argument(
        "--cues",
        required=True,
        help="Fish cues JSON (pause_after_ms / tagged_text)",
    )
    parser.add_argument(
        "--utterances",
        default=None,
        help="Optional precomputed utterances JSON (skips regroup from cues)",
    )
    parser.add_argument("-o", "--output", default="output/episode_fish.wav")
    parser.add_argument("--no-beds", action="store_true")
    parser.add_argument(
        "--save-utterances",
        default=None,
        help="Write the utterance plan JSON (for debugging / VM sync)",
    )
    args = parser.parse_args()

    if args.utterances:
        utterances = json.loads(Path(args.utterances).read_text(encoding="utf-8"))
    else:
        cues = load_cues(args.cues)
        utterances = group_utterances(cues)

    if args.save_utterances:
        save_utterances(args.save_utterances, utterances)
        print(f"Wrote utterance plan: {args.save_utterances} ({len(utterances)} utts)")

    gaps = [int(u.get("pause_after_ms") or 0) for u in utterances]
    print(
        f"Assembling {len(utterances)} utterances; "
        f"pause_after_ms min/avg/max = "
        f"{min(gaps) if gaps else 0}/"
        f"{(sum(gaps) // len(gaps)) if gaps else 0}/"
        f"{max(gaps) if gaps else 0}"
    )

    episode = assemble_utterance_clips(
        utterances,
        args.clips,
        with_beds=not args.no_beds,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".wav":
        episode.export(out, format="wav")
        print(f"Wrote {out} ({len(episode) / 60000:.1f} min)")
    else:
        raw = out.with_suffix(".raw.wav")
        episode.export(raw, format="wav")
        normalize_loudness(raw, out)
        raw.unlink(missing_ok=True)
        print(f"Wrote {out} ({len(episode) / 60000:.1f} min)")


if __name__ == "__main__":
    main()
