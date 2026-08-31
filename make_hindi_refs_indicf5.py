#!/usr/bin/env python3
"""Generate Hindi Fish identity refs with IndicF5 (Mac).

Uses bundled AI4Bharat seed prompt wavs (downloaded on first run) + Devanagari
identity scripts from config/hindi_hosts.yaml. Compare with Parler refs via A/B on Fish.

Examples:
  python make_hindi_refs_indicf5.py
  python make_hindi_refs_indicf5.py --speaker male --takes 4
  python make_hindi_refs_indicf5.py --list
  python make_hindi_refs_indicf5.py --pick female=1

Requires: pip install git+https://github.com/AI4Bharat/IndicF5.git
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

from pipeline.hindi_voice_refs import (
    FILENAMES,
    HI_CAND_DIR,
    HI_REF_DIR,
    dynamics_score,
    ensure_indicf5_prompt,
    get_device,
    load_scores,
    promote_take,
    ref_sample,
    save_ref_texts,
    save_scores,
)

INDICF5_MODEL = "ai4bharat/IndicF5"
BACKEND = "indicf5"
OUTPUT_SR = 24000


def _normalize_audio(audio) -> np.ndarray:
    wav = np.asarray(audio, dtype=np.float32).squeeze()
    if wav.dtype == np.int16 or (wav.size and np.max(np.abs(wav)) > 1.5):
        wav = wav.astype(np.float32)
        if np.max(np.abs(wav)) > 1.0:
            wav = wav / 32768.0
    return wav


def synthesize(
    model,
    text: str,
    ref_audio_path: Path,
    ref_text: str,
    *,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    audio = model(str(text), ref_audio_path=str(ref_audio_path), ref_text=str(ref_text))
    return _normalize_audio(audio)


def generate(speakers: list[str], n_takes: int) -> None:
    device = get_device()
    print(f"Loading {INDICF5_MODEL} (trust_remote_code) on {device}...")
    model = AutoModel.from_pretrained(INDICF5_MODEL, trust_remote_code=True)
    if hasattr(model, "to"):
        model = model.to(device)

    HI_REF_DIR.mkdir(parents=True, exist_ok=True)
    HI_CAND_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()

    for key in speakers:
        text = ref_sample(key)
        prompt_path, prompt_text = ensure_indicf5_prompt(key)
        if not text:
            raise SystemExit(f"Missing ref_sample for {key} in config/hindi_hosts.yaml")
        print(f"\n=== {key} seed={prompt_path.name} ===")
        print(f"Synth text: {text[:120]}{'...' if len(text) > 120 else ''}")
        prefer = "female_low_mid" if key == "female" else "male_adult"
        scored: list[tuple[float, Path, int]] = []
        base_seed = 7000 if key == "male" else 8000
        for i in range(n_takes):
            wav = synthesize(
                model,
                text,
                prompt_path,
                prompt_text,
                seed=base_seed + i,
            )
            path = HI_CAND_DIR / f"{BACKEND}_{key}_take{i:02d}.wav"
            sf.write(path, wav, OUTPUT_SR)
            score, mean_f0 = dynamics_score(wav, OUTPUT_SR, prefer_pitch=prefer)
            scored.append((score, path, i))
            f0_note = f", mean_f0={mean_f0:.0f}Hz" if mean_f0 else ""
            print(f"  take {i:02d}: score={score:.3f}{f0_note} ({len(wav)/OUTPUT_SR:.1f}s) -> {path.name}")

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_path, best_i = scored[0]
        out = HI_REF_DIR / FILENAMES[key]
        shutil.copy(best_path, out)
        scores[f"{BACKEND}_{key}"] = [
            {
                "take": i,
                "score": float(s),
                "path": str(p),
                "auto_best": i == best_i,
            }
            for s, p, i in scored
        ]
        print(f"  AUTO-BEST take {best_i:02d} (score={best_score:.3f}) -> {out}")

    save_scores(scores)
    save_ref_texts(
        ref_sample("male"),
        ref_sample("female"),
        backend=BACKEND,
        model=INDICF5_MODEL,
    )
    print(f"\nWrote {HI_REF_DIR / 'ref_texts.json'}")
    print("Compare with Parler refs on Fish, then SCP the winner to ~/fish-s2/inputs/")


def list_candidates() -> None:
    scores = load_scores()
    if not scores:
        print("No scores yet. Run generation first.")
        return
    for label, rows in scores.items():
        print(f"\n=== {label} ===")
        for row in sorted(rows, key=lambda r: r["score"], reverse=True):
            mark = " <-- auto-best" if row.get("auto_best") else ""
            print(f"  take {row['take']:02d}: {row['score']:.3f}  {row['path']}{mark}")


def parse_pick(items: list[str]) -> list[tuple[str, int]]:
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Bad --pick '{item}', expected speaker=N")
        speaker, take_s = item.split("=", 1)
        speaker = speaker.strip().lower()
        if speaker not in FILENAMES:
            raise SystemExit(f"Unknown speaker '{speaker}'")
        out.append((speaker, int(take_s)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--speaker", choices=["male", "female", "both"], default="both")
    parser.add_argument("--takes", type=int, default=4, help="Takes per speaker (default 4)")
    parser.add_argument(
        "--pick",
        nargs="+",
        metavar="SPEAKER=N",
        help="Promote indicf5 candidate, e.g. --pick male=0",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_candidates()
        return
    if args.pick:
        for speaker, take in parse_pick(args.pick):
            dst = promote_take(speaker, take, backend=BACKEND)
            print(f"Promoted -> {dst}")
        save_ref_texts(
            ref_sample("male"),
            ref_sample("female"),
            backend=BACKEND,
            model=INDICF5_MODEL,
        )
        return

    speakers = ["male", "female"] if args.speaker == "both" else [args.speaker]
    generate(speakers, args.takes)


if __name__ == "__main__":
    main()
