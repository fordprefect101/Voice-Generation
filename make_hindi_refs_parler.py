#!/usr/bin/env python3
"""Generate Hindi Fish identity refs with Indic Parler-TTS (Mac).

No input reference audio — uses named speakers Rohit (M) and Divya (F).
Outputs Devanagari transcripts locked in ref_texts.json for Fish --prompt-text.

Examples:
  python make_hindi_refs_parler.py
  python make_hindi_refs_parler.py --speaker female --takes 6
  python make_hindi_refs_parler.py --list
  python make_hindi_refs_parler.py --pick male=2

Then SCP to Fish VM:
  voices/refs/hi/male_ref.wav female_ref.wav ref_texts.json
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

from pipeline.hindi_voice_refs import (
    FILENAMES,
    HI_CAND_DIR,
    HI_REF_DIR,
    dynamics_score,
    get_device,
    load_scores,
    parler_description,
    promote_take,
    ref_sample,
    save_ref_texts,
    save_scores,
)

PARLER_MODEL = "ai4bharat/indic-parler-tts"
BACKEND = "parler"


def _to_numpy(wav) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).squeeze()


def generate_parler(
    model: ParlerTTSForConditionalGeneration,
    tokenizer,
    description_tokenizer,
    *,
    text: str,
    description: str,
    device: str,
    seed: int,
) -> tuple[np.ndarray, int]:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    desc_ids = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_ids = tokenizer(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        audio = model.generate(
            input_ids=desc_ids.input_ids,
            attention_mask=desc_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
        )
    return _to_numpy(audio), int(model.config.sampling_rate)


def generate(speakers: list[str], n_takes: int) -> None:
    device = get_device()
    print(f"Loading {PARLER_MODEL} on {device}...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(PARLER_MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(PARLER_MODEL)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    HI_REF_DIR.mkdir(parents=True, exist_ok=True)
    HI_CAND_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()

    for key in speakers:
        text = ref_sample(key)
        description = parler_description(key)
        if not text or not description:
            raise SystemExit(f"Missing ref_sample or parler_description for {key} in config/hindi_hosts.yaml")
        print(f"\n=== {key} ({description[:60]}...) ===")
        print(f"Text: {text[:120]}{'...' if len(text) > 120 else ''}")
        prefer = "female_low_mid" if key == "female" else "male_adult"
        scored: list[tuple[float, Path, int]] = []
        base_seed = 5000 if key == "male" else 6000
        for i in range(n_takes):
            wav, sr = generate_parler(
                model,
                tokenizer,
                description_tokenizer,
                text=text,
                description=description,
                device=device,
                seed=base_seed + i,
            )
            path = HI_CAND_DIR / f"{BACKEND}_{key}_take{i:02d}.wav"
            sf.write(path, wav, sr)
            score, mean_f0 = dynamics_score(wav, sr, prefer_pitch=prefer)
            scored.append((score, path, i))
            f0_note = f", mean_f0={mean_f0:.0f}Hz" if mean_f0 else ""
            print(f"  take {i:02d}: score={score:.3f}{f0_note} ({len(wav)/sr:.1f}s) -> {path.name}")

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
        model=PARLER_MODEL,
    )
    print(f"\nWrote {HI_REF_DIR / 'ref_texts.json'}")
    print("Fish VM: gcloud compute scp voices/refs/hi/*.wav voices/refs/hi/ref_texts.json $VM:~/fish-s2/inputs/ --zone=$ZONE")


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
    parser.add_argument("--takes", type=int, default=6, help="Takes per speaker (default 6)")
    parser.add_argument(
        "--pick",
        nargs="+",
        metavar="SPEAKER=N",
        help="Promote candidate without regenerating, e.g. --pick female=2",
    )
    parser.add_argument("--list", action="store_true", help="List candidate scores")
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
            model=PARLER_MODEL,
        )
        return

    speakers = ["male", "female"] if args.speaker == "both" else [args.speaker]
    generate(speakers, args.takes)


if __name__ == "__main__":
    main()
