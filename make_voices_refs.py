#!/usr/bin/env python3
"""Step 1 — generate IDENTITY voice refs (VoiceDesign).

Identity = living warm default voice (code key "neutral"), NOT emotionless.
Lock who M/F are first. Then run make_emotion_refs.py to clone emotion tilts
from these files (same timbre, different delivery).

Examples:
  python make_voices_refs.py                      # both, 8 takes each
  python make_voices_refs.py --speaker female --takes 12
  python make_voices_refs.py --pick female=3
  python make_voices_refs.py --list

Outputs:
  voices/refs/male_ref.wav
  voices/refs/female_ref.wav
  voices/refs/ref_texts.json   (identity texts; emotion keys preserved if present)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from qwen_tts import Qwen3TTSModel

from pipeline.voice_refs import NEUTRAL_REF_TEXTS, REF_DIR, merge_identity_texts

CAND_DIR = REF_DIR / "candidates"
SCORES_PATH = CAND_DIR / "scores.json"

REF_TEXTS = NEUTRAL_REF_TEXTS

FILENAMES = {"male": "male_ref.wav", "female": "female_ref.wav"}


def get_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def dynamics_score(
    wav: np.ndarray,
    sr: int,
    *,
    prefer_pitch: str | None = None,
) -> tuple[float, float | None]:
    f0, voiced, _ = librosa.pyin(wav, fmin=60, fmax=450, sr=sr)
    f0 = f0[np.asarray(voiced, dtype=bool)]
    f0 = f0[~np.isnan(f0)]
    pitch_var = float(np.std(np.log(f0))) if len(f0) > 10 else 0.0
    rms = librosa.feature.rms(y=wav)[0]
    energy_var = float(np.std(rms) / (np.mean(rms) + 1e-8))
    # Prefer some movement (not dead) but not manic — soft-cap high dynamics.
    score = pitch_var + energy_var
    if energy_var > 0.55:
        score -= (energy_var - 0.55) * 1.5
    mean_f0 = float(np.mean(f0)) if len(f0) > 10 else None
    if prefer_pitch == "female_low_mid" and mean_f0 is not None:
        if mean_f0 > 220:
            score -= (mean_f0 - 220) / 80.0
        elif mean_f0 < 140:
            score -= (140 - mean_f0) / 140.0
    elif prefer_pitch == "male_adult" and mean_f0 is not None:
        if mean_f0 > 180:
            score -= (mean_f0 - 180) / 80.0
        elif mean_f0 < 90:
            score -= (90 - mean_f0) / 90.0
    return score, mean_f0


def write_ref_texts() -> None:
    merge_identity_texts(REF_TEXTS["male"], REF_TEXTS["female"])


def load_scores() -> dict:
    if SCORES_PATH.exists():
        return json.loads(SCORES_PATH.read_text(encoding="utf-8"))
    return {}


def save_scores(scores: dict) -> None:
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")


def list_candidates() -> None:
    scores = load_scores()
    if not scores:
        print("No scores yet. Run generation first.")
        return
    for speaker, rows in scores.items():
        print(f"\n=== {speaker} ===")
        for row in sorted(rows, key=lambda r: r["score"], reverse=True):
            mark = " <-- current best auto-pick" if row.get("auto_best") else ""
            print(f"  take {row['take']:02d}: dynamics={row['score']:.3f}  {row['path']}{mark}")
    print(f"\nPromote with: python make_voices_refs.py --pick female=3")


def pick_take(speaker: str, take: int) -> None:
    src = CAND_DIR / f"{speaker}_take{take:02d}.wav"
    if not src.exists():
        raise SystemExit(f"Missing {src}. Generate takes first.")
    dst = REF_DIR / FILENAMES[speaker]
    REF_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    write_ref_texts()
    print(f"Promoted {src.name} -> {dst}")
    print(f"Transcript locked in {REF_DIR / 'ref_texts.json'}")
    print("Next: python make_emotion_refs.py")


def generate(speakers: list[str], n_takes: int) -> None:
    device, dtype = get_device_dtype()
    print(f"Loading VoiceDesign on {device} ({dtype})...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    voices = yaml.safe_load(Path("config/voices.yaml").read_text(encoding="utf-8"))
    REF_DIR.mkdir(parents=True, exist_ok=True)
    CAND_DIR.mkdir(parents=True, exist_ok=True)

    scores = load_scores()
    for key in speakers:
        try:
            from pipeline.hosts import host_qwen_instruct

            instruct = host_qwen_instruct("M" if key == "male" else "F") or voices[key][
                "qwen_instruct"
            ].strip()
        except Exception:
            instruct = voices[key]["qwen_instruct"].strip()
        text = REF_TEXTS[key]
        print(f"\n=== {key} IDENTITY (VoiceDesign): {n_takes} takes ===")
        print(f"Instruct ({len(instruct.split())} words): {instruct}")
        scored: list[tuple[float, Path, int]] = []
        for i in range(n_takes):
            torch.manual_seed(3000 + (0 if key == "male" else 100) + i)
            wavs, sr = model.generate_voice_design(
                text=text, language="English", instruct=instruct
            )
            w = wavs[0]
            wav = w.detach().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)
            wav = np.asarray(wav, dtype=np.float32).squeeze()
            path = CAND_DIR / f"{key}_take{i:02d}.wav"
            sf.write(path, wav, sr)
            prefer = "female_low_mid" if key == "female" else "male_adult"
            score, mean_f0 = dynamics_score(wav, sr, prefer_pitch=prefer)
            scored.append((score, path, i))
            f0_note = f", mean_f0={mean_f0:.0f}Hz" if mean_f0 is not None else ""
            print(f"  take {i:02d}: score={score:.3f}{f0_note} ({len(wav) / sr:.1f}s) -> {path}")

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_path, best_i = scored[0]
        out = REF_DIR / FILENAMES[key]
        shutil.copy(best_path, out)
        scores[key] = [
            {
                "take": i,
                "score": float(s),
                "path": str(p),
                "auto_best": i == best_i,
            }
            for s, p, i in scored
        ]
        print(f"  AUTO-BEST take {best_i:02d} (dynamics={best_score:.3f}) -> {out}")
        print(f"  To choose another: python make_voices_refs.py --pick {key}={best_i}")

    save_scores(scores)
    write_ref_texts()
    print(f"\nWrote {REF_DIR / 'ref_texts.json'}")
    print("When identity sounds right:")
    print("  python make_emotion_refs.py")
    print("Upload later: male_ref.wav female_ref.wav male_*.wav female_*.wav ref_texts.json")


def parse_pick(items: list[str]) -> list[tuple[str, int]]:
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Bad --pick '{item}', expected speaker=N e.g. female=3")
        speaker, take_s = item.split("=", 1)
        speaker = speaker.strip().lower()
        if speaker not in FILENAMES:
            raise SystemExit(f"Unknown speaker '{speaker}'")
        out.append((speaker, int(take_s)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speaker", choices=["male", "female", "both"], default="both")
    parser.add_argument("--takes", type=int, default=8, help="Takes per speaker (default 8)")
    parser.add_argument(
        "--pick",
        nargs="+",
        metavar="SPEAKER=N",
        help="Promote candidate take(s) without regenerating, e.g. --pick female=3",
    )
    parser.add_argument("--list", action="store_true", help="List last candidate scores")
    args = parser.parse_args()

    if args.list:
        list_candidates()
        return
    if args.pick:
        for speaker, take in parse_pick(args.pick):
            pick_take(speaker, take)
        return

    speakers = ["male", "female"] if args.speaker == "both" else [args.speaker]
    generate(speakers, args.takes)


if __name__ == "__main__":
    main()
