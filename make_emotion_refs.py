#!/usr/bin/env python3
"""Step 2 — clone emotion TILTS from locked warm identity refs.

Identity (male_ref / female_ref) is already the living default voice.
Emotion banks are optional delivery tilts — same person, different lean.
Keys + sample lines live in config/emotions.yaml (extensible).

Requires:
  voices/refs/male_ref.wav
  voices/refs/female_ref.wav
  voices/refs/ref_texts.json   (identity transcripts)

Uses Qwen Base voice-clone so timbre stays the same person; only delivery changes.

Examples:
  python make_emotion_refs.py
  python make_emotion_refs.py --speaker female
  python make_emotion_refs.py --emotions curious,warm,wonder
  python make_emotion_refs.py --takes 3

Outputs:
  voices/refs/male_curious.wav, male_warm.wav, ...
  voices/refs/female_curious.wav, ...
  voices/refs/ref_texts.json   (emotions.* updated)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from qwen_tts import Qwen3TTSModel

from pipeline.voice_refs import (
    EMOTION_BANK,
    EMOTION_SAMPLE_TEXTS,
    REF_DIR,
    emotion_path,
    identity_path,
    load_ref_texts,
    set_emotion_text,
    speaker_key,
)

load_dotenv()


def get_device_dtype():
    preferred = os.getenv("QWEN_DEVICE", "").lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def _to_wav(w) -> np.ndarray:
    wav = w.detach().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)
    return np.asarray(wav, dtype=np.float32).squeeze()


def generate_emotions(
    speakers: list[str],
    emotions: list[str],
    n_takes: int,
) -> None:
    texts = load_ref_texts()
    for sp in speakers:
        sk = speaker_key(sp)
        id_path = identity_path(sk)
        if not id_path.exists():
            raise SystemExit(
                f"Missing {id_path}. Run: python make_voices_refs.py --speaker {sk}"
            )
        if sk not in texts or not texts[sk]:
            raise SystemExit(
                f"Missing identity transcript for {sk} in {REF_DIR / 'ref_texts.json'}"
            )

    device, dtype = get_device_dtype()
    model_id = os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    print(f"Loading Base clone model {model_id} on {device} ({dtype})...")
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )

    texts = load_ref_texts()
    prompts = {}
    for sp in speakers:
        sk = speaker_key(sp)
        print(f"Building identity prompt for {sk} from {identity_path(sk).name}...")
        prompts[sk] = model.create_voice_clone_prompt(
            ref_audio=str(identity_path(sk)),
            ref_text=texts[sk],
        )

    TEMP = {
        "curious": 0.9,
        "playful": 0.95,
        "excited": 0.95,
        "warm": 0.88,
        "wonder": 0.86,
        "explanatory": 0.85,
    }
    # Unknown future keys from config/emotions.yaml get a mild default temp.

    cand_root = REF_DIR / "emotion_candidates"
    cand_root.mkdir(parents=True, exist_ok=True)

    for sk in [speaker_key(s) for s in speakers]:
        for emotion in emotions:
            sample = EMOTION_SAMPLE_TEXTS[emotion]
            print(f"\n=== {sk} / {emotion} ({n_takes} take(s)) ===")
            print(f"  sample: {sample[:70]}...")
            best_path = None
            best_dur = -1.0
            for i in range(n_takes):
                torch.manual_seed(5000 + hash((sk, emotion, i)) % 100000)
                wavs, sr = model.generate_voice_clone(
                    text=sample,
                    language="English",
                    voice_clone_prompt=prompts[sk],
                    temperature=TEMP.get(emotion, 0.9),
                )
                wav = _to_wav(wavs[0])
                take_path = cand_root / f"{sk}_{emotion}_take{i:02d}.wav"
                sf.write(take_path, wav, sr)
                dur = len(wav) / float(sr)
                print(f"  take {i:02d}: {dur:.1f}s -> {take_path}")
                # Prefer plausible short refs (2–12s); avoid runaway.
                if 1.5 <= dur <= 14 and dur > best_dur:
                    best_dur = dur
                    best_path = take_path
            if best_path is None:
                best_path = cand_root / f"{sk}_{emotion}_take00.wav"
            out = emotion_path(sk, emotion)
            out.write_bytes(best_path.read_bytes())
            set_emotion_text(sk, emotion, sample)
            print(f"  -> {out.name} (from {best_path.name})")

    print(f"\nDone. Emotion texts in {REF_DIR / 'ref_texts.json'}")
    print("Upload identity + emotion wavs + ref_texts.json to Colab inputs/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--speaker", choices=["male", "female", "both"], default="both")
    parser.add_argument(
        "--emotions",
        default=",".join(EMOTION_BANK),
        help=f"Comma list (default: {','.join(EMOTION_BANK)})",
    )
    parser.add_argument(
        "--takes",
        type=int,
        default=2,
        help="Takes per emotion; keeps a plausible-duration take (default 2)",
    )
    args = parser.parse_args()

    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    unknown = [e for e in emotions if e not in EMOTION_SAMPLE_TEXTS]
    if unknown:
        raise SystemExit(f"Unknown emotions: {unknown}. Allowed: {list(EMOTION_SAMPLE_TEXTS)}")
    speakers = ["male", "female"] if args.speaker == "both" else [args.speaker]
    generate_emotions(speakers, emotions, args.takes)


if __name__ == "__main__":
    main()
