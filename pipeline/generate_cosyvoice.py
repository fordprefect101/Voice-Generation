"""CosyVoice 3 episode TTS: identity ref wav + emotion instruct.

Requires a local CosyVoice checkout + Fun-CosyVoice3 weights.
  ./setup_cosyvoice.sh
  export COSYVOICE_REPO=... COSYVOICE_MODEL_DIR=...

Or use cosyvoice_dialogue_colab.ipynb on a CUDA GPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from pipeline.emotions import (
    build_cosyvoice_instruct,
    cosyvoice_speed,
    normalize_line_emotion,
)
from pipeline.models import EnrichedLine, load_lines_json
from pipeline.voice_refs import REF_DIR, load_ref_texts, resolve_ref

load_dotenv()

_MODEL = None


def _repo_root() -> Path:
    raw = os.getenv("COSYVOICE_REPO", "third_party/CosyVoice")
    return Path(raw).expanduser().resolve()


def _model_dir() -> Path:
    raw = os.getenv(
        "COSYVOICE_MODEL_DIR",
        str(_repo_root() / "pretrained_models" / "Fun-CosyVoice3-0.5B"),
    )
    return Path(raw).expanduser().resolve()


def _ensure_cosyvoice_path() -> Path:
    repo = _repo_root()
    if not repo.is_dir():
        raise FileNotFoundError(
            f"CosyVoice repo not found at {repo}. Run: ./setup_cosyvoice.sh"
        )
    matcha = repo / "third_party" / "Matcha-TTS"
    for p in (repo, matcha):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return repo


def load_cosyvoice_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    _ensure_cosyvoice_path()
    from cosyvoice.cli.cosyvoice import AutoModel

    model_dir = _model_dir()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"CosyVoice model not found at {model_dir}. Run: ./setup_cosyvoice.sh"
        )

    male_ref = REF_DIR / "male_ref.wav"
    female_ref = REF_DIR / "female_ref.wav"
    if not male_ref.exists() or not female_ref.exists():
        raise FileNotFoundError(
            "Missing identity refs. Run: python make_voices_refs.py"
        )

    print(f"Loading CosyVoice from {model_dir}...")
    _MODEL = AutoModel(model_dir=str(model_dir))
    return _MODEL


def _identity_ref(speaker: str) -> tuple[Path, str]:
    texts = load_ref_texts()
    path, text = resolve_ref(speaker, "neutral", ref_texts=texts)
    if not path.exists():
        raise FileNotFoundError(f"Missing identity ref for {speaker}: {path}")
    return path, text


def _save_speech(chunks, sample_rate: int, out_path: Path) -> None:
    import torch
    import torchaudio

    waves = [c["tts_speech"] for c in chunks]
    if not waves:
        raise RuntimeError(f"CosyVoice returned no audio for {out_path.name}")
    audio = torch.cat(waves, dim=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), audio.cpu(), sample_rate)


def generate_line_audio(line: EnrichedLine, out_path: str | Path) -> str:
    out_path = Path(out_path)
    model = load_cosyvoice_model()
    emotion = normalize_line_emotion(line.emotion)
    ref_path, _ref_text = _identity_ref(line.speaker)
    instruct = build_cosyvoice_instruct(
        emotion,
        speaker=line.speaker,
        emphasize=line.emphasize,
    )
    speed = cosyvoice_speed(emotion)
    print(f"  cosy {line.speaker}/{emotion} speed={speed} ref={ref_path.name}")

    chunks = list(
        model.inference_instruct2(
            line.text,
            instruct,
            str(ref_path),
            stream=False,
            speed=speed,
        )
    )
    _save_speech(chunks, model.sample_rate, out_path)
    return str(out_path)


def _filter_lines(lines, only=None, from_index=None, to_index=None):
    if only is not None:
        return [l for l in lines if l.index == only]
    if from_index is not None or to_index is not None:
        start = from_index or 1
        end = to_index or max(l.index for l in lines)
        return [l for l in lines if start <= l.index <= end]
    return lines


def generate_all(
    lines: list[EnrichedLine],
    config_path: str | Path = "config/voices.yaml",
    clips_dir: str | Path = "output/clips",
    only: int | None = None,
    from_index: int | None = None,
    to_index: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    del config_path  # reserved for future voice config hooks
    clips_dir = Path(clips_dir)
    selected = _filter_lines(lines, only, from_index, to_index)
    paths: list[str] = []

    for line in selected:
        out_path = clips_dir / f"{line.index:03d}_{line.speaker}.wav"
        if dry_run or os.getenv("COSYVOICE_SKIP", "0") == "1":
            emotion = normalize_line_emotion(line.emotion)
            from pipeline.pace import target_wpm

            print(f"[dry-run] {out_path.name}")
            print(f"  speaker: {line.speaker} emotion: {emotion}")
            print(f"  instruct: {build_cosyvoice_instruct(emotion, speaker=line.speaker, emphasize=line.emphasize)}")
            if line.emphasize:
                print(f"  emphasize: {line.emphasize}")
            print(f"  target_wpm: {target_wpm(emotion)}")
            print(f"  text: {line.text[:120]}...")
            paths.append(str(out_path))
            continue
        print(f"Generating line {line.index} ({line.speaker}/{line.emotion})...")
        paths.append(generate_line_audio(line, out_path))
        print(f"  -> {out_path}")

    if selected and not dry_run and os.getenv("COSYVOICE_SKIP", "0") != "1":
        if os.getenv("PACE_SKIP", "0") != "1":
            from pipeline.pace import adjust_all_clips

            print("Adjusting clip pace (WPM targets)...")
            adjust_all_clips(selected, clips_dir)

    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", default="output/lines.json")
    parser.add_argument("--config", default="config/voices.yaml")
    parser.add_argument("--clips", default="output/clips")
    parser.add_argument("--only", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    generate_all(
        load_lines_json(args.lines),
        config_path=args.config,
        clips_dir=args.clips,
        only=args.only,
        dry_run=args.dry_run,
    )
