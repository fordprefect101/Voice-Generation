from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
import yaml
from dotenv import load_dotenv

from pipeline.models import EnrichedLine, load_lines_json

load_dotenv()

_MODEL = None
_CLONE_PROMPTS = None

# MUST match make_voices_refs.py exactly
REF_TEXT = (
    "Hi, I'm glad we're exploring this place together today. "
    "It feels good to take it slow and notice the details."
)

def get_device() -> str:
    preferred = os.getenv("QWEN_DEVICE", "mps").lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_voice_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_qwen_model():
    global _MODEL, _CLONE_PROMPTS
    if _MODEL is not None:
        return _MODEL

    from qwen_tts import Qwen3TTSModel

    device = get_device()
    dtype = torch.float32 if device == "mps" else (
        torch.bfloat16 if device == "cuda" else torch.float32
    )
    model_id = os.getenv(
        "QWEN_MODEL_PATH",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )
    print(f"Loading Qwen from {model_id} on {device} ({dtype})...")
    kwargs = {"device_map": device, "dtype": dtype, "attn_implementation": "sdpa"}
    if device == "cuda":
        kwargs["attn_implementation"] = "flash_attention_2"

    _MODEL = Qwen3TTSModel.from_pretrained(model_id, **kwargs)

    male_ref = Path("voices/refs/male_ref.wav")
    female_ref = Path("voices/refs/female_ref.wav")
    if not male_ref.exists() or not female_ref.exists():
        raise FileNotFoundError(
            "Missing voice refs. Run: python make_voices_refs.py"
        )

    print("Building clone prompts from voice refs...")
    _CLONE_PROMPTS = {
        "M": _MODEL.create_voice_clone_prompt(
            ref_audio=str(male_ref),
            ref_text=REF_TEXT,
        ),
        "F": _MODEL.create_voice_clone_prompt(
            ref_audio=str(female_ref),
            ref_text=REF_TEXT,
        ),
    }
    return _MODEL

def generate_line_audio(line: EnrichedLine, out_path: str | Path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_qwen_model()
    prompt = _CLONE_PROMPTS[line.speaker]
    print(f"  cloning speaker {line.speaker}...")

    wavs, sr = model.generate_voice_clone(
        text=line.text,
        language="English",
        voice_clone_prompt=prompt,
    )
    sf.write(out_path, wavs[0], sr)
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
    clips_dir = Path(clips_dir)
    selected = _filter_lines(lines, only, from_index, to_index)
    paths: list[str] = []

    for line in selected:
        out_path = clips_dir / f"{line.index:03d}_{line.speaker}.wav"
        if dry_run or os.getenv("QWEN_SKIP", "0") == "1":
            print(f"[dry-run] {out_path.name}")
            print(f"  speaker: {line.speaker}")
            print(f"  text: {line.text[:120]}...")
            paths.append(str(out_path))
            continue
        print(f"Generating line {line.index} ({line.speaker})...")
        paths.append(generate_line_audio(line, out_path))
        print(f"  -> {out_path}")
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