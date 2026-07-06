from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from dotenv import load_dotenv

from pipeline.emotions import apply_emotion_tag
from pipeline.models import EnrichedLine, load_lines_json

load_dotenv()

# Maya1 token IDs (from Hugging Face model card)
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
TEXT_EOT_ID = 128009

_MODEL = None
_TOKENIZER = None
_SNAC = None

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name, str(default)).strip().lower()
    return val in {"1", "true", "yes", "on"}

def get_device() -> str:
    preferred = os.getenv("MAYA_DEVICE", "mps").lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _model_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32

def load_voice_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_voice_prompt(speaker: str, config: dict) -> str:
    key = "male" if speaker == "M" else "female"
    return config[key]["maya_prompt"].strip()

def build_maya_prompt(tokenizer, description: str, text: str) -> str:
    soh = tokenizer.decode([SOH_ID])
    eoh = tokenizer.decode([EOH_ID])
    soa = tokenizer.decode([SOA_ID])
    sos = tokenizer.decode([CODE_START_TOKEN_ID])
    eot = tokenizer.decode([TEXT_EOT_ID])
    bos = tokenizer.bos_token

    formatted = f'<description="{description}"> {text}'
    return soh + bos + formatted + eot + eoh + soa + sos

def _extract_snac_codes(token_ids: list[int]) -> list[int]:
    try:
        eos_idx = token_ids.index(CODE_END_TOKEN_ID)
    except ValueError:
        eos_idx = len(token_ids)
    return [t for t in token_ids[:eos_idx] if SNAC_MIN_ID <= t <= SNAC_MAX_ID]

def _unpack_snac_from_7(snac_tokens: list[int]) -> list[list[int]]:
    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]

    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    snac_tokens = snac_tokens[: frames * SNAC_TOKENS_PER_FRAME]
    if frames == 0:
        return [[], [], []]

    l1, l2, l3 = [], [], []
    for i in range(frames):
        slots = snac_tokens[i * 7 : (i + 1) * 7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend([
            (slots[1] - CODE_TOKEN_OFFSET) % 4096,
            (slots[4] - CODE_TOKEN_OFFSET) % 4096,
        ])
        l3.extend([
            (slots[2] - CODE_TOKEN_OFFSET) % 4096,
            (slots[3] - CODE_TOKEN_OFFSET) % 4096,
            (slots[5] - CODE_TOKEN_OFFSET) % 4096,
            (slots[6] - CODE_TOKEN_OFFSET) % 4096,
        ])
    return [l1, l2, l3]

def load_maya_model():
    global _MODEL, _TOKENIZER, _SNAC
    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _SNAC

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snac import SNAC

    device = get_device()
    dtype = _model_dtype(device)
    model_path = os.getenv("MAYA_MODEL_PATH", "maya-research/maya1")

    print(f"Loading Maya from {model_path} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    _MODEL, _TOKENIZER, _SNAC = model, tokenizer, snac
    return model, tokenizer, snac

def generate_line_audio(
    line: EnrichedLine,
    voice_prompt: str,
    out_path: str | Path,
) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tagged_text = apply_emotion_tag(line.text, line.emotion)
    model, tokenizer, snac_model = load_maya_model()
    device = get_device()

    prompt = build_maya_prompt(tokenizer, voice_prompt, tagged_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    temperature = float(os.getenv("MAYA_TEMPERATURE", "0.35"))
    top_p = float(os.getenv("MAYA_TOP_P", "0.9"))
    repetition_penalty = float(os.getenv("MAYA_REPETITION_PENALTY", "1.1"))
    max_new_tokens = int(os.getenv("MAYA_MAX_NEW_TOKENS", "2048"))

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=28,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            eos_token_id=CODE_END_TOKEN_ID,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0, inputs["input_ids"].shape[1] :].tolist()
    snac_tokens = _extract_snac_codes(generated)
    if len(snac_tokens) < 7:
        raise RuntimeError(
            f"Line {line.index}: not enough SNAC tokens generated ({len(snac_tokens)})"
        )

    levels = _unpack_snac_from_7(snac_tokens)
    codes = [
        torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
        for level in levels
    ]

    with torch.inference_mode():
        z_q = snac_model.quantizer.from_codes(codes)
        audio = snac_model.decoder(z_q)[0, 0].cpu().numpy()

    if len(audio) > 2048:
        audio = audio[2048:]

    sf.write(out_path, audio, 24000)
    return str(out_path)

def _filter_lines(
    lines: list[EnrichedLine],
    only: int | None = None,
    from_index: int | None = None,
    to_index: int | None = None,
) -> list[EnrichedLine]:
    if only is not None:
        return [line for line in lines if line.index == only]
    if from_index is not None or to_index is not None:
        start = from_index or 1
        end = to_index or max(line.index for line in lines)
        return [line for line in lines if start <= line.index <= end]
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
    config = load_voice_config(config_path)
    clips_dir = Path(clips_dir)
    selected = _filter_lines(lines, only=only, from_index=from_index, to_index=to_index)
    paths: list[str] = []

    for line in selected:
        voice_prompt = get_voice_prompt(line.speaker, config)
        tagged = apply_emotion_tag(line.text, line.emotion)
        out_path = clips_dir / f"{line.index:03d}_{line.speaker}.wav"

        if dry_run or _env_bool("MAYA_SKIP"):
            print(f"[dry-run] {out_path.name}")
            print(f"  voice: {voice_prompt[:80]}...")
            print(f"  text:  {tagged[:120]}...")
            paths.append(str(out_path))
            continue

        print(f"Generating line {line.index} ({line.speaker})...")
        paths.append(generate_line_audio(line, voice_prompt, out_path))
        print(f"  -> {out_path}")

    return paths

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Maya TTS clips")
    parser.add_argument("--lines", default="output/lines.json")
    parser.add_argument("--config", default="config/voices.yaml")
    parser.add_argument("--clips", default="output/clips")
    parser.add_argument("--only", type=int, default=None)
    parser.add_argument("--from", dest="from_index", type=int, default=None)
    parser.add_argument("--to", dest="to_index", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lines = load_lines_json(args.lines)
    generate_all(
        lines,
        config_path=args.config,
        clips_dir=args.clips,
        only=args.only,
        from_index=args.from_index,
        to_index=args.to_index,
        dry_run=args.dry_run,
    )