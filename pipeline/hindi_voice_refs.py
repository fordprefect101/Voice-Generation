"""Shared paths and helpers for Hindi Fish voice refs (Mac-only generation)."""
from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import yaml

HI_REF_DIR = Path("voices/refs/hi")
HI_CAND_DIR = HI_REF_DIR / "candidates"
HI_PROMPTS_DIR = HI_REF_DIR / "indicf5_prompts"
HI_CONFIG_PATH = Path("config/hindi_hosts.yaml")

FILENAMES = {"male": "male_ref.wav", "female": "female_ref.wav"}

INDICF5_PROMPT_BASE = (
    "https://raw.githubusercontent.com/AI4Bharat/IndicF5/main/prompts"
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def load_hindi_hosts(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else HI_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def speaker_block(speaker: str) -> dict[str, Any]:
    key = "male" if speaker in ("male", "M") else "female"
    block = load_hindi_hosts().get(key) or {}
    if not isinstance(block, dict):
        raise ValueError(f"Invalid hindi_hosts.yaml entry for {key}")
    return block


def ref_sample(speaker: str) -> str:
    return " ".join(str(speaker_block(speaker).get("ref_sample") or "").split())


def parler_description(speaker: str) -> str:
    return " ".join(str(speaker_block(speaker).get("parler_description") or "").split())


def indicf5_seed(speaker: str) -> dict[str, str]:
    seeds = load_hindi_hosts().get("indicf5_seeds") or {}
    key = "male" if speaker in ("male", "M") else "female"
    block = seeds.get(key) or {}
    if not block.get("prompt_file") or not block.get("ref_text"):
        raise ValueError(f"Missing indicf5_seeds.{key} in hindi_hosts.yaml")
    return {
        "prompt_file": str(block["prompt_file"]).strip(),
        "ref_text": " ".join(str(block["ref_text"]).split()),
    }


def load_ref_texts(path: Path | None = None) -> dict[str, Any]:
    p = path or (HI_REF_DIR / "ref_texts.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_ref_texts(
    male: str,
    female: str,
    *,
    backend: str,
    model: str,
    path: Path | None = None,
) -> Path:
    p = path or (HI_REF_DIR / "ref_texts.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load_ref_texts(p)
    data["male"] = male
    data["female"] = female
    data["backend"] = backend
    data["model"] = model
    data["language"] = "hi"
    data["script"] = "devanagari"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


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


def load_scores(path: Path | None = None) -> dict[str, Any]:
    p = path or (HI_CAND_DIR / "scores.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_scores(scores: dict[str, Any], path: Path | None = None) -> None:
    p = path or (HI_CAND_DIR / "scores.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scores, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def promote_take(speaker: str, take: int, *, backend: str) -> Path:
    src = HI_CAND_DIR / f"{backend}_{speaker}_take{take:02d}.wav"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")
    dst = HI_REF_DIR / FILENAMES[speaker]
    HI_REF_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return dst


def ensure_indicf5_prompt(speaker: str) -> tuple[Path, str]:
    """Download bundled IndicF5 seed wav if needed; return (path, ref_text)."""
    seed = indicf5_seed(speaker)
    HI_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    local = HI_PROMPTS_DIR / seed["prompt_file"]
    if not local.exists():
        import urllib.request

        url = f"{INDICF5_PROMPT_BASE}/{seed['prompt_file']}"
        print(f"Downloading IndicF5 seed {url} ...")
        urllib.request.urlretrieve(url, local)
    return local, seed["ref_text"]
