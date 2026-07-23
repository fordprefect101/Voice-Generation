"""Smoke test: Qwen3 VoiceDesign. Listen to output/qwen_hello.wav"""
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

print(f"Loading VoiceDesign on {DEVICE} ({DTYPE})...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map=DEVICE,
    dtype=DTYPE,
    attn_implementation="sdpa",  # Mac: not flash_attention_2
)

wavs, sr = model.generate_voice_design(
    text="Hello.",
    language="English",
    instruct=(
        "Male, late 20s, warm Indian accent, "
        "relaxed conversational pacing, neutral tone"
    ),
)

out = Path("output/qwen_hello.wav")
out.parent.mkdir(parents=True, exist_ok=True)
sf.write(out, wavs[0], sr)
print(f"Wrote {out} — open and listen.")