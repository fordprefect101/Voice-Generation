from pathlib import Path
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from pipeline.hosts import host_qwen_instruct, host_ref_sample

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

print(f"Loading VoiceDesign on {DEVICE} ({DTYPE})...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map=DEVICE,
    dtype=DTYPE,
    attn_implementation="sdpa",
)

out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)

# Instruct style aligned with Qwen3-TTS VoiceDesign docs:
# gender, age, timbre, accent, pace, emotion/prosody — not cast-relationship prose.
# https://qwen.ai/blog?id=qwen3tts-0115
jobs = [
    {
        "file": "qwen_male.wav",
        "text": host_ref_sample("M"),
        "instruct": host_qwen_instruct("M"),
    },
    {
        "file": "qwen_female.wav",
        "text": host_ref_sample("F"),
        "instruct": host_qwen_instruct("F"),
    },
]
for job in jobs:
    print(f"Generating {job['file']}...")
    print(f"  instruct: {job['instruct']}")
    wavs, sr = model.generate_voice_design(
        text=job["text"],
        language="English",
        instruct=job["instruct"],
    )
    path = out_dir / job["file"]
    sf.write(path, wavs[0], sr)
    print(f"  -> {path}")

print("Done. Listen to both WAVs.")
