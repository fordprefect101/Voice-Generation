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
    attn_implementation="sdpa",
)

out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)

jobs = [
    {
        "file": "qwen_male.wav",
        "text": "Wow, this place looks even better than I imagined.",
        "instruct": (
            "Male, late 20s, warm Indian accent, "
            "relaxed conversational pacing, curious discovery tone"
        ),
    },
    {
        "file": "qwen_female.wav",
        "text": "I know, right? Wait until you see the courtyard.",
        "instruct": (
            "Female, early 30s, warm Indian accent, "
            "calm friendly voice, slightly slower pacing, playful amused tone"
        ),
    },
]

for job in jobs:
    print(f"Generating {job['file']}...")
    wavs, sr = model.generate_voice_design(
        text=job["text"],
        language="English",
        instruct=job["instruct"],
    )
    path = out_dir / job["file"]
    sf.write(path, wavs[0], sr)
    print(f"  -> {path}")

print("Done. Listen to both WAVs.")