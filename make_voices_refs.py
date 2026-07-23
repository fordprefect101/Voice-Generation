from pathlib import Path
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
import yaml

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.float32

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map=DEVICE,
    dtype=DTYPE,
    attn_implementation="sdpa",
)

voices = yaml.safe_load(open("config/voices.yaml"))
ref_dir = Path("voices/refs")
ref_dir.mkdir(parents=True, exist_ok=True)

# Same neutral line for both — identity, not emotion
REF_TEXT = (
    "Hi, I'm glad we're exploring this place together today. "
    "It feels good to take it slow and notice the details."
)

for key, filename in [("male", "male_ref.wav"), ("female", "female_ref.wav")]:
    instruct = voices[key]["qwen_instruct"].strip() + ", neutral calm delivery"
    print(f"Creating {filename}...")
    wavs, sr = model.generate_voice_design(
        text=REF_TEXT,
        language="English",
        instruct=instruct,
    )
    path = ref_dir / filename
    sf.write(path, wavs[0], sr)
    print(f"  -> {path}")

print("Listen to both refs. If either is wrong, delete and re-run that speaker.")