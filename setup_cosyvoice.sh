#!/usr/bin/env bash
# Clone CosyVoice + download Fun-CosyVoice3-0.5B weights.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="${COSYVOICE_REPO:-$ROOT/third_party/CosyVoice}"
MODEL_DIR="${COSYVOICE_MODEL_DIR:-$REPO/pretrained_models/Fun-CosyVoice3-0.5B}"

mkdir -p "$(dirname "$REPO")"

if [[ ! -d "$REPO/.git" ]]; then
  echo "Cloning CosyVoice -> $REPO"
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$REPO"
else
  echo "CosyVoice already present at $REPO"
  git -C "$REPO" submodule update --init --recursive
fi

echo "Installing CosyVoice Python deps into current env..."
python -m pip install -r "$REPO/requirements.txt"

if [[ ! -f "$MODEL_DIR/cosyvoice3.yaml" && ! -f "$MODEL_DIR/cosyvoice.yaml" ]]; then
  echo "Downloading Fun-CosyVoice3-0.5B-2512 -> $MODEL_DIR"
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    local_dir="$MODEL_DIR",
)
print("Model ready:", "$MODEL_DIR")
PY
else
  echo "Model already present at $MODEL_DIR"
fi

echo
echo "Done."
echo "  export COSYVOICE_REPO=$REPO"
echo "  export COSYVOICE_MODEL_DIR=$MODEL_DIR"
echo "  python smoke_cosyvoice.py"
echo "  python run_pipeline.py your_script.txt --dry-run"
