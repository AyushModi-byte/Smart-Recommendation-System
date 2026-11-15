#!/usr/bin/env bash
# One-time environment setup on the DGX. Run from anywhere:
#   bash Smart-Recommendation-System/scripts/dgx_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p brain runs

python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| gpus:", torch.cuda.device_count())
EOF
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
