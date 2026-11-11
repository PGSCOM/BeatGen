#!/usr/bin/env bash
# Entrena en otra máquina con GPU sin dejar rastro.
#   bash gpu.sh train  [args de train.py]   # venv + datos + entrenamiento + sube ckpt.pt a HF
#   bash gpu.sh clean                       # borra venv, datos, cachés y el token de HF
# Requisitos: uv y una GPU NVIDIA con driver. Ejecutar desde la carpeta del repo.
set -euo pipefail
cd "$(dirname "$0")"
CKPT_REPO=PGSCOM/beatgen-ckpt

case "${1:-}" in
train)
  shift
  [ -d .venv ] || uv venv --python 3.12 .venv
  source .venv/bin/activate
  uv pip install -q -r requirements.txt huggingface_hub
  python -c "import torch; assert torch.cuda.is_available(), 'torch no ve la GPU'; print(torch.cuda.get_device_name())"
  hf auth whoami >/dev/null 2>&1 || hf auth login
  hf download PGSCOM/beatgen-maps --type dataset --include "proc/*" --local-dir data   # solo los .npz (~2 GB)
  python train.py "$@" 2>&1 | tee train.log
  hf repos create "$CKPT_REPO" --private --exist-ok
  hf upload "$CKPT_REPO" ckpt.pt && hf upload "$CKPT_REPO" train.log
  ;;
clean)
  hf auth logout 2>/dev/null || true
  rm -rf .venv data ckpt.pt train.log __pycache__ ~/.cache/huggingface
  uv cache clean
  echo "Limpio. Borra la carpeta del repo con: rm -rf $(pwd)"
  ;;
*) sed -n 2,5p "$0" ;;
esac
