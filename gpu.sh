#!/usr/bin/env bash
# Entrena en otra máquina con GPU sin dejar rastro.
#   bash gpu.sh train  [args de train.py]   # venv + entrenamiento sobre data/proc + sube ckpt.pt a HF
#   CKPT=ckpt_big.pt bash gpu.sh train ...  # otro nombre: no pisa el checkpoint anterior
#   bash gpu.sh clean                       # borra venv, datos, cachés y el token de HF
# Requisitos: uv, una GPU NVIDIA con driver y data/proc (download.py --from + prepare.py).
# Ejecutar desde la carpeta del repo. CKPT_REPO: repo de HF al que se sube el modelo (privado).
set -euo pipefail
cd "$(dirname "$0")"
CKPT_REPO=${CKPT_REPO:-PGSCOM/beatgen-ckpt}
CKPT=${CKPT:-ckpt.pt}
LOG=${CKPT%.pt}.log

case "${1:-}" in
train)
  shift
  [ -d .venv ] || uv venv --python 3.12 .venv
  source .venv/bin/activate
  uv pip install -q -r requirements.txt huggingface_hub
  python -c "import torch; assert torch.cuda.is_available(), 'torch no ve la GPU'; print(torch.cuda.get_device_name())"
  hf auth whoami >/dev/null 2>&1 || hf auth login
  [ -d data/proc ] || { echo "Faltan los datos: python download.py --from meta.jsonl && python prepare.py"; exit 1; }
  python train.py --out "$CKPT" "$@" 2>&1 | tee "$LOG"
  hf repos create "$CKPT_REPO" --private --exist-ok
  hf upload "$CKPT_REPO" "$CKPT" && hf upload "$CKPT_REPO" "$LOG"
  ;;
clean)
  hf auth logout 2>/dev/null || true
  rm -rf .venv data *.pt *.log __pycache__ ~/.cache/huggingface
  uv cache clean
  echo "Limpio. Borra la carpeta del repo con: rm -rf $(pwd)"
  ;;
*) sed -n 2,7p "$0" ;;
esac
