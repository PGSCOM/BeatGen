#!/usr/bin/env bash
# Train on a separate GPU machine and leave nothing behind.
#   bash gpu.sh train [train.py args]    # needs data/proc (download.py --from + prepare.py)
#   CKPT_REPO=user/repo bash gpu.sh ...  # also upload the checkpoint to a private HF repo
#   bash gpu.sh clean                    # remove venv, data, checkpoints, caches and the HF token
set -euo pipefail
cd "$(dirname "$0")"
CKPT=${CKPT:-ckpt.pt}
LOG=${CKPT%.pt}.log

case "${1:-}" in
train)
  shift
  [ -d data/proc ] || { echo "No data: run python download.py --from meta.jsonl && python prepare.py"; exit 1; }
  [ -d .venv ] || uv venv --python 3.12 .venv
  source .venv/bin/activate
  uv pip install -q -r requirements.txt
  python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print(torch.cuda.get_device_name())"
  python train.py --out "$CKPT" "$@" 2>&1 | tee "$LOG"
  if [ -n "${CKPT_REPO:-}" ]; then
    hf auth whoami >/dev/null 2>&1 || hf auth login
    hf repos create "$CKPT_REPO" --private --exist-ok
    hf upload "$CKPT_REPO" "$CKPT" && hf upload "$CKPT_REPO" "$LOG"
  fi
  ;;
clean)
  hf auth logout 2>/dev/null || true
  rm -rf .venv data *.pt *.log __pycache__ ~/.cache/huggingface
  uv cache clean
  echo "Clean. Remove the repo folder with: rm -rf $(pwd)"
  ;;
*) sed -n 2,5p "$0" ;;
esac
