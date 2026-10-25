#!/usr/bin/env bash
# Orchestrates the Min-K%++ pipeline inside the GPU container.
set -euo pipefail

MODE="${1:-${MODE:-reproduce}}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
N_BOOT="${N_BOOT:-2000}"
NOTEBOOKS="${NOTEBOOKS:-0}"
HF_MODEL="${HF_MODEL:-state-spaces/mamba-1.4b-hf}"

short_model() {
  local m="$1"
  basename "${m}"
}

SHORT_MODEL="${SHORT_MODEL:-$(short_model "${HF_MODEL}")}"

have_cuda() {
  python - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
}

echo "=== Min-K%++ pipeline ==="
echo "mode=${MODE}  device=${DEVICE}  dtype=${DTYPE}  model=${HF_MODEL} (${SHORT_MODEL})"
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f"  gpu={torch.cuda.get_device_name(0)}  vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print()
PY

if [[ "${DEVICE}" == "cuda" ]] && ! have_cuda; then
  echo "WARNING: DEVICE=cuda but torch.cuda.is_available() is False."
  echo "         Falling back to cpu. Check nvidia-container-toolkit / --gpus all."
  DEVICE=cpu
fi

echo "--- self-check ---"
python scripts/minkpp.py
python scripts/01_cache.py --self-check

cache_one() {
  local hf="$1"
  local short
  short="$(short_model "${hf}")"
  local need=0
  local split
  for split in WikiMIA_length32 WikiMIA_length64 WikiMIA_length128; do
    if [[ ! -f "cache/${short}_${split}_${DTYPE}.npz" ]]; then
      need=1
      break
    fi
  done
  if [[ "${need}" -eq 0 ]]; then
    echo "caches for ${short} (${DTYPE}) already present — skip 01_cache"
    return 0
  fi
  echo "--- cache ${hf} on ${DEVICE}/${DTYPE} ---"
  python scripts/01_cache.py \
    --model "${hf}" \
    --all-splits \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"
}

aggregate_and_figures() {
  local short="$1"
  echo "--- aggregate ${short} ---"
  python scripts/02_aggregate.py --model "${short}" --dtype "${DTYPE}" --n-boot "${N_BOOT}"
  echo "--- figures ${short} ---"
  python scripts/03_figures.py --model "${short}"
}

run_notebooks() {
  local short="$1"
  echo "--- notebooks (MINKPP_MODEL=${short}) ---"
  MINKPP_MODEL="${short}" python scripts/build_notebooks.py --execute
}

case "${MODE}" in
  reproduce)
    # Rebuild tables/figures from committed caches (mamba + pythia-160m, float32).
    # No model download if caches are present.
    DTYPE=float32
    for short in mamba-1.4b-hf pythia-160m; do
      if ls cache/${short}_WikiMIA_length*_float32.npz >/dev/null 2>&1; then
        aggregate_and_figures "${short}"
      else
        echo "skip ${short}: no float32 caches"
      fi
    done
    if [[ "${NOTEBOOKS}" == "1" ]]; then
      run_notebooks mamba-1.4b-hf
    fi
    ;;

  cache)
    cache_one "${HF_MODEL}"
    ;;

  full)
    cache_one "${HF_MODEL}"
    aggregate_and_figures "${SHORT_MODEL}"
    if [[ "${NOTEBOOKS}" == "1" ]]; then
      run_notebooks "${SHORT_MODEL}"
    fi
    ;;

  # Convenience: GPU-cache Pythia-6.9B (the main outstanding model), then score it.
  # float16 is required to fit ~14 GB; override with PYTHIA69_DTYPE if needed.
  pythia69)
    DTYPE="${PYTHIA69_DTYPE:-float16}"
    HF_MODEL=EleutherAI/pythia-6.9b
    SHORT_MODEL=pythia-6.9b
    cache_one "${HF_MODEL}"
    aggregate_and_figures "${SHORT_MODEL}"
    ;;

  shell)
    exec bash
    ;;

  *)
    echo "Unknown mode: ${MODE}"
    echo "Usage: entrypoint.sh [reproduce|full|cache|pythia69|shell]"
    exit 1
    ;;
esac

echo "=== done (${MODE}) ==="
