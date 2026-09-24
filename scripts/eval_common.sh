#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mllm
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
MODEL_NAME="${MODEL_NAME:-llava}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/llava-v1.5-7b}"
ALPHA="${ALPHA:?Set a positive ALPHA for evaluation}"
SEED="${SEED:-42}"
: "${RISK_CHECKPOINT:?Set RISK_CHECKPOINT for evaluation}"
if [[ ! -f "$RISK_CHECKPOINT" ]]; then
  echo "Missing checkpoint: $RISK_CHECKPOINT" >&2; exit 1
fi
RUN_NAME="${RUN_NAME:-$(basename "${RISK_CHECKPOINT}" .pt)_alpha${ALPHA}_adaptive${ADAPTIVE_ALPHA:-0}}"
OUT_DIR="${OUT_DIR:-outputs/eval/${MODEL_NAME}/${RUN_NAME}}"
mkdir -p "$OUT_DIR"
COMMON_ARGS=(--model-name "$MODEL_NAME" --model-path "$MODEL_PATH" --seed "$SEED" --alpha "$ALPHA" --batch-size "${BATCH_SIZE:-8}")
COMMON_ARGS+=(--risk-checkpoint "$RISK_CHECKPOINT")
if [[ "${ADAPTIVE_ALPHA:-0}" == 1 ]]; then COMMON_ARGS+=(--adaptive-alpha); fi
