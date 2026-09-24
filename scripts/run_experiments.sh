#!/usr/bin/env bash
# Each model generates and trains on its own annotated token trajectories.
set -euo pipefail
# sbatch may execute a spool copy, so BASH_SOURCE alone is not a repo path.
is_project_root() {
  [[ -f "$1/scripts/risk_pipeline.py" && -f "$1/model/base.py" ]]
}
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  if ! is_project_root "$PROJECT_ROOT"; then
    echo "Invalid PROJECT_ROOT: $PROJECT_ROOT" >&2; exit 1
  fi
else
  SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  for candidate in "$SCRIPT_DIR/.." "${SLURM_SUBMIT_DIR:-$PWD}" "${SLURM_SUBMIT_DIR:-$PWD}/.." "$PWD"; do
    if is_project_root "$candidate"; then
      PROJECT_ROOT="$candidate"
      break
    fi
  done
  if [[ -z "${PROJECT_ROOT:-}" ]]; then
    echo 'Cannot locate ForeSteer. Set PROJECT_ROOT to the absolute project directory when submitting the job.' >&2
    exit 1
  fi
fi
cd -- "$PROJECT_ROOT"
export PROJECT_ROOT="$PWD"
echo "Project directory: $PROJECT_ROOT"
TARGET="${1:?Usage: bash scripts/run_experiments.sh llava|qwen_vl|mplug_owl2 prepare|data|train|chair|pope|mme|eval [chair|pope|mme|all]}"
ACTION="${2:-prepare}"
case "$TARGET" in
  llava|llava-v1.5-7b)
    export MODEL_NAME=llava
    export MODEL_PATH="${MODEL_PATH:-$HOME/models/llava-v1.5-7b}" ;;
  qwen_vl|qwen-vl|qwen-vl-chat)
    export MODEL_NAME=qwen-vl-chat
    export MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen-VL-Chat}" ;;
  mplug_owl2|mplug-owl2)
    export MODEL_NAME=mplug-owl2
    export MODEL_PATH="${MODEL_PATH:-$HOME/models/mplug-owl2-llama2-7b}" ;;
  *) echo "Unsupported target: $TARGET" >&2; exit 1 ;;
esac
case "$ACTION" in prepare|data|train|chair|pope|mme|eval) ;; *) echo "Unknown action: $ACTION" >&2; exit 1 ;; esac
BENCHMARK="${3:-all}"
case "$BENCHMARK" in chair|pope|mme|all) ;; *) echo "Unknown benchmark: $BENCHMARK" >&2; exit 1 ;; esac
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/outputs/${MODEL_NAME}_coco_train2014}"
RISK_OUTPUT="${RISK_OUTPUT:-$PROJECT_ROOT/outputs/risk/${MODEL_NAME}}"
echo "Data directory: $DATA_DIR"
echo "Risk directory: $RISK_OUTPUT"
COCO_ROOT="${COCO_ROOT:-$HOME/dataset/coco}"
LAYER="${LAYER:-16}"
GAMMA="${GAMMA:-0.9}"
RISK_HORIZON="${RISK_HORIZON:-16}"
export SEED="${SEED:-42}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
if [[ -f "$RISK_OUTPUT/manifest.json" ]] && {
  [[ "$ACTION" == prepare ]] || [[ "$ACTION" == train && "${SKIP_EXTRACT:-0}" != 1 ]];
}; then
  echo 'Extraction exists; use SKIP_EXTRACT=1 with train, or choose a new RISK_OUTPUT before prepare.' >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mllm
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/foresteer-matplotlib}"
if [[ "$ACTION" == prepare || "$ACTION" == data || "$ACTION" == train ]]; then
  if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
  else
    GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
    GPU_LIST=()
    for ((gpu=0; gpu<GPU_COUNT; gpu++)); do GPU_LIST+=("$gpu"); done
  fi
  NUM_GPUS=${#GPU_LIST[@]}
  if (( NUM_GPUS == 0 )) || [[ "${GPU_LIST[0]:-}" == -1 ]]; then
    echo 'No visible GPU selected' >&2; exit 1
  fi
  declare -A SEEN_GPUS=()
  for gpu in "${GPU_LIST[@]}"; do
    if [[ -z "$gpu" || -n "${SEEN_GPUS[$gpu]:-}" ]]; then
      echo 'Visible GPU entries must be nonempty and unique' >&2; exit 1
    fi
    SEEN_GPUS[$gpu]=1
  done
  PIDS=()
  cleanup_workers() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  }
  trap cleanup_workers EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  wait_workers() {
    local failed=0
    for pid in "${PIDS[@]}"; do
      if ! wait "$pid"; then failed=1; fi
    done
    PIDS=()
    if (( failed )); then
      echo 'A GPU worker failed; no merge or training will run. Check shard logs.' >&2
      exit 1
    fi
  }
  echo "Using $NUM_GPUS GPU workers: ${GPU_LIST[*]}"
fi
if [[ "$ACTION" == prepare || "$ACTION" == data ]]; then
  mkdir -p "$DATA_DIR/logs"
  SHARD_FILES=()
  for ((rank=0; rank<NUM_GPUS; rank++)); do
    shard="$DATA_DIR/captions.rank-${rank}-of-${NUM_GPUS}.jsonl"
    SHARD_FILES+=("$shard")
    CUDA_VISIBLE_DEVICES="${GPU_LIST[rank]}" python -u scripts/generate_coco_captions.py \
      --model-name "$MODEL_NAME" --model-path "$MODEL_PATH" --coco-root "$COCO_ROOT" \
      --split train2014 --output "$shard" --num-shards "$NUM_GPUS" --shard-index "$rank" \
      --max-samples "${MAX_SAMPLES:-5000}" --max-new-tokens "${MAX_NEW_TOKENS:-128}" --seed "$SEED" \
      > "$DATA_DIR/logs/generate-${rank}.log" 2>&1 &
    PIDS+=("$!")
  done
  echo "Generation logs: $DATA_DIR/logs/generate-*.log"
  wait_workers
  python scripts/merge_caption_shards.py --input "${SHARD_FILES[@]}" --output "$DATA_DIR/captions.jsonl"
  python -u scripts/annotate_chair_tokens.py --input "$DATA_DIR/captions.jsonl" \
    --output "$DATA_DIR/chair_tokens.jsonl" --summary "$DATA_DIR/chair_summary.json" \
    --coco-root "$COCO_ROOT" --cache "${CHAIR_CACHE:-outputs/cache/chair_coco2014.pkl}"
fi
if [[ "$ACTION" == prepare || "$ACTION" == train ]]; then
  if [[ "${SKIP_EXTRACT:-0}" != 1 ]]; then
    mkdir -p "$RISK_OUTPUT/logs"
    for ((rank=0; rank<NUM_GPUS; rank++)); do
      CUDA_VISIBLE_DEVICES="${GPU_LIST[rank]}" python -u scripts/risk_pipeline.py extract --output "$RISK_OUTPUT" \
        --model-name "$MODEL_NAME" --model-path "$MODEL_PATH" --layers "$LAYER" --seed "$SEED" \
        --num-shards "$NUM_GPUS" --shard-index "$rank" \
        --input "${TRAIN_INPUT:-$DATA_DIR/chair_tokens.jsonl}" --coco-root "$COCO_ROOT" \
        > "$RISK_OUTPUT/logs/extract-${rank}.log" 2>&1 &
      PIDS+=("$!")
    done
    echo "Extraction logs: $RISK_OUTPUT/logs/extract-*.log"
    wait_workers
    if (( NUM_GPUS > 1 )); then
      python scripts/merge_risk_shards.py --output "$RISK_OUTPUT" --num-shards "$NUM_GPUS"
    fi
  fi
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -u scripts/risk_pipeline.py train --output "$RISK_OUTPUT" \
    --layer "$LAYER" --gamma "$GAMMA" --risk-horizon "$RISK_HORIZON" --seed "$SEED"
fi
case "$ACTION" in prepare|data|train) exit 0 ;; esac
export RISK_CHECKPOINT="${RISK_CHECKPOINT:-$RISK_OUTPUT/risk_layer${LAYER}_gamma${GAMMA}_h${RISK_HORIZON}_seed${SEED}.pt}"
if [[ ! -f "$RISK_CHECKPOINT" ]]; then
  echo "Missing checkpoint: $RISK_CHECKPOINT; run prepare first" >&2; exit 1
fi
export ALPHA="${ALPHA:?Set ALPHA explicitly for the target-model experiment}"
export RUN_NAME="${RUN_NAME:-risk_layer${LAYER}_gamma${GAMMA}_h${RISK_HORIZON}_seed${SEED}_alpha${ALPHA}_adaptive${ADAPTIVE_ALPHA:-0}}"
if [[ "$ACTION" != eval ]]; then BENCHMARK="$ACTION"; fi
if [[ "$BENCHMARK" == chair || "$BENCHMARK" == all ]]; then bash chair.sh; fi
if [[ "$BENCHMARK" == pope || "$BENCHMARK" == all ]]; then
  for split in random popular adversarial; do bash pope.sh coco "$split"; done
fi
if [[ "$BENCHMARK" == mme || "$BENCHMARK" == all ]]; then bash mme.sh; fi
