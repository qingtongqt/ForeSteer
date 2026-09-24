#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:-coco}"
SPLIT="${2:-random}"
source "scripts/eval_common.sh"
QUESTION_FILE="${QUESTION_FILE:-data/pope/$DATASET/${DATASET}_pope_${SPLIT}.json}"
python infer_pope.py "${COMMON_ARGS[@]}" \
  --image-folder "${IMAGE_FOLDER:-$HOME/dataset/coco/val2014}" \
  --question-file "$QUESTION_FILE" --answers-file "$OUT_DIR/pope_${DATASET}_${SPLIT}.jsonl" --max_length "${MAX_LENGTH:-3}"
python eval_pope.py --gt_files "$QUESTION_FILE" --gen_files "$OUT_DIR/pope_${DATASET}_${SPLIT}.jsonl" \
  --save-json "$OUT_DIR/pope_${DATASET}_${SPLIT}_metrics.json"
