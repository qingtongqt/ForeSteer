#!/usr/bin/env bash
set -euo pipefail
source "scripts/eval_common.sh"
python infer_chair.py "${COMMON_ARGS[@]}" \
  --image-folder "${IMAGE_FOLDER:-$HOME/dataset/coco/val2014}" \
  --answers-file "$OUT_DIR/chair.jsonl" --max-samples "${MAX_SAMPLES:-500}" --max_length "${MAX_LENGTH:-64}"
python eval_chair.py --cap_file "$OUT_DIR/chair.jsonl" \
  --coco_path "${COCO_ANNOTATION_PATH:-$HOME/dataset/coco/annotations}" \
  --cache "${CHAIR_CACHE:-outputs/cache/chair_eval_coco2014.pkl}" --save_path "$OUT_DIR/chair_metrics.json"
