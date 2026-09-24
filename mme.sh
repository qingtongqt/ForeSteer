#!/usr/bin/env bash
set -euo pipefail
source "scripts/eval_common.sh"
python infer_mme.py "${COMMON_ARGS[@]}" \
  --reference-dir "${REFERENCE_DIR:-data/mme/eval_tool/Your_Results}" \
  --base-dir "${BASE_DIR:-data/mme/MME_Benchmark}" \
  --answers-dir "$OUT_DIR/mme" --max_length "${MAX_LENGTH:-3}"
python eval_mme.py --results_dir "$OUT_DIR/mme" --save-json "$OUT_DIR/mme_metrics.json"
