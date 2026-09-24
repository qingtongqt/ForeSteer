#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PROJECT_ROOT/scripts/prepare_llava_ablation.sh"

if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash scripts/run_llava_ablation.sh --alphas 1 5 10'
  echo 'Or: ALPHAS="1 5 10" bash scripts/run_llava_ablation.sh'
  echo 'Environment: OUTPUT, RISK_OUTPUT, TRAIN_INPUT, MODEL_PATH, COCO_ROOT, LAYERS, GAMMAS, SEED,'
  echo 'RISK_HORIZON, BATCH_SIZE, CHAIR_SAMPLES, CHAIR_MAX_TOKENS, POPE_MAX_TOKENS, ADAPTIVE_ALPHA, DRY_RUN'
  exit 0
fi
if [[ "${1:-}" == --alphas ]]; then
  shift
  [[ $# -gt 0 ]] || { echo '--alphas requires values' >&2; exit 1; }
  ALPHAS="$*"
elif [[ $# -gt 0 ]]; then
  echo 'Unknown arguments; use --help' >&2; exit 1
fi
: "${ALPHAS:?Specify positive alpha values: --alphas 1 5 10 (or ALPHAS environment variable)}"
ablation_init
PREDICTOR_OUTPUT="$OUTPUT/predictors"
RISK_HORIZON="${RISK_HORIZON:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CHAIR_SAMPLES="${CHAIR_SAMPLES:-500}"
CHAIR_MAX_TOKENS="${CHAIR_MAX_TOKENS:-64}"
POPE_MAX_TOKENS="${POPE_MAX_TOKENS:-3}"
ADAPTIVE_ALPHA="${ADAPTIVE_ALPHA:-1}"
QUESTION_FILE="${QUESTION_FILE:-$PROJECT_ROOT/data/pope/coco/coco_pope_random.json}"
# Normalize floats, particularly gamma=0/1 checkpoint names, and reject NaN/Inf.
NORMALIZED=$(python - "${GAMMAS:-0 0.5 0.9 1}" "$ALPHAS" <<'PY'
import math, sys
for index, text in enumerate(sys.argv[1:]):
    values = list(dict.fromkeys(map(float, text.split())))
    if not values or any(not math.isfinite(v) or v < 0 or (index == 0 and v > 1) or (index == 1 and v == 0) for v in values):
        raise SystemExit('Require gamma in [0,1] and finite positive alpha')
    print(' '.join(map(str, values)))
PY
)
read -r -a GAMMA_LIST <<< "${NORMALIZED%%$'\n'*}"
read -r -a ALPHA_LIST <<< "${NORMALIZED#*$'\n'}"
for value in "$RISK_HORIZON" "$BATCH_SIZE" "$CHAIR_SAMPLES" "$CHAIR_MAX_TOKENS" "$POPE_MAX_TOKENS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Sizes/horizon/token limits must be positive integers' >&2; exit 1; }
done
[[ "$ADAPTIVE_ALPHA" == 0 || "$ADAPTIVE_ALPHA" == 1 ]] || { echo 'ADAPTIVE_ALPHA must be 0 or 1' >&2; exit 1; }
if [[ "$DRY_RUN" != 1 ]]; then
  validate_manifest
  mkdir -p "$OUTPUT/eval"
  # Alpha/layer/gamma lists may expand on later runs; other settings stay fixed.
  CONFIG=$(printf '%s\n' "$MODEL_PATH" "$COCO_ROOT" "$RISK_OUTPUT" "$SEED" "$RISK_HORIZON" \
    "$BATCH_SIZE" "$CHAIR_SAMPLES" "$CHAIR_MAX_TOKENS" "$POPE_MAX_TOKENS" "$ADAPTIVE_ALPHA"; \
    sha256sum "$RISK_OUTPUT/manifest.json" "$QUESTION_FILE")
  (
    flock -x 9
    if [[ -f "$OUTPUT/ablation_config.txt" ]]; then
      if [[ "$(cat "$OUTPUT/ablation_config.txt")" != "$CONFIG" ]]; then
        echo 'Experiment settings changed; choose a new OUTPUT (RISK_OUTPUT can reuse prepared features).' >&2; exit 1
      fi
    else
      printf '%s\n' "$CONFIG" > "$OUTPUT/ablation_config.txt"
    fi
  ) 9>"$OUTPUT/ablation_config.lock"
fi

append_summary() {
  python - "$OUTPUT" "$1" <<'PY'
import csv, fcntl, json, os, sys
from pathlib import Path
root, folder = map(Path, sys.argv[1:])
row = json.loads((folder/'run.json').read_text())
p = json.loads((folder/'pope_random_metrics.json').read_text())
c = json.loads((folder/'chair_metrics.json').read_text())['overall_metrics']
row.update({f'pope_{k}': v for k,v in p.items() if isinstance(v, (int,float))})
row.update({f'chair_{k}': v for k,v in c.items()})
# Lock the stable lock file across checking the header, deduplication and append.
with (root/'summary.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    with (root/'summary.csv').open('a+', newline='') as f:
        f.seek(0)
        reader = csv.DictReader(f)
        fields = reader.fieldnames or list(row)
        if any(r['run'] == row['run'] for r in reader):
            sys.exit(0)
        if set(row) - set(fields):
            raise SystemExit('Existing summary.csv has incompatible columns')
        f.seek(0, os.SEEK_END)
        writer = csv.DictWriter(f, fieldnames=fields)
        if f.tell() == 0: writer.writeheader()
        writer.writerow(row)
        f.flush(); os.fsync(f.fileno())
PY
}

evaluate() (
  local layer="$1" gamma="$2" alpha="$3" checkpoint="${4:?Risk checkpoint required}" name out
  name="layer${layer}_gamma${gamma}_alpha${alpha}"
  out="$OUTPUT/eval/$name"
  local args=(--model-name llava --model-path "$MODEL_PATH" --seed "$SEED" --batch-size "$BATCH_SIZE"
    --alpha "$alpha" --risk-checkpoint "$checkpoint" --image-folder "$COCO_ROOT/val2014")
  if [[ "$ADAPTIVE_ALPHA" == 1 ]]; then args+=(--adaptive-alpha); fi
  if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$out"
    # Serialize identical evaluation configurations.
    exec 8>"$out/run.lock"
    flock -x 8
    # Completeness is restored only after both metrics are available.
    if [[ -f "$out/complete" ]]; then unlink "$out/complete"; fi
  fi
  ablation_step "$out/pope_infer.log" "$out/pope_random.jsonl" \
    python -u infer_pope.py "${args[@]}" --question-file "$QUESTION_FILE" \
    --answers-file "$out/pope_random.jsonl" --max_length "$POPE_MAX_TOKENS"
  # Always rescore predictions, so a retried inference cannot leave stale metrics.
  if [[ "$DRY_RUN" != 1 ]]; then printf '' > "$out/pope_eval.log.done"; fi
  ablation_step "$out/pope_eval.log" "$out/pope_random_metrics.json" \
    python eval_pope.py --gt_files "$QUESTION_FILE" --gen_files "$out/pope_random.jsonl" \
    --save-json "$out/pope_random_metrics.json"
  ablation_step "$out/chair_infer.log" "$out/chair.jsonl" \
    python -u infer_chair.py "${args[@]}" --answers-file "$out/chair.jsonl" \
    --max-samples "$CHAIR_SAMPLES" --max_length "$CHAIR_MAX_TOKENS"
  if [[ "$DRY_RUN" != 1 ]]; then printf '' > "$out/chair_eval.log.done"; fi
  ablation_step "$out/chair_eval.log" "$out/chair_metrics.json" \
    flock -x "$OUTPUT/eval/chair_cache.lock" python eval_chair.py --cap_file "$out/chair.jsonl" --coco_path "$COCO_ROOT/annotations" \
    --cache "$PROJECT_ROOT/outputs/cache/chair_eval_coco2014.pkl" --save_path "$out/chair_metrics.json"
  if [[ "$DRY_RUN" != 1 ]]; then
    python - "$out/run.json" "$name" "$layer" "$gamma" "$alpha" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(zip(['run','layer','gamma','alpha'],sys.argv[2:])), indent=2))
PY
    touch "$out/complete"
    append_summary "$out"
  fi
)

for layer in "${LAYER_LIST[@]}"; do
  for gamma in "${GAMMA_LIST[@]}"; do
    checkpoint="$PREDICTOR_OUTPUT/risk_layer${layer}_gamma${gamma}_h${RISK_HORIZON}_seed${SEED}.pt"
    trained=0
    for alpha in "${ALPHA_LIST[@]}"; do
      if [[ "$trained" == 0 ]]; then
        train_log="$PREDICTOR_OUTPUT/logs/$(basename "$checkpoint").train.log"
        (
          if [[ "$DRY_RUN" != 1 ]]; then
            mkdir -p "$(dirname "$train_log")"
            exec 9>"$train_log.lock"
            flock -x 9
            if [[ ! -s "$checkpoint" && -f "$train_log.done" ]]; then
              printf '' > "$train_log.done"
            fi
          fi
          ablation_step "$train_log" "${checkpoint%.pt}.json" \
            python -u scripts/risk_pipeline.py train --output "$PREDICTOR_OUTPUT" --features-dir "$RISK_OUTPUT" --layer "$layer" \
            --gamma "$gamma" --risk-horizon "$RISK_HORIZON" --seed "$SEED"
        )
        trained=1
      fi
      evaluate "$layer" "$gamma" "$alpha" "$checkpoint"
    done
  done
done
echo "Results: $OUTPUT/eval/"
echo "Summary: $OUTPUT/summary.csv"
