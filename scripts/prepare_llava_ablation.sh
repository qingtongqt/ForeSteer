#!/usr/bin/env bash
# Shared setup is also sourced by run_llava_ablation.sh.
set -euo pipefail

ablation_init() {
  PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
  cd -- "$PROJECT_ROOT"
  PROJECT_ROOT="$PWD"
  OUTPUT="${OUTPUT:-$PROJECT_ROOT/outputs/llava_ablation}"
  MODEL_PATH="${MODEL_PATH:-$HOME/models/llava-v1.5-7b}"
  COCO_ROOT="${COCO_ROOT:-$HOME/dataset/coco}"
  TRAIN_INPUT="${TRAIN_INPUT:-$PROJECT_ROOT/outputs/llava_coco_train2014/chair_tokens.jsonl}"
  RISK_OUTPUT="${RISK_OUTPUT:-$PROJECT_ROOT/outputs/risk/llava}"
  SEED="${SEED:-42}"
  DRY_RUN="${DRY_RUN:-0}"
  read -r -a LAYER_LIST <<< "${LAYERS:-8 16 24 32}"
  [[ ${#LAYER_LIST[@]} -gt 0 ]] || { echo 'LAYERS is empty' >&2; exit 1; }
  local layer
  for layer in "${LAYER_LIST[@]}"; do
    [[ "$layer" =~ ^[1-9][0-9]*$ ]] && ((layer <= 32)) || { echo "Invalid layer: $layer" >&2; exit 1; }
  done
  if [[ "$DRY_RUN" != 1 && "${CONDA_DEFAULT_ENV:-}" != mllm ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate mllm
  fi
  export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/foresteer-matplotlib}"
}

# One marker per successful command, with an exact command signature.
ablation_step() {
  local log="$1" product="$2" signature
  shift 2
  printf -v signature '%q ' "$@"
  if [[ -s "$product" && -f "$log.done" && "$(cat "$log.done")" == "$signature" ]]; then
    echo "[skip] $product"
    return
  fi
  echo "$signature"
  [[ "$DRY_RUN" == 1 ]] && return 0
  mkdir -p -- "$(dirname -- "$log")" "$(dirname -- "$product")"
  # Invalidate success before retrying, including if a command fails midway.
  printf '' > "$log.done"
  if ! "$@" > "$log" 2>&1; then
    echo "Failed; see $log" >&2
    return 1
  fi
  [[ -s "$product" ]] || { echo "Missing output: $product" >&2; return 1; }
  printf '%s\n' "$signature" > "$log.done"
}

validate_manifest() {
  python - "$RISK_OUTPUT" "$TRAIN_INPUT" "$MODEL_PATH" "$SEED" "${1:-complete}" "${LAYER_LIST[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
risk, source, model = map(Path, sys.argv[1:4])
m = json.loads((risk / 'manifest.json').read_text())
if (m['model_name'] != 'llava' or Path(m['model_path']).resolve() != model.resolve()
        or m['seed'] != int(sys.argv[4])
        or (sys.argv[5] == 'complete' and not set(map(int, sys.argv[6:])) <= set(m['layers']))
        or m['input_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest()
        or not m['records'] or not all((risk / row['file']).is_file() for row in m['records'])):
    raise SystemExit('Prepared states mismatch or incomplete: check input/model/layers/seed; use a new RISK_OUTPUT')
if sys.argv[5] == 'missing':
    print(' '.join(map(str, sorted(set(map(int, sys.argv[6:])) - set(m['layers'])))))
PY
}

merge_features() {
  python - "$RISK_OUTPUT" "$1" <<'PY'
import json, sys
from pathlib import Path
import torch
root, pending = map(Path, sys.argv[1:])
old = json.loads((root/'manifest.json').read_text())
new = json.loads((pending/'manifest.json').read_text())
for key in ('input_sha256', 'model_path', 'model_name', 'seed'):
    if old[key] != new[key]: raise SystemExit(f'Merge mismatch: {key}')
if len(old['records']) != len(new['records']): raise SystemExit('Record count mismatch')
pairs = list(zip(old['records'], new['records']))
for a,b in pairs:
    if any(a[k] != b[k] for k in ('file', 'image_id', 'split')):
        raise SystemExit('Record order/image/split mismatch')
# Validate every file before changing existing features. Re-running after an
# interrupted merge is safe: already-added layers must match exactly.
for a,b in pairs:
    x = torch.load(root/a['file'], map_location='cpu')
    y = torch.load(pending/b['file'], map_location='cpu')
    # Historical and current replay outputs can store labels as Python lists.
    if not torch.equal(torch.as_tensor(x['labels']), torch.as_tensor(y['labels'])):
        raise SystemExit('Token labels mismatch')
    if not set(old['layers']) <= set(x['features']) or set(y['features']) != set(new['layers']):
        raise SystemExit('Missing/unexpected feature layers')
    if any(v.shape[0] != len(x['labels']) for v in y['features'].values()):
        raise SystemExit('Feature length mismatch')
    for layer in set(x['features']) & set(y['features']):
        if not torch.equal(x['features'][layer], y['features'][layer]):
            raise SystemExit(f'Existing layer {layer} differs; refusing overwrite')
    if 'terminated' in x and x['terminated'] != y['terminated']:
        raise SystemExit('Termination mismatch')
for a,b in pairs:
    file = root/a['file']
    x = torch.load(file, map_location='cpu')
    y = torch.load(pending/b['file'], map_location='cpu')
    x['features'].update(y['features'])
    x['terminated'] = y['terminated']
    temporary = file.with_suffix('.pt.tmp')
    torch.save(x, temporary); temporary.replace(file)
    a['terminated'] = y['terminated']
old['layers'] = sorted(set(old['layers']) | set(new['layers']))
temporary = root/'manifest.json.tmp'
temporary.write_text(json.dumps(old, indent=2)+'\n')
temporary.replace(root/'manifest.json')
print(f"Merged layers {old['layers']} into {root}; existing layers preserved")
PY
}

prepare_main() {
  if [[ "${1:-}" == --help ]]; then
    echo 'Usage: [TRAIN_INPUT=existing/chair_tokens.jsonl] [LAYERS="8 16 24 32"] bash scripts/prepare_llava_ablation.sh'
    echo 'Environment: OUTPUT, RISK_OUTPUT, MODEL_PATH, COCO_ROOT, SEED, DRY_RUN=1, CUDA_VISIBLE_DEVICES'
    return
  fi
  [[ $# == 0 ]] || { echo 'Unknown arguments; use --help' >&2; exit 1; }
  ablation_init
  [[ -s "$TRAIN_INPUT" ]] || { echo "Missing existing annotated data: $TRAIN_INPUT" >&2; exit 1; }
  if [[ -f "$RISK_OUTPUT/manifest.json" ]]; then
    local missing pending
    missing=$(validate_manifest missing)
    if [[ -z "$missing" ]]; then
      echo "[skip] All requested layers already exist: $RISK_OUTPUT"
      return
    fi
    local missing_layers
    read -r -a missing_layers <<< "$missing"
    pending="$RISK_OUTPUT/.pending_layers_${missing// /_}"
    echo "Extract missing layers: $missing; then merge into $RISK_OUTPUT"
    if [[ ! -f "$pending/manifest.json" ]]; then
      ablation_step "$pending/logs/extract.log" "$pending/manifest.json" \
        python -u scripts/risk_pipeline.py extract --model-name llava --model-path "$MODEL_PATH" \
        --coco-root "$COCO_ROOT" --input "$TRAIN_INPUT" --output "$pending" \
        --seed "$SEED" --layers "${missing_layers[@]}"
    fi
    if [[ "$DRY_RUN" != 1 ]]; then merge_features "$pending"; fi
  else
    ablation_step "$RISK_OUTPUT/logs/extract.log" "$RISK_OUTPUT/manifest.json" \
      python -u scripts/risk_pipeline.py extract --model-name llava --model-path "$MODEL_PATH" \
      --coco-root "$COCO_ROOT" --input "$TRAIN_INPUT" --output "$RISK_OUTPUT" \
      --seed "$SEED" --layers "${LAYER_LIST[@]}"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then prepare_main "$@"; fi
