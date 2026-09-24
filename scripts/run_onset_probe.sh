#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mllm
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/foresteer-matplotlib
export TOKENIZERS_PARALLELISM=false
python -u scripts/onset_probe.py prepare "$@"
python -u scripts/onset_probe.py extract "$@"
python -u scripts/onset_probe.py train "$@"
