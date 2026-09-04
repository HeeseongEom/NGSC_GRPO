#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

run_py "${SCRIPT_DIR}/build_matrix.py" --config "${CONFIG}"
run_py "${SCRIPT_DIR}/run_matrix.py" --config "${CONFIG}" --job-type ablation --gpus 0 1 --workers-per-gpu 2
run_py "${SCRIPT_DIR}/summarize.py" --config "${CONFIG}"

echo "EXP0_1 reward-ablation pipeline complete"
