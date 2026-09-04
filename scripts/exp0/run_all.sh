#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

run_py "${SCRIPT_DIR}/make_splits.py" --config "${CONFIG}"
run_py "${SCRIPT_DIR}/build_matrix.py" --config "${CONFIG}"
run_py "${SCRIPT_DIR}/build_cache_all.py" --config "${CONFIG}" --gpus 0 1
run_py "${SCRIPT_DIR}/run_matrix.py" --config "${CONFIG}" --job-type upper_bound --gpus 0 1 --workers-per-gpu 3
run_py "${SCRIPT_DIR}/run_matrix.py" --config "${CONFIG}" --job-type ablation --gpus 0 1 --workers-per-gpu 3
run_py "${SCRIPT_DIR}/summarize.py" --config "${CONFIG}"

echo "EXP0 pipeline complete"
