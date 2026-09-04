#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_PYTHON="/data2/hseum/.conda/envs/medclipsamv2/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/feasibility.yaml}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

run_ngsc() {
  "${PYTHON_BIN}" -m ngsc_grpo --config "${CONFIG}" "$@"
}

configured_seeds() {
  "${PYTHON_BIN}" -c 'import sys,yaml; print(*yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["grpo"]["seeds"])' "${CONFIG}"
}
