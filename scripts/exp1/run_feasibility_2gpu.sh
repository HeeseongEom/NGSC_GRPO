#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

LOG_ROOT="${PROJECT_ROOT}/outputs/grpo_ngsc_feasibility_v1/runner_logs"
mkdir -p "${LOG_ROOT}"

run_method_on_gpu() {
  local method="$1"
  local gpu="$2"
  local log_path="${LOG_ROOT}/${method}.log"
  {
    echo "[$(date --iso-8601=seconds)] START method=${method} device=cuda:${gpu}"
    bash "${SCRIPT_DIR}/run_method.sh" "${method}" --device "cuda:${gpu}"
    echo "[$(date --iso-8601=seconds)] END method=${method} device=cuda:${gpu}"
  } 2>&1 | tee "${log_path}"
}

bash "${SCRIPT_DIR}/00_validate_project.sh"
bash "${SCRIPT_DIR}/01_make_splits.sh"

set +e
run_method_on_gpu MaskCLIP 0 & pid_mask=$!
run_method_on_gpu SCLIP 1 & pid_sclip=$!
run_method_on_gpu ClearCLIP 0 & pid_clear=$!
run_method_on_gpu NACLIP 1 & pid_na=$!
wait "${pid_mask}"; status_mask=$?
wait "${pid_sclip}"; status_sclip=$?
wait "${pid_clear}"; status_clear=$?
wait "${pid_na}"; status_na=$?
set -e

if (( status_mask != 0 || status_sclip != 0 || status_clear != 0 || status_na != 0 )); then
  printf 'method failures: MaskCLIP=%d SCLIP=%d ClearCLIP=%d NACLIP=%d\n' \
    "${status_mask}" "${status_sclip}" "${status_clear}" "${status_na}" >&2
  exit 1
fi

bash "${SCRIPT_DIR}/05_summarize.sh"
