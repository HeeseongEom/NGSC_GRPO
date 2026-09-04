#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${SCRIPT_DIR}/../../configs/smoke.yaml"

bash "${SCRIPT_DIR}/00_validate_project.sh"
bash "${SCRIPT_DIR}/01_make_splits.sh" --force
for method in MaskCLIP SCLIP ClearCLIP NACLIP; do
  bash "${SCRIPT_DIR}/run_method.sh" "${method}" --force
done
