#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

METHODS=(MaskCLIP SCLIP ClearCLIP NACLIP)
read -r -a SEEDS <<< "$(configured_seeds)"
for method in "${METHODS[@]}"; do
  run_ngsc evaluate --method "${method}" --baseline original_ngsc "$@"
  run_ngsc evaluate --method "${method}" --baseline core_fixed "$@"
  run_ngsc evaluate --method "${method}" --baseline source_static "$@"
  for seed in "${SEEDS[@]}"; do
    run_ngsc evaluate --method "${method}" --baseline conditional_grpo --seed "${seed}" "$@"
  done
done
