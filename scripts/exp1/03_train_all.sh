#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

METHODS=(MaskCLIP SCLIP ClearCLIP NACLIP)
read -r -a SEEDS <<< "$(configured_seeds)"
for method in "${METHODS[@]}"; do
  run_ngsc static-search --method "${method}" "$@"
  for seed in "${SEEDS[@]}"; do
    run_ngsc train --method "${method}" --seed "${seed}" "$@"
  done
done
