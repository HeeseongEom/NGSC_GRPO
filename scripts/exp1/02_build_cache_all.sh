#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

METHODS=(MaskCLIP SCLIP ClearCLIP NACLIP)
for method in "${METHODS[@]}"; do
  run_ngsc cache --method "${method}" --role source "$@"
  run_ngsc cache --method "${method}" --role target "$@"
done
