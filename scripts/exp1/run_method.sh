#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/exp1/run_method.sh {MaskCLIP|SCLIP|ClearCLIP|NACLIP}" >&2
  exit 2
fi
METHOD="$1"
shift
case "${METHOD}" in
  MaskCLIP|SCLIP|ClearCLIP|NACLIP) ;;
  *) echo "invalid dense CLIP method: ${METHOD}" >&2; exit 2 ;;
esac

run_ngsc make-splits
run_ngsc cache --method "${METHOD}" --role source "$@"
run_ngsc cache --method "${METHOD}" --role target "$@"
run_ngsc static-search --method "${METHOD}" "$@"
read -r -a SEEDS <<< "$(configured_seeds)"
for seed in "${SEEDS[@]}"; do
  run_ngsc train --method "${METHOD}" --seed "${seed}" "$@"
done
run_ngsc evaluate --method "${METHOD}" --baseline original_ngsc "$@"
run_ngsc evaluate --method "${METHOD}" --baseline core_fixed "$@"
run_ngsc evaluate --method "${METHOD}" --baseline source_static "$@"
for seed in "${SEEDS[@]}"; do
  run_ngsc evaluate --method "${METHOD}" --baseline conditional_grpo --seed "${seed}" "$@"
done
run_ngsc summarize --method "${METHOD}"
