#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/00_validate_project.sh"
bash "${SCRIPT_DIR}/01_make_splits.sh"
bash "${SCRIPT_DIR}/02_build_cache_all.sh"
bash "${SCRIPT_DIR}/03_train_all.sh"
bash "${SCRIPT_DIR}/04_test_all.sh"
bash "${SCRIPT_DIR}/05_summarize.sh"
