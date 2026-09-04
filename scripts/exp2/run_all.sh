#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR
source "${SCRIPT_DIR}/common.sh"

LOG_ROOT="${PROJECT_ROOT}/outputs/grpo_ngsc_exp2/runner_logs"
mkdir -p "${LOG_ROOT}"

run_gpu_queue() {
  local gpu="$1"
  shift
  local methods=("$@")
  local method
  for method in "${methods[@]}"; do
    METHOD="${method}" GPU="${gpu}" bash "${CURRENT_STAGE_SCRIPT}"
  done
}

parallel_methods() {
  local label="$1"
  local body="$2"
  CURRENT_STAGE_SCRIPT="${LOG_ROOT}/.${label}_stage.sh"
  printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' \
    'source "'"${SCRIPT_DIR}"'/common.sh"' "${body}" > "${CURRENT_STAGE_SCRIPT}"
  chmod +x "${CURRENT_STAGE_SCRIPT}"
  export CURRENT_STAGE_SCRIPT
  run_gpu_queue 0 MaskCLIP ClearCLIP >"${LOG_ROOT}/${label}_gpu0.log" 2>&1 & p0=$!
  run_gpu_queue 1 SCLIP NACLIP >"${LOG_ROOT}/${label}_gpu1.log" 2>&1 & p1=$!
  wait "${p0}"
  wait "${p1}"
}

echo "[stage1] current full4 hard oracle"
parallel_methods stage1_oracle '
run_py "${SCRIPT_DIR}/oracle.py" --config "${CONFIG}" --method "${METHOD}" --action-set full4 --reward hard --device "cuda:${GPU}"
'

echo "[stage2] TRIPS-faithful global Beta-GRPO"
parallel_methods stage2_trips '
run_py "${SCRIPT_DIR}/train.py" --config "${CONFIG}" --method "${METHOD}" --run-name s2_trips_global_full4_hard --kind global --action-set full4 --reward hard --seed 0 --device "cuda:${GPU}" --optimization trips_replication
'

echo "[stage3] reward ablation"
parallel_methods stage3_reward '
for reward in hard soft_iou soft_iou_youden; do
  run_py "${SCRIPT_DIR}/train.py" --config "${CONFIG}" --method "${METHOD}" --run-name "s3_global_full4_${reward}" --kind global --action-set full4 --reward "${reward}" --seed 0 --device "cuda:${GPU}" --optimization main_optimization
done
'

echo "[stage4] continuous action oracle and global ablation"
parallel_methods stage4_action '
run_py "${SCRIPT_DIR}/oracle.py" --config "${CONFIG}" --method "${METHOD}" --action-set eta --reward soft_iou_youden --device "cuda:${GPU}"
run_py "${SCRIPT_DIR}/oracle.py" --config "${CONFIG}" --method "${METHOD}" --action-set eta_q --reward soft_full --device "cuda:${GPU}"
run_py "${SCRIPT_DIR}/oracle.py" --config "${CONFIG}" --method "${METHOD}" --action-set full4 --reward soft_iou_youden --device "cuda:${GPU}"
run_py "${SCRIPT_DIR}/oracle.py" --config "${CONFIG}" --method "${METHOD}" --action-set full5q --reward soft_full --device "cuda:${GPU}"
run_py "${SCRIPT_DIR}/train.py" --config "${CONFIG}" --method "${METHOD}" --run-name s4_global_eta --kind global --action-set eta --reward soft_iou_youden --seed 0 --device "cuda:${GPU}" --optimization main_optimization
run_py "${SCRIPT_DIR}/train.py" --config "${CONFIG}" --method "${METHOD}" --run-name s4_global_eta_q --kind global --action-set eta_q --reward soft_full --seed 0 --device "cuda:${GPU}" --optimization main_optimization
run_py "${SCRIPT_DIR}/train.py" --config "${CONFIG}" --method "${METHOD}" --run-name s4_global_full4 --kind global --action-set full4 --reward soft_iou_youden --seed 0 --device "cuda:${GPU}" --optimization main_optimization
run_py "${SCRIPT_DIR}/train.py" --config "${CONFIG}" --method "${METHOD}" --run-name s4_global_full5q --kind global --action-set full5q --reward soft_full --seed 0 --device "cuda:${GPU}" --optimization main_optimization
'
run_py "${SCRIPT_DIR}/select_models.py" action --config "${CONFIG}"
read -r ACTION_SET REWARD < <(run_py -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x["action_set"],x["reward"])' "${PROJECT_ROOT}/outputs/grpo_ngsc_exp2/selection/selected_action.json")
export ACTION_SET REWARD

echo "[stage5] all-source conditional base11"
parallel_methods stage5_conditional '
run_py "${SCRIPT_DIR}/train.py" --config "${CONFIG}" --method "${METHOD}" --run-name s5_conditional_base11 --kind conditional --action-set "${ACTION_SET}" --reward "${REWARD}" --seed 0 --device "cuda:${GPU}" --optimization main_optimization --state-mode base11
'

echo "[stage6] global-vs-conditional LODO"
parallel_methods stage6_lodo '
for domain in BrainMRI BUSI KiTS ColonDB; do
  run_py "${SCRIPT_DIR}/train.py" --config "${CONFIG}" --method "${METHOD}" --run-name "s6_lodo_global_base11_${domain}" --kind global --action-set "${ACTION_SET}" --reward "${REWARD}" --seed 0 --device "cuda:${GPU}" --optimization main_optimization --heldout-domain "${domain}" --state-mode base11
  run_py "${SCRIPT_DIR}/train.py" --config "${CONFIG}" --method "${METHOD}" --run-name "s6_lodo_conditional_base11_${domain}" --kind conditional --action-set "${ACTION_SET}" --reward "${REWARD}" --seed 0 --device "cuda:${GPU}" --optimization main_optimization --heldout-domain "${domain}" --state-mode base11
done
'

echo "[stage7a] prompt disagreement cache"
parallel_methods stage7_prompt_cache '
if [[ "${METHOD}" == "ClearCLIP" || "${METHOD}" == "NACLIP" ]]; then batch_size=1280; else batch_size=512; fi
run_py "${SCRIPT_DIR}/prompt_disagreement.py" --config "${CONFIG}" --method "${METHOD}" --device "cuda:${GPU}" --batch-size "${batch_size}"
'

echo "[stage7b] prompt12 LODO"
parallel_methods stage7_prompt_lodo '
for domain in BrainMRI BUSI KiTS ColonDB; do
  run_py "${SCRIPT_DIR}/train.py" --config "${CONFIG}" --method "${METHOD}" --run-name "s7_lodo_conditional_prompt12_${domain}" --kind conditional --action-set "${ACTION_SET}" --reward "${REWARD}" --seed 0 --device "cuda:${GPU}" --optimization main_optimization --heldout-domain "${domain}" --state-mode prompt12
done
'
run_py "${SCRIPT_DIR}/select_models.py" state --config "${CONFIG}"
read -r STATE_MODE < <(run_py -c 'import json,sys; x=json.load(open(sys.argv[1])); print(x["conditional_state_mode"])' "${PROJECT_ROOT}/outputs/grpo_ngsc_exp2/selection/selected_model.json")
export STATE_MODE

echo "[stage8a] final 3-seed global and conditional training"
parallel_methods stage8_train '
for seed in 0 1 2; do
  run_py "${SCRIPT_DIR}/train.py" --config "${CONFIG}" --method "${METHOD}" --run-name s8_final_global --kind global --action-set "${ACTION_SET}" --reward "${REWARD}" --seed "${seed}" --device "cuda:${GPU}" --optimization main_optimization --state-mode base11
  run_py "${SCRIPT_DIR}/train.py" --config "${CONFIG}" --method "${METHOD}" --run-name "s8_final_conditional_${STATE_MODE}" --kind conditional --action-set "${ACTION_SET}" --reward "${REWARD}" --seed "${seed}" --device "cuda:${GPU}" --optimization main_optimization --state-mode "${STATE_MODE}"
done
'

echo "[stage8b] final internal then external evaluation"
parallel_methods stage8_eval '
args=()
for seed in 0 1 2; do
  args+=(--checkpoint "${PROJECT_ROOT}/outputs/grpo_ngsc_exp2/runs/s8_final_global/${METHOD}/seed_${seed}/policy_best.pt")
  args+=(--checkpoint "${PROJECT_ROOT}/outputs/grpo_ngsc_exp2/runs/s8_final_conditional_${STATE_MODE}/${METHOD}/seed_${seed}/policy_best.pt")
done
run_py "${SCRIPT_DIR}/evaluate_many.py" --config "${CONFIG}" "${args[@]}" --split internal --device "cuda:${GPU}" --auroc-workers 6
run_py "${SCRIPT_DIR}/evaluate_many.py" --config "${CONFIG}" "${args[@]}" --split external --device "cuda:${GPU}" --auroc-workers 6
'

echo "[report] final CSV, mobile snapshot, Korean report, and email draft"
run_py "${SCRIPT_DIR}/build_final_report.py"
run_py "${SCRIPT_DIR}/build_report_ko.py"
run_py "${SCRIPT_DIR}/build_email_draft.py"

echo "exp2 pipeline complete"
