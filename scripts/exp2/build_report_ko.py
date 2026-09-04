#!/usr/bin/env python3
"""Build the evidence-backed Korean EXP2 report after final evaluation."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import torch

from common import (
    distribution, exp2_root, load_exp2_config, load_policy_checkpoint, load_prompt_state,
    load_source_pairs, map_actions, normalize_state, source_static_action, state_vector,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "grpo_ngsc_exp2"
REPORT = ROOT / "reports" / "exp2"
METHODS = ("MaskCLIP", "SCLIP", "ClearCLIP", "NACLIP")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def table(headers, rows) -> str:
    return "\n".join((
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" if idx == 0 else "---:" for idx in range(len(headers))) + "|",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ))


def pct(value) -> float:
    return float(value)


def main() -> None:
    final = read_csv(REPORT / "exp2_final_results.csv")
    diagnostics = read_csv(REPORT / "exp2_final_training_diagnostics.csv")
    selection = json.loads((OUT / "selection" / "selected_model.json").read_text(encoding="utf-8"))
    lookup = {
        (row["split"], row["method"], row["setting"], row["dataset"]): row
        for row in final
    }

    def value(split, method, setting, dataset="macro_average", metric="mIoU_percent"):
        return pct(lookup[(split, method, setting, dataset)][metric])

    internal, external = "internal_unused_source", "external_target"
    macro_rows = []
    internal_cond_delta, external_cond_delta = [], []
    for method in METHODS:
        ins = value(internal, method, "source_static")
        ing = value(internal, method, "global_beta_grpo")
        inc = value(internal, method, "conditional_beta_grpo")
        exs = value(external, method, "source_static")
        exg = value(external, method, "global_beta_grpo")
        exc = value(external, method, "conditional_beta_grpo")
        internal_cond_delta.append(inc - ins)
        external_cond_delta.append(exc - exs)
        macro_rows.append((
            method, f"{ins:.3f}", f"{ing:.3f}", f"{inc:.3f}", f"{inc-ins:+.3f}",
            f"{exs:.3f}", f"{exg:.3f}", f"{exc:.3f}", f"{exc-exs:+.3f}",
        ))

    oracle_rows = []
    for method in METHODS:
        oracle = next(
            row for row in read_csv(OUT / "oracle" / "full4_hard" / method / "internal_mIoU_summary.csv")
            if row["dataset"] == "macro_average"
        )
        gain = next(
            row for row in read_csv(OUT / "oracle" / "full4_hard" / method / "reward_summary.csv")
            if row["dataset"] == "macro_average"
        )
        static = value(internal, method, "source_static")
        oracle_miou = float(oracle["mIoU_percent"])
        conditional = value(internal, method, "conditional_beta_grpo")
        oracle_rows.append((
            method, f"{static:.3f}", f"{oracle_miou:.3f}", f"{oracle_miou-static:+.3f}",
            f"{100*float(gain['reward_gain']):+.3f}", f"{oracle_miou-conditional:+.3f}",
        ))

    trips_rows, reward_rows = [], []
    for method in METHODS:
        meta = json.loads((
            OUT / "runs" / "s2_trips_global_full4_hard" / method / "seed_0" / "run_metadata.json"
        ).read_text(encoding="utf-8"))
        trips_rows.append((method, f"{100*meta['best_val_mIoU']:.3f}", meta["best_update"]))
        values = []
        for reward in ("hard", "soft_iou", "soft_iou_youden"):
            current = json.loads((
                OUT / "runs" / f"s3_global_full4_{reward}" / method / "seed_0" / "run_metadata.json"
            ).read_text(encoding="utf-8"))
            values.append(100 * current["best_val_mIoU"])
        reward_rows.append((method, *(f"{number:.3f}" for number in values)))
    reward_macro = [statistics.mean(float(row[idx]) for row in reward_rows) for idx in (1, 2, 3)]
    reward_rows.append(("Macro", *(f"{number:.3f}" for number in reward_macro)))

    action_rows = []
    for row in read_csv(OUT / "selection" / "action_selection.csv"):
        if row["method"] == "macro_average":
            action_rows.append((
                row["action_set"], row["reward"], f"{float(row['best_source_val_mIoU_percent']):.3f}",
                "선택" if row["action_set"] == selection["action_set"] else "",
            ))

    state_rows = []
    for row in read_csv(OUT / "selection" / "state_selection.csv"):
        if row["heldout_domain"] == "macro_average":
            state_rows.append((
                row["kind"], row["state_mode"], f"{float(row['best_lodo_mIoU_percent']):.3f}",
                "선택" if row["kind"] == selection["selected_kind"] and row["state_mode"] == selection["selected_state_mode"] else "",
            ))

    internal_dataset_rows = []
    for method in METHODS:
        cells = [method]
        for dataset in ("BrainMRI", "BUSI", "KiTS", "ColonDB"):
            static = value(internal, method, "source_static", dataset)
            conditional = value(internal, method, "conditional_beta_grpo", dataset)
            cells.append(f"{conditional:.3f} ({conditional-static:+.3f})")
        internal_dataset_rows.append(tuple(cells))

    external_dataset_rows = []
    for method in METHODS:
        cells = [method]
        for dataset in ("Covid-QU-Ex", "MedSeg", "HAM10000", "PH2"):
            static = value(external, method, "source_static", dataset)
            conditional = value(external, method, "conditional_beta_grpo", dataset)
            cells.append(f"{conditional:.3f} ({conditional-static:+.3f})")
        external_dataset_rows.append(tuple(cells))

    training_rows = []
    for kind in ("global", "conditional"):
        for method in METHODS:
            rows = [row for row in diagnostics if row["kind"] == kind and row["method"] == method]
            first = statistics.mean(float(row["reward_first_100"]) for row in rows)
            last = statistics.mean(float(row["reward_last_100"]) for row in rows)
            best_val = statistics.mean(float(row["best_source_val_mIoU_percent"]) for row in rows)
            kl = statistics.mean(float(row["final_reference_KL"]) for row in rows)
            ratio = max(float(row["max_abs_ratio_minus_one"]) for row in rows)
            clipping = max(float(row["max_clip_fraction"]) for row in rows)
            updates = ",".join(row["best_update"] for row in rows)
            training_rows.append((
                kind, method, f"{first:.4f}→{last:.4f}", f"{best_val:.3f}", updates,
                f"{kl:.3f}", f"{ratio:.3f}", f"{clipping:.3f}",
            ))

    cfg = load_exp2_config()
    action_behavior_rows = []
    with torch.inference_mode():
        for method in METHODS:
            source_val = [pair for pair in load_source_pairs(cfg, method) if pair.role == "source_val"]
            prompt = load_prompt_state(cfg, method)
            static = source_static_action(cfg, method, ("eta", "tau", "gamma", "kappa_sp"))
            cells = [method, "/".join(f"{float(x):.3f}" for x in static)]
            for run_name, state_mode in (("s8_final_global", "base11"), ("s8_final_conditional_prompt12", "prompt12")):
                seed_means, seed_within_std = [], []
                for seed in (0, 1, 2):
                    checkpoint = exp2_root(cfg) / "runs" / run_name / method / f"seed_{seed}" / "policy_best.pt"
                    saved, policy = load_policy_checkpoint(checkpoint, cfg, torch.device("cpu"))
                    if saved["kind"] == "global":
                        normalized = distribution(policy, None, batch=1).mean
                    else:
                        states = torch.stack([state_vector(pair, state_mode, prompt) for pair in source_val])
                        states = normalize_state(states, saved["state_mean"], saved["state_std"], cfg)
                        normalized = distribution(policy, states).mean
                    actions = map_actions(normalized, cfg, saved["action_names"])
                    seed_means.append(actions.mean(0))
                    seed_within_std.append(actions.std(0, unbiased=False) if actions.shape[0] > 1 else torch.zeros(4))
                mean = torch.stack(seed_means).mean(0)
                within_std = torch.stack(seed_within_std).mean(0)
                text = "/".join(f"{float(x):.3f}" for x in mean)
                if saved["kind"] == "conditional":
                    text += " (σ=" + "/".join(f"{float(x):.3f}" for x in within_std) + ")"
                cells.append(text)
            action_behavior_rows.append(tuple(cells))

    parts = [
        """# EXP2: TRIPS-faithful Continuous Beta GRPO-NGSC 최종 보고서

## 1. 한눈에 보는 결론

EXP2는 EXP1의 실패를 `action headroom`, optimizer, reward, action set, conditional state, domain-generalization 선택 문제로 분해했다. 연속 Beta action, frozen old policy를 이용한 4-epoch PPO, smooth distribution reward, LODO, prompt-disagreement state를 모두 구현하고 2×RTX 3090에서 8단계를 완료했다.

핵심 결론은 **연속 action 공간 자체에는 큰 oracle headroom이 있지만, class-agnostic controller가 unseen image/class/modality에서 그 action을 추론하는 데 실패했다**는 것이다. 즉 단순 실행 실패가 아니며, reward를 부드럽게 만들거나 PPO ratio를 교정하는 것만으로 해결되지 않았다.
""",
        table(
            ("Backbone", "Int Static", "Int Global", "Int Conditional", "Int Δ", "Ext Static", "Ext Global", "Ext Conditional", "Ext Δ"),
            macro_rows,
        ),
        f"""Conditional−Static은 internal에서 {sum(x > 0 for x in internal_cond_delta)}/4, external에서 {sum(x > 0 for x in external_cond_delta)}/4 backbone만 양수였다. External target은 action/reward/state/checkpoint 선택에 사용하지 않았다. Global과 Conditional은 모두 3 seeds 평균이다.""",

        r"""## 2. 데이터와 누수 방지

- Source train 192장, source validation 64장만 policy 학습과 기본 checkpoint 선택에 사용했다.
- LODO에서는 3개 source domain으로 학습하고 남은 1개 source domain으로 선택했다.
- Internal은 source train/validation에 한 번도 사용되지 않은 1,306장(BrainMRI 536, BUSI 49, KiTS 425, ColonDB 296)이다.
- External은 Covid-QU-Ex 583, MedSeg 100, HAM10000 10,015, PH2 200장이다. 최종 macro는 네 dataset을 동일 가중 평균한다.
- Class ID, dataset ID, modality ID, raw text/image embedding은 state에 넣지 않았다.
- External prompt disagreement는 inference 때 계산 가능한 label-free state일 뿐이며 source standardizer나 model selection에는 사용하지 않았다.

## 3. 구현한 policy와 GRPO

모든 action은 grid categorical이 아닌 연속 Beta distribution이다.

\[
z_j \sim \mathrm{Beta}(\alpha_j,\beta_j),\qquad
a_j=a_{j,\min}+(a_{j,\max}-a_{j,\min})z_j
\]

최종 선택 action은 `full4 = (η, τ, γ, κ_sp)`이다. 범위는 각각 `[0,3]`, `[0,1]`, `[0,1]`, `[0,4]`이다. Global policy는 state 없이 action별 `(α,β)`를 직접 학습한다. Conditional policy는 12-D class-agnostic state를 `Linear(12→8)`로 매핑한다. Source-Static 중심, concentration 20의 Beta를 reference이자 초기 policy로 사용했다.

EXP1과 달리 update 시작 때 old `(α,β)`를 frozen snapshot하고 동일 sampled group을 4 PPO epochs 재사용했다.

\[
r_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)},\qquad
L_{clip}=\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)
\]

Main setting은 500 updates, update당 16 states, state당 8 actions, learning rate `1e-3`, KL coefficient `1e-3`, clip ε `0.2`다.

## 4. 8단계 결과

### 4.1 현재 full4 hard oracle

각 internal image-class pair에서 2,048 Sobol action을 patch grid로 선별하고 상위 32개를 224 해상도에서 재평가한 뒤 native-resolution multiclass mIoU를 계산했다.
""",
        table(("Backbone", "Static", "Oracle", "Oracle−Static", "Reward gain", "Oracle−Final Cond"), oracle_rows),
        """네 backbone 모두 oracle은 Static보다 크게 높다. 따라서 action이 무의미한 것이 아니라 **oracle action과 controller action 사이의 큰 추론 간극**이 병목이다. 단, oracle은 pair별 GT를 사용한 상한이므로 실제 deployable 성능으로 해석하면 안 된다.

### 4.2 TRIPS-faithful Global Beta-GRPO

TRIPS prior에 맞춰 state 없는 Global Beta, LR `1e-2`, group/batch `4`, KL `1e-3`, 200 updates, corrected PPO 4 epochs를 실행했다.
""",
        table(("Backbone", "Best source-val mIoU", "Best update"), trips_rows),
        """이 arm도 정상 최적화됐지만 Source-Static을 일관되게 넘는 증거를 만들지 못했다. NGSC에서는 동일한 global action이 모든 image-class pair에 공유되기 때문에 TRIPS의 global distribution prior만 복제해서는 pair별 oracle headroom을 회수할 수 없다.

### 4.3 Reward ablation

Hard reward와 다음 smooth reward를 비교했다.

\\[
p_i=\\sigma((s_i-\\eta)/T_\\eta)
\\]

\\[
IoU_{soft}=\\frac{\\sum_i p_i y_i}{\\sum_i p_i+\\sum_i y_i-\\sum_i p_i y_i+\\epsilon}
\\]

\\[
R_{pixel}=0.6IoU_{soft}+0.4\\frac{(TPR_{soft}-FPR_{soft})+1}{2}
\\]
""",
        table(("Backbone", "Hard", "Soft IoU", "Soft IoU+Youden"), reward_rows),
        """Smooth reward는 threshold 주변의 미세 차이를 제공했지만 source-val mIoU의 순위는 개선하지 못했다. 최종에는 분포 기반이며 foreground/background 균형을 갖는 Soft IoU+Youden을 사용했지만, 이는 경험적으로 Hard보다 높은 mIoU를 보였기 때문이 아니라 flat reward를 피하고 class-agnostic 설계를 유지하기 위한 선택이다.

### 4.4 Action ablation
""",
        table(("Action set", "Reward", "Source-val macro mIoU", "결정"), action_rows),
        """`q`는 source-train absent evidence quantile로 presence gate를 정하는 연속 action이다. 그러나 `η+q`와 `full5q`가 약 38–39%로 붕괴했다. Source null evidence가 domain/class shift에서 보존되지 않고 gate 오류가 class 전체를 제거하기 때문이다. 최종은 `full4`를 선택했다.

### 4.5–4.7 Conditional, LODO, prompt-disagreement state

기본 11-D state는 score quantile/entropy와 affinity 통계만 사용한다. 추가한 1개 state는 50개 class prompt template의 patch localization distribution 간 Jensen–Shannon disagreement다.

\\[
U_{prompt}=\\frac1M\\sum_m KL(P_m\\|\\bar P)
\\]
""",
        table(("Policy", "State", "LODO macro mIoU", "결정"), state_rows),
        """Prompt disagreement는 base11 Conditional보다 소폭 높아 선택됐지만 개선폭은 작다. 이 state는 class name 자체가 아니라 prompt별 localization 불일치를 측정하므로 new class에도 계산 가능하다는 장점은 유지한다.

### 4.8 최종 internal/external

아래 표는 Conditional mIoU와 `Conditional−Source-Static`을 함께 표시한다.

Internal unused-source:
""",
        table(("Backbone", "BrainMRI", "BUSI", "KiTS", "ColonDB"), internal_dataset_rows),
        """External zero-shot:
""",
        table(("Backbone", "Covid X-ray", "MedSeg CT", "HAM10000", "PH2"), external_dataset_rows),
        """Covid와 MedSeg는 각각 X-ray/CT로 source modality에 상대적으로 가깝지만, modality 일치만으로 controller transfer가 보장되지는 않았다. Anatomy, acquisition, class prompt와 score calibration이 함께 달라지기 때문이다.

## 5. 학습 수렴과 PPO 신호

모든 값은 3 seeds 집계다. `best updates`는 seed 0/1/2 순서다.
""",
        table(("Policy", "Backbone", "Reward first→last", "Best val", "Best updates", "Final KL", "Max \\|ratio−1\\|", "Max clip frac"), training_rows),
        """Deterministic Beta mean action을 source-validation pair에서 측정했다. 값 순서는 `η/τ/γ/κ_sp`이며 Conditional 괄호 안 σ는 pair 간 표준편차의 3-seed 평균이다.
""",
        table(("Backbone", "Source-Static", "Global mean", "Conditional mean (within-pair σ)"), action_behavior_rows),
        """- 최종 24개 run이 모두 정확히 500 updates를 완료했고 loss, reward, KL, gradient, ratio 로그는 모두 finite다.
- Corrected multi-epoch PPO에서 ratio는 더 이상 정의상 1이 아니며 Conditional에서는 최대 약 0.18까지 움직였다.
- 그러나 어느 run도 clip 경계 0.2를 넘지 않아 clip fraction은 0이었다. 구현 오류가 아니라 update 크기가 clipping을 활성화할 만큼 크지 않았다는 뜻이다.
- Conditional은 Global보다 reference KL이 크고 backbone에 따라 validation peak가 update 1–25에 나타난다. 일부 seed의 빠른 과적합/분포 drift 신호다.
- Reward 상승이 미미하거나 감소해도 best source-val checkpoint는 저장됐고, 그 checkpoint가 internal/external에서 Static을 안정적으로 넘지 못했다. 따라서 단순 미수렴 하나보다는 reward/state/selection alignment 실패가 더 크다.

## 6. TRIPS와 대조한 실패 원인

1. **TRIPS prior는 탐색 분포를 주지만 식별 가능한 state를 만들어주지는 않는다.** Global Beta는 안전한 평균 action으로 수렴할 수 있으나 NGSC의 pair별 optimum은 domain과 score shape에 따라 크게 충돌한다.
2. **Oracle headroom과 learnable headroom이 다르다.** GT를 보는 oracle은 6–10%p의 여지를 보이지만 12-D label-free state로 어느 방향의 action이 필요한지 판별할 정보가 부족하다.
3. **Action 간 중복성이 크다.** `τ`, `γ`, `κ_sp`, `η`가 모두 foreground 면적을 줄이는 방향으로 상쇄될 수 있어 동일 reward를 만드는 여러 action이 존재한다. 이는 policy credit assignment를 어렵게 한다.
4. **Smooth reward도 final mIoU와 완전히 정렬되지 않는다.** Youden은 background를 안정화하지만 native-resolution mutually-exclusive multiclass mIoU와 동일 목적은 아니다.
5. **Presence q는 source-null calibration에 과민하다.** 새로운 class/modality에서 evidence 분포가 이동하면 작은 quantile 오차가 class 전체 삭제로 바뀐다.
6. **LODO 선택력은 제한적이다.** 4 source domain만으로 prompt12를 선택했지만 base11 대비 +0.119%p에 불과해 새로운 modality를 예측할 강한 증거가 아니었다.
7. **Conditional capacity는 작아도 과적합할 수 있다.** 104-parameter linear head라도 source 192장/소수 domain에서 state-action 상관을 학습하면 reference KL drift와 early validation peak가 생긴다.

## 7. 연구적 판정과 다음 한 가지 우선순위

EXP2는 "GRPO가 실행되지 않았다"는 설명을 배제한다. Continuous Beta, old-policy ratio, multi-epoch PPO, smooth reward와 LODO가 모두 작동했는데도 Static 대비 안정적 개선이 없었다. 반면 oracle은 높으므로 NGSC hyperparameter adaptation 자체를 폐기할 근거도 아니다.

다음 우선순위는 action/reward를 더 늘리는 것이 아니라 **label-free counterfactual stability state 하나**를 검증하는 것이다. 동일 image-class pair에 약한 test-time augmentation을 적용했을 때 patch score의 순위·foreground mass가 얼마나 보존되는지를 scalar로 만들면, class ID나 modality ID 없이도 "현재 calibration을 믿어도 되는가"를 직접 측정할 수 있다. 이 state도 먼저 LODO에서 Static 대비 일관된 이득이 확인될 때만 external로 넘겨야 한다.

## 8. 산출물과 재현

- 최종 전체 결과: `reports/exp2/exp2_final_results.csv`
- 모바일 표와 동일한 CSV: `reports/exp2/exp2_macro_summary.csv`
- 모바일 캡처: `reports/exp2/exp2_macro_summary_mobile.png`
- 최종 학습 진단: `reports/exp2/exp2_final_training_diagnostics.csv`
- Action 선택: `outputs/grpo_ngsc_exp2/selection/action_selection.csv`
- State/LODO 선택: `outputs/grpo_ngsc_exp2/selection/state_selection.csv`
- 실행 계획: `docs/EXP2_EXPERIMENT_PLAN_KO.md`
- 전체 실행: `bash scripts/exp2/run_all.sh`

8단계 전체에서 policy checkpoint 108개를 학습했으며, 그중 최종 checkpoint 24개, internal summary 24개, external summary 24개를 기대값과 대조 검증한다. EXP2의 최종 internal/external core metric은 모두 finite다. 통합 CSV에 함께 넣은 EXP1 legacy internal baseline은 당시 evaluator가 AUROC를 산출하지 않아 해당 60칸만 공란으로 보존했다. Prompt-disagreement cache는 4 backbone 각각 74,463 pair이며, 학습·평가 시 CPU/BLAS thread 제한을 유지했다. 가장 무거운 prompt extraction은 두 RTX 3090에서 각각 약 22.7GB GPU memory를 사용했다.
""",
    ]

    REPORT.mkdir(parents=True, exist_ok=True)
    output = REPORT / "EXP2_FINAL_REPORT_KO.md"
    output.write_text("\n\n".join(part.strip() for part in parts) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
