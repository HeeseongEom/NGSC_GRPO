# NGSC에서 GRPO-NGSC까지: 전체 실험 요약

## 1. 연구 목표와 진행 흐름

본 연구의 목표는 frozen vision-language segmentation model의 NGSC inference hyperparameter를 image-class별로 자동 조정하되, **학습에 없던 class prompt와 modality에도 과적합 없이 적용하는 것**이다.

전체 진행은 다음과 같다.

```text
Original NGSC
  → NGSC core 오류/혼입 보정(Core-Fixed)
  → source에서 하나의 최적 action 탐색(Source-Static)
  → 11-D state 기반 Conditional GRPO(EXP1)
  → internal test와 학습 신호로 실패 원인 진단
  → TRIPS식 Continuous Beta, corrected PPO, smooth reward, LODO 도입(EXP2)
  → oracle/action/reward/state ablation
  → 3-seed internal/external 최종 평가
```

사용한 dense CLIP backbone은 `MaskCLIP`, `SCLIP`, `ClearCLIP`, `NACLIP` 네 가지다.
설계는 원 NGSC 논문, TRIPS, 교수님 제안서 `Final_GRPO_NGSC.pdf`와 feasibility 설계 문서를 기준으로 했다.

## 2. 출발점: Original NGSC

Original NGSC는 frozen BiomedCLIP의 patch-class score에 SCM/CDAM affinity를 적용한 뒤 고정 hyperparameter로 foreground를 결정한다.

| Hyperparameter | 역할 | 기존값 |
|---|---|---:|
| `η` | foreground threshold | radiology 1.4, 그 외 0.8 |
| `τ` | low-affinity cutoff | 0.6 |
| `γ` | low-affinity suppression | 0.5 |
| `κ_sp` | seed와의 거리 감쇠 | 0 |

초기 구현은 한 이미지의 모든 foreground class와 patch를 함께 정규화했다. 이 경우 다른 class의 score 분포가 서로 영향을 주고 class별 seed가 불안정해질 수 있었다. 또한 dataset/modality에 따라 적합한 threshold가 다르지만 모든 image-class pair에 거의 같은 고정값을 사용했다.

## 3. EXP1: Core 보정과 Conditional GRPO feasibility

### 3.1 단계별로 추가한 것

#### Core-Fixed

- frozen backbone, prompt ensemble, SCM, CDAM은 유지했다.
- score를 class별로 독립 z-normalization했다.
- seed도 class별 argmax로 선택했다.
- action은 Original NGSC와 같은 고정값을 사용했다.

목적은 core correction 효과와 hyperparameter optimization 효과를 분리하는 것이었다.

#### Source-Static

Source train pair에서 256개 Sobol action을 평가해 backbone별로 하나의 global action을 선택했다.

| Backbone | `η` | `τ` | `γ` | `κ_sp` |
|---|---:|---:|---:|---:|
| MaskCLIP | 1.295 | 0.718 | 0.402 | 2.664 |
| SCLIP | 1.222 | 0.780 | 0.677 | 1.877 |
| ClearCLIP | 1.222 | 0.780 | 0.677 | 1.877 |
| NACLIP | 0.827 | 0.951 | 0.571 | 3.580 |

#### Conditional GRPO

각 `(image, class)` pair의 11-D class-agnostic state에서 네 continuous action의 Beta distribution을 예측했다.

- State: raw/normalized score의 평균·표준편차·quantile·maximum·entropy·positive 비율과 affinity 통계
- 제외한 정보: class ID, dataset ID, modality ID, class/raw embedding
- Policy: `Linear(11→8)`, 96 parameters
- Action: `(η, τ, γ, κ_sp)`
- Positive reward: hard Dice
- Absent-class reward: `1 − predicted foreground area`
- 학습: 500 updates, group size 8, 16 states/update, 3 seeds
- Inference: sampling 없이 Beta mean action 사용

### 3.2 평가 프로토콜

| 구분 | Dataset | 역할 |
|---|---|---|
| Source train/val | BrainMRI, BUSI, KiTS, ColonDB | policy 학습·선택 |
| Internal | 위 source에서 train/val 미사용 1,306장 | 같은 domain의 sample generalization |
| External | Covid-QU-Ex, MedSeg, HAM10000, PH2 | unseen class/dataset/modality zero-shot |

Target image, label, threshold sweep은 학습과 model selection에 사용하지 않았다.

### 3.3 EXP1 최종 결과

단위는 macro mIoU `%`이며 GRPO는 3 seeds 평균이다.

| Backbone | Int Original | Int Core | Int Static | Int GRPO | GRPO−Static | Ext Original | Ext Core | Ext Static | Ext GRPO | GRPO−Static |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MaskCLIP | 53.895 | 54.130 | 56.008 | 56.016 | +0.007 | 35.296 | 34.534 | 30.930 | 30.428 | −0.502 |
| SCLIP | 52.767 | 53.056 | 55.456 | 55.192 | −0.264 | 32.114 | 34.044 | 29.902 | 30.798 | +0.896 |
| ClearCLIP | 53.182 | 52.930 | 55.437 | 55.482 | +0.045 | 34.370 | 34.751 | 30.713 | 30.560 | −0.153 |
| NACLIP | 46.586 | 49.686 | 53.089 | 52.434 | −0.655 | 30.967 | 35.956 | 30.632 | 28.070 | −2.563 |

EXP1에서 확인한 사실은 다음과 같다.

1. Source-Static은 internal에서 Original보다 일관되게 높았다. Core 보정과 source calibration 자체는 같은 source domain에서 유효했다.
2. Source-Static은 external에서 대체로 Original보다 낮았다. Source calibration이 target으로 전이되지 않는 별도 문제가 있었다.
3. Conditional GRPO는 internal에서도 Static을 거의 개선하지 못했다. 따라서 실패 원인은 external modality shift만이 아니었다.
4. 모든 run이 정상 종료되고 reward도 일부 상승했으므로 단순 실행 실패는 아니었다.

### 3.4 EXP1 구현에서 발견한 한계

- Hard threshold 때문에 다른 sampled action이 같은 mask/reward를 만드는 flat region이 많았다.
- `ppo_epochs=1`이고 optimizer step 전 old/new policy가 같아 ratio가 항상 1이었다. 실제 동작은 clipped PPO보다 group-relative policy gradient에 가까웠다.
- 네 action이 모두 foreground 면적을 줄이는 방향으로 서로 상쇄될 수 있어 credit assignment가 어려웠다.
- Random source validation은 새로운 domain에 대한 model-selection 신호가 약했다.
- 11-D state가 optimal action의 방향을 식별할 수 있는지 확인되지 않았다.

## 4. EXP2 설계: TRIPS에 더 가깝게 재구성

EXP2는 EXP1 실패를 `action headroom`, optimizer, reward, action set, state, domain-generalization 선택 문제로 나눠 검증했다.

### 핵심 변경

- Grid categorical action이 아닌 Continuous Beta distribution 유지
- State가 없는 Global Beta arm을 추가해 TRIPS prior를 직접 검증
- Frozen old policy와 동일 sampled group을 사용한 PPO 4 epochs
- Hard reward와 Soft IoU/Soft Youden 비교
- Source-null quantile `q` presence action 검증
- 4 source domain Leave-One-Domain-Out(LODO) 선택
- 50개 prompt template의 localization Jensen–Shannon disagreement를 12번째 state로 추가
- 모든 설계가 끝난 뒤에만 external metric 평가

## 5. EXP2의 8단계 실험과 결과

| 단계 | 질문 | 핵심 결과 |
|---|---|---|
| 1. Full4 oracle | 현재 action에 실제 headroom이 있는가? | Static보다 +6.5~+9.5%p 높은 internal oracle 확인 |
| 2. TRIPS Global Beta | State 없이 distribution만 학습하면 되는가? | 정상 최적화됐지만 Static 이상의 일관된 개선 없음 |
| 3. Reward ablation | Smooth reward가 hard reward 문제를 해결하는가? | Hard 49.509, Soft IoU 49.497, Soft+Youden 49.420%; 사실상 동률 |
| 4. Action ablation | 어떤 action set이 유효한가? | `full4` 49.420%로 선택; `q` 포함 시 38~39%로 붕괴 |
| 5. Conditional base11 | Corrected PPO에서 11-D state가 유효한가? | 일부 source-val 상승은 있었지만 일반화 근거 부족 |
| 6. LODO | Domain-aware 선택이 Global/Conditional을 구분하는가? | Global 50.198, Conditional base11 50.279% |
| 7. Prompt state | class-agnostic state 하나로 개선되는가? | prompt12 50.398%; base11 대비 +0.119%p로 선택 |
| 8. Final 3-seed | 선택 조합이 internal/external에서 Static을 넘는가? | Internal 0/4, External 2/4 backbone만 양수 |

### 5.1 Oracle이 보여준 action headroom

Internal image-class pair별로 2,048 Sobol action을 선별한 뒤 GT를 사용해 상한을 측정했다.

| Backbone | Source-Static | Oracle | Oracle−Static |
|---|---:|---:|---:|
| MaskCLIP | 56.008 | 64.284 | +8.275 |
| SCLIP | 55.456 | 61.974 | +6.518 |
| ClearCLIP | 55.437 | 64.947 | +9.510 |
| NACLIP | 53.089 | 62.505 | +9.416 |

이는 action adaptation 자체가 무의미한 것이 아니라, **label-free state에서 좋은 action을 예측하는 controller가 oracle headroom을 회수하지 못한 것**임을 보여준다.

### 5.2 Reward와 action ablation

| Reward | Source-val macro mIoU |
|---|---:|
| Hard | 49.509 |
| Soft IoU | 49.497 |
| Soft IoU + Youden | 49.420 |

Smooth reward는 threshold 주변의 미세한 학습 신호를 제공했지만 final mIoU와의 정렬을 개선하지 못했다.

| Action set | Source-val macro mIoU |
|---|---:|
| `η` | 46.602 |
| `η + q` | 38.358 |
| `η + τ + γ + κ_sp` | 49.420 |
| `η + τ + γ + κ_sp + q` | 38.989 |

`q`는 source absent-class evidence의 quantile로 class 전체를 차단한다. Evidence distribution이 class/domain에 따라 이동하면서 작은 calibration 오차가 class 전체 삭제로 이어져 제외했다.

### 5.3 EXP2 최종 결과

최종 선택은 `full4 + Soft IoU/Youden + Conditional prompt12`였으며 Global과 Conditional을 각각 3 seeds 재학습했다.

| Backbone | Int Static | Int Global | Int Conditional | Cond−Static | Ext Static | Ext Global | Ext Conditional | Cond−Static |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MaskCLIP | 56.008 | 56.069 | 55.766 | −0.242 | 30.930 | 30.694 | 30.940 | +0.009 |
| SCLIP | 55.456 | 55.462 | 55.455 | −0.001 | 29.902 | 29.852 | 30.121 | +0.219 |
| ClearCLIP | 55.437 | 55.646 | 55.266 | −0.171 | 30.713 | 29.862 | 30.052 | −0.661 |
| NACLIP | 53.089 | 53.001 | 51.282 | −1.807 | 30.632 | 30.279 | 28.791 | −1.841 |

- Global Beta는 대부분 Source-Static 근처에 머물며 안정적인 평균 action을 재현했다.
- Conditional은 image-class별 action 분산을 만들었지만 internal 성능을 개선하지 못했다.
- NACLIP Conditional은 internal seed 표준편차가 0.874%p로 특히 불안정했다.
- External의 MaskCLIP `+0.009`, SCLIP `+0.219%p`는 backbone 간 재현성이 없고 크기도 작아 유효한 일반화 향상으로 보기 어렵다.

### 5.4 가까운 modality에서도 일관되지 않은 전이

Conditional−Static mIoU `%p`다.

| Backbone | Covid X-ray | MedSeg CT |
|---|---:|---:|
| MaskCLIP | +1.861 | −1.541 |
| SCLIP | +0.049 | −0.019 |
| ClearCLIP | +0.027 | −1.042 |
| NACLIP | +0.399 | −1.942 |

Covid에서는 모두 상승했지만 MedSeg에서는 모두 하락했다. Modality가 가깝더라도 anatomy, acquisition, class prompt와 score calibration 차이 때문에 적합한 action 방향이 달라졌다.

### 5.5 학습과 수렴 신호

- 최종 24개 run이 모두 500 updates를 완료했고 loss/reward/KL/gradient/ratio는 모두 finite였다.
- Corrected PPO에서 probability ratio는 실제로 변했지만 최대 `|ratio−1|≈0.185`로 clip 경계 0.2를 넘지 않아 clip fraction은 0이었다.
- Conditional은 Global보다 reference KL이 컸고 일부 seed는 validation best가 update 1~25에 나타났다.
- SCLIP Conditional은 reward가 하락하고 final reference KL이 평균 7.166까지 증가했다.
- 따라서 실패는 단순 미수렴보다는 state-action 식별 실패에 conditional drift/과적합이 더해진 것으로 판단한다.

## 6. 전체 실험에서 얻은 결론

### 성능에 대한 결론

현재 GRPO-NGSC가 Source-Static보다 높은 internal/external 일반화 성능을 보인다는 근거는 없다. EXP1과 EXP2 모두 핵심 가설을 지지하지 않았다.

### 연구적으로 얻은 수확

1. NGSC core 보정과 source global calibration은 internal source sample에는 유효하다.
2. Source에서 좋아진 calibration이 새로운 dataset/class/modality로 자동 전이되지는 않는다.
3. Continuous action 공간에는 6.5~9.5%p의 큰 oracle headroom이 있다.
4. PPO 구현을 교정해도 Conditional이 개선되지 않아 optimizer 오류만이 주원인은 아니다.
5. Smooth reward만으로는 final multiclass mIoU alignment 문제가 해결되지 않는다.
6. Source-null 기반 hard presence gate `q`는 domain generalization에 부적합하다.
7. Prompt disagreement는 class-agnostic하고 계산 가능하지만 optimal action 방향을 정하기에는 정보량이 부족하다.
8. 가장 큰 병목은 **현재 label-free state가 image-class별 optimal action을 식별하지 못하는 것**이다.

## 7. 다음 우선순위

Action이나 reward를 더 늘리기보다 **label-free counterfactual stability state 하나**를 먼저 검증하는 것이 타당하다.

- 동일 image-class pair에 약한 test-time augmentation을 적용한다.
- 변환 전후 patch-score 순위와 predicted foreground mass의 안정성을 scalar로 측정한다.
- Class/dataset/modality ID 없이 현재 calibration을 신뢰할 수 있는지를 표현한다.
- 먼저 LODO에서 Static 대비 일관된 개선이 확인될 때만 external 평가로 진행한다.

## 8. 주요 산출물

- EXP1 보고서: `reports/exp1/EXP1_FINAL_REPORT_KO.md`
- EXP1 결과: `reports/exp1/exp1_final_results.csv`
- EXP2 상세 보고서: `reports/exp2/EXP2_FINAL_REPORT_KO.md`
- EXP2 전체 결과: `reports/exp2/exp2_final_results.csv`
- EXP2 모바일 요약: `reports/exp2/exp2_macro_summary.csv`
- EXP2 학습 진단: `reports/exp2/exp2_final_training_diagnostics.csv`
- EXP2 설계: `docs/EXP2_EXPERIMENT_PLAN_KO.md`
- EXP1/EXP2 실행 코드: `scripts/exp1/`, `scripts/exp2/`

EXP2에서는 전체 108개 policy checkpoint를 학습했고, 최종 internal/external 48개 summary와 48개 per-pair 결과를 검증했다.
