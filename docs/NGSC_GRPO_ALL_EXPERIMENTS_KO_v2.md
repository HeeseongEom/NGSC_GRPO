# NGSC에서 GRPO-NGSC까지: 전체 실험 요약 v2

## 1. 연구 질문과 결론

본 연구는 frozen BiomedCLIP segmentation backbone의 NGSC 후처리 hyperparameter를 image-class pair마다 GRPO controller가 선택하면, 학습에 없던 class prompt와 modality에서도 고정 hyperparameter보다 좋아지는지를 검증했다. Backbone 자체는 학습하지 않았다.

결론은 다음과 같다.

- Source에서 고정 action 하나를 고른 `Source-Static`은 internal에서 강한 baseline이었다.
- EXP1과 EXP2 Conditional GRPO는 Source-Static을 internal/external에서 일관되게 넘지 못했다.
- GT를 보는 Pair-reward Oracle은 큰 개선 여지를 보였지만, 현재 label-free state는 좋은 action을 식별하지 못했다.
- 따라서 실패의 주원인은 실행 오류나 단순 미수렴보다 `state → optimal action`의 식별력 부족으로 판단한다.

성능 단위는 dataset별 multiclass mIoU `%`다. 각 foreground class IoU와 background IoU를 동일 가중 평균하므로, foreground가 좋아져도 background가 더 나빠지면 mIoU는 하락할 수 있다.

---

## 2. 학습·검증·평가 데이터

### 2.1 Source data 선정

Source는 MRI, ultrasound, CT, endoscopy의 네 domain으로 구성했다. Dataset 크기가 policy 학습 비중을 결정하지 않도록 각 dataset에서 최대 64장을 class-stratified로 선정한 뒤 48장을 train, 16장을 validation으로 사용했다. Split seed는 `2027`이다.

| Dataset | Modality | Foreground class | Train | Val | Internal 미사용 |
|---|---|---|---:|---:|---:|
| BrainMRI | MRI | glioma, meningioma, pituitary tumor | 48 | 16 | 536 |
| BUSI | Ultrasound | benign, malignant tumor | 48 | 16 | 49 |
| KiTS | CT | kidney tumor | 48 | 16 | 425 |
| ColonDB | Endoscopy | polyp | 48 | 16 | 296 |
| 합계 | 4 modalities | 7 prompts | 192 | 64 | 1,306 |

Train의 class 구성은 BrainMRI `16/16/16`, BUSI `benign 24/malignant 24`, KiTS 48, ColonDB 48이다. Validation은 BrainMRI `6/5/5`, BUSI `benign 9/malignant 7`, KiTS 16, ColonDB 16이다.

Controller의 학습 단위는 image가 아니라 `(image, prompted class)` pair다. 예를 들어 BrainMRI 한 장은 세 tumor prompt와 각각 짝지어 3개 pair가 된다. Source train은 총 336 pair이며, 실제 class가 있는 positive pair와 없는 absent pair를 모두 포함한다.

### 2.2 Internal과 external의 의미

**Internal test**는 같은 source dataset에서 train/validation에 쓰지 않은 1,306장이다. 같은 class와 modality의 새로운 image에 적용되는지를 본다. 단, BUSI internal 49장은 모두 benign이므로 BUSI malignant 일반화 결과로 해석하면 안 된다.

**External test**는 학습·설정 선택에 사용하지 않은 다음 네 dataset 전체다. Label은 최종 평가 때만 읽었다.

| Dataset | Modality | Foreground class | 평가 이미지 | Source와의 차이 |
|---|---|---|---:|---|
| Covid-QU-Ex | X-ray | covid lungs | 583 | 새 dataset/class/modality |
| MedSeg | CT | ground glass, consolidation, pleural effusion | 100 | CT는 같지만 장기·질환 class가 다름 |
| HAM10000 | Dermoscopy | 7개 피부 병변 | 10,015 | 새 dataset/class/modality |
| PH2 | Dermoscopy | common nevus, atypical nevus, melanoma | 200 | 새 dataset/class/modality |

Source와 external은 dataset, sample, foreground class 이름이 겹치지 않는다. External 총 10,898장의 결과를 평균할 때는 이미지 수가 아니라 네 dataset을 동일 가중한다.

---

## 3. 실험 설정 용어 사전

| 문서의 설정명 | 정확히 무엇을 하는가 | Test 시 action |
|---|---|---|
| `Original` | 기존 NGSC. 한 image의 모든 class/patch를 함께 정규화하고 기존 고정값 사용 | dataset의 radiology flag에 따라 정해진 고정값 |
| `Core-Fixed` | class별 score 정규화와 class별 seed 선택으로 core만 보정. 학습 없음 | Original과 같은 고정값 |
| `Source-Static` 또는 `Static` | Source-train에서 256개 Sobol 후보를 비교해 backbone별 action 하나를 선택. GRPO 아님 | 모든 image/class에 같은 backbone별 action |
| `EXP1 Cond` | 11-D state에서 pair별 Beta action을 내는 Conditional GRPO | pair마다 Beta mean action |
| `Pair-reward Oracle` | Internal GT를 보며 image-class pair마다 2,048개 후보 중 action을 별도 선택 | pair마다 GT로 선택한 action; 배포 불가 |
| `EXP2 Global` | State 없이 backbone별 Beta distribution 하나를 GRPO로 학습 | 모든 pair에 같은 Beta mean action |
| `EXP2 Cond` | Prompt12 state에서 pair별 Beta action을 내는 corrected Conditional GRPO | pair마다 Beta mean action |

`Static`은 “아무 후처리도 하지 않음”이 아니다. 항상 `Source-Static`, 즉 source label로 미리 골라 둔 강한 고정-action baseline을 뜻한다.

### 3.1 NGSC action 이름

| 코드명 | 의미 | 범위 | 큰 값의 일반적 효과 |
|---|---|---:|---|
| `eta` | foreground score threshold | 0–3 | foreground를 적게 선택 |
| `tau` | low-affinity 판정 기준 | 0–1 | 더 많은 patch가 억제 대상이 될 수 있음 |
| `gamma` | low-affinity patch 억제 강도 | 0–1 | 억제 대상 score를 더 낮춤 |
| `kappa_sp` | class seed와의 거리 감쇠 강도 | 0–4 | seed에서 먼 patch의 affinity를 더 낮춤 |
| `q` | absent evidence 분포의 quantile로 만든 class-presence gate | 0.90–0.999 | EXP2 ablation에서만 사용 후 제외 |

Source-Static으로 실제 선택된 `eta/tau/gamma/kappa_sp`는 다음과 같다.

| Backbone | `eta` | `tau` | `gamma` | `kappa_sp` |
|---|---:|---:|---:|---:|
| MaskCLIP | 1.295 | 0.718 | 0.402 | 2.664 |
| SCLIP | 1.222 | 0.780 | 0.677 | 1.877 |
| ClearCLIP | 1.222 | 0.780 | 0.677 | 1.877 |
| NACLIP | 0.827 | 0.951 | 0.571 | 3.580 |

### 3.2 State와 reward

`base11` state는 class/dataset/modality ID 없이 다음 통계만 사용한다.

- Raw patch score: mean, standard deviation, 90% quantile, 99% quantile, maximum
- Class별 normalized score: maximum, top-10% mean, positive-patch ratio, entropy
- Seed affinity: mean, 90% quantile

`prompt12`는 base11에 **50개 prompt template의 localization disagreement** 한 값을 더한다. 문장 표현이 바뀔 때 localization 분포가 많이 달라질수록 큰 값이다. 새 class에서도 label 없이 계산할 수 있다.

Reward 용어는 다음과 같다.

- `Hard`: positive pair는 thresholded mask의 Dice, absent pair는 `1 − predicted foreground 비율`.
- `Soft IoU`: 0/1 mask로 끊기 전에 foreground 확률을 사용해 교집합과 합집합을 계산한 IoU. Threshold 주변의 작은 action 차이에도 reward가 변한다.
- `Soft Youden`: soft true-positive rate에서 soft false-positive rate를 뺀 값. Foreground 검출과 background false positive를 함께 본다.
- EXP2 final reward: `0.6 × Soft IoU + 0.4 × normalized Soft Youden`.

---

## 4. EXP1에서 EXP2까지 무엇을 검증했는가

### 4.1 EXP1

EXP1 controller는 `Linear(11→8)`, 96 parameters다. Source-Static을 중심으로 한 Beta distribution에서 시작해 500 updates, 16 states/update, 8 actions/state, learning rate `1e-3`, 3 seeds로 학습했다.

모든 run은 종료됐지만 `ppo_epochs=1`이고 optimizer step 전 old/new policy가 같아 PPO ratio가 항상 1이었다. 따라서 clipping이 작동하는 PPO라기보다 group-relative policy gradient에 가까웠다. Internal에서도 Static 개선이 거의 없어 external shift만의 문제는 아니었다.

### 4.2 EXP2의 8단계

| 단계 | 확인한 질문 | 결과와 결정 |
|---:|---|---|
| 1. Pair-reward Oracle | Action 공간에 개선 여지가 있는가? | 대부분 dataset에서 큰 여지 확인. 단, 최종 mIoU의 엄밀한 상한은 아님 |
| 2. Global Beta | TRIPS식 state-free distribution만으로 충분한가? | 정상 학습했지만 final에서 Static 근처 |
| 3. Reward ablation | Soft reward가 hard reward보다 나은가? | Source-val macro: Hard 49.509, Soft IoU 49.497, Soft+Youden 49.420; 개선 없음 |
| 4. Action ablation | 어떤 action을 남길 것인가? | `eta` 46.602, `eta+q` 38.358, `full4` 49.420, `full4+q` 38.989; full4 선택 |
| 5. Corrected base11 | PPO 구현 교정만으로 해결되는가? | 해결되지 않음 |
| 6. LODO | Dataset 단위 선택에서 conditional이 나은가? | Global 50.198, base11 50.279, prompt12 50.398 |
| 7. Prompt12 | New class에도 계산 가능한 state 하나가 도움 되는가? | base11 대비 +0.119%p로 prompt12 선택, 효과는 작음 |
| 8. Final 3-seed | Internal/external에서 Static을 넘는가? | Internal 0/4, external 2/4 backbone만 macro 양수 |

LODO는 controller update와 state 표준화에서 held-out source dataset을 제외하고, 그 dataset의 64장 calibration subset으로 선택했다. 다만 초기/reference Source-Static action은 네 source 전체로 이미 선정됐으므로 완전히 독립적인 from-scratch domain holdout은 아니다.

`q`는 absent evidence cutoff가 조금만 틀려도 class mask 전체를 제거해 성능이 38–39%로 붕괴했으므로 제외했다. Soft reward는 flatness는 줄였지만 실제 mIoU를 높이지는 않았다.

---

## 5. Pair-reward Oracle을 정확히 해석하는 법

### 5.1 Oracle 표의 Static은 무엇인가

`Static`은 앞에서 정의한 `Source-Static`이다. Source-train 전체를 이용해 backbone마다 한 번 선택한 고정 action을 internal 모든 image-class pair에 그대로 적용한 결과다. Oracle이 넘어야 할 비교 기준이다.

`Pair-reward Oracle`은 internal의 각 image-class pair에서 GT를 보며 서로 다른 action을 선택한다. 2,048개 Sobol 후보를 patch grid에서 1차 평가하고, 상위 32개를 224×224에서 다시 평가했다. Positive pair의 선택 기준은 hard Dice이고, absent pair는 predicted foreground 면적이다.

### 5.2 왜 Oracle인데 KiTS에서 Static보다 낮은가

이 Oracle은 **224×224 pair reward를 최대화**하지만, 표의 성능은 **native-resolution dataset mIoU**다. 두 목적함수가 다르므로 최종 mIoU가 항상 높다는 보장은 없다.

KiTS는 foreground class가 하나이므로 최종 mIoU가 `kidney tumor IoU`와 `background IoU`를 1:1로 평균한다. Oracle은 foreground Dice를 높이는 action을 골랐지만 MaskCLIP과 SCLIP에서는 foreground 증가보다 background 감소가 더 컸다.

| Backbone | 설정 | Foreground IoU | Background IoU | 최종 mIoU |
|---|---|---:|---:|---:|
| MaskCLIP | Static | 21.298 | 97.906 | 59.602 |
|  | Pair-reward Oracle | 26.688 | 89.960 | 58.324 |
| SCLIP | Static | 17.679 | 96.762 | 57.221 |
|  | Pair-reward Oracle | 21.339 | 85.545 | 53.442 |
| ClearCLIP | Static | 22.260 | 97.356 | 59.808 |
|  | Pair-reward Oracle | 28.511 | 93.119 | 60.815 |
| NACLIP | Static | 20.607 | 96.152 | 58.379 |
|  | Pair-reward Oracle | 30.061 | 95.102 | 62.581 |

MaskCLIP은 foreground가 `+5.390%p` 올랐지만 background가 `−7.946%p` 내려 최종 mIoU가 `−1.278%p`가 됐다. SCLIP도 foreground `+3.660%p`, background `−11.216%p`라 최종 `−3.779%p`였다. 반면 ClearCLIP과 NACLIP은 foreground 이득이 더 커 최종 mIoU도 상승했다.

따라서 이 실험은 `Oracle upper bound`보다 **GT 기반 pair-reward action headroom**이라고 부르는 것이 정확하다. 향후 진짜 상한을 재려면 action 선택 단계부터 native-resolution foreground/background mIoU 또는 최종 multiclass metric을 직접 최적화해야 한다.

### 5.3 Dataset별 Pair-reward Oracle 결과

| Backbone | Dataset | Static | Pair-reward Oracle | 차이 |
|---|---|---:|---:|---:|
| MaskCLIP | BrainMRI | 37.865 | 56.349 | +18.484 |
|  | BUSI | 67.138 | 77.809 | +10.671 |
|  | KiTS | 59.602 | 58.324 | −1.278 |
|  | ColonDB | 59.428 | 64.652 | +5.224 |
| SCLIP | BrainMRI | 38.201 | 57.095 | +18.894 |
|  | BUSI | 67.613 | 75.154 | +7.541 |
|  | KiTS | 57.221 | 53.442 | −3.779 |
|  | ColonDB | 58.790 | 62.205 | +3.415 |
| ClearCLIP | BrainMRI | 33.472 | 58.282 | +24.810 |
|  | BUSI | 69.825 | 77.508 | +7.683 |
|  | KiTS | 59.808 | 60.815 | +1.007 |
|  | ColonDB | 58.642 | 63.185 | +4.544 |
| NACLIP | BrainMRI | 35.069 | 53.440 | +18.371 |
|  | BUSI | 64.424 | 74.009 | +9.585 |
|  | KiTS | 58.379 | 62.581 | +4.202 |
|  | ColonDB | 54.483 | 59.991 | +5.508 |

---

## 6. 개별 dataset 최종 결과

`EXP1 Cond`와 EXP2 두 설정은 3-seed 평균이다. `Δ`는 `EXP2 Cond − Static`으로, 최종 conditional controller가 고정 baseline을 얼마나 바꿨는지 나타낸다.

### 6.1 Internal: source에서 train/validation에 쓰지 않은 image

| Backbone | Dataset | Original | Core-Fixed | Static | EXP1 Cond | EXP2 Global | EXP2 Cond | Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MaskCLIP | BrainMRI | 36.578 | 34.791 | 37.865 | 39.263 | 38.211 | 39.061 | +1.196 |
|  | BUSI | 63.568 | 66.318 | 67.138 | 66.979 | 67.023 | 65.310 | −1.828 |
|  | KiTS | 57.998 | 57.985 | 59.602 | 58.379 | 59.694 | 58.644 | −0.958 |
|  | ColonDB | 57.437 | 57.425 | 59.428 | 59.440 | 59.347 | 60.049 | +0.621 |
| SCLIP | BrainMRI | 32.937 | 35.409 | 38.201 | 38.101 | 38.201 | 38.145 | −0.055 |
|  | BUSI | 67.941 | 66.665 | 67.613 | 67.318 | 67.624 | 67.639 | +0.027 |
|  | KiTS | 56.104 | 56.088 | 57.221 | 56.998 | 57.216 | 57.221 | +0.001 |
|  | ColonDB | 54.087 | 54.062 | 58.790 | 58.351 | 58.805 | 58.814 | +0.024 |
| ClearCLIP | BrainMRI | 29.247 | 30.564 | 33.472 | 33.986 | 34.888 | 35.381 | +1.909 |
|  | BUSI | 71.413 | 69.137 | 69.825 | 69.238 | 69.650 | 66.593 | −3.232 |
|  | KiTS | 57.380 | 57.363 | 59.808 | 59.838 | 60.030 | 60.148 | +0.340 |
|  | ColonDB | 54.688 | 54.658 | 58.642 | 58.866 | 58.015 | 58.940 | +0.298 |
| NACLIP | BrainMRI | 25.504 | 29.307 | 35.069 | 35.454 | 34.866 | 35.314 | +0.245 |
|  | BUSI | 58.285 | 66.992 | 64.424 | 60.239 | 63.651 | 55.770 | −8.654 |
|  | KiTS | 52.753 | 52.701 | 58.379 | 58.921 | 58.850 | 57.818 | −0.562 |
|  | ColonDB | 49.801 | 49.745 | 54.483 | 55.122 | 54.635 | 56.224 | +1.741 |

Internal에서 EXP2 Conditional은 BrainMRI와 ColonDB 일부 backbone에서 좋아졌지만, BUSI에서 크게 하락했다. 네 dataset 평균으로는 네 backbone 모두 Static을 넘지 못했다.

### 6.2 External: 새 dataset과 class prompt

| Backbone | Dataset | Original | Core-Fixed | Static | EXP1 Cond | EXP2 Global | EXP2 Cond | Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MaskCLIP | Covid-QU-Ex | 58.500 | 58.530 | 55.430 | 53.464 | 54.932 | 57.291 | +1.861 |
|  | MedSeg | 32.891 | 33.293 | 32.335 | 31.469 | 32.291 | 30.795 | −1.541 |
|  | HAM10000 | 20.886 | 19.302 | 15.567 | 15.407 | 15.364 | 15.080 | −0.487 |
|  | PH2 | 28.907 | 27.011 | 20.388 | 21.373 | 20.189 | 20.593 | +0.204 |
| SCLIP | Covid-QU-Ex | 53.440 | 53.459 | 49.819 | 49.373 | 49.773 | 49.869 | +0.049 |
|  | MedSeg | 29.035 | 32.436 | 30.007 | 29.725 | 29.982 | 29.988 | −0.019 |
|  | HAM10000 | 13.023 | 18.784 | 16.284 | 17.302 | 16.257 | 16.455 | +0.170 |
|  | PH2 | 32.958 | 31.499 | 23.497 | 26.792 | 23.396 | 24.172 | +0.676 |
| ClearCLIP | Covid-QU-Ex | 59.699 | 59.729 | 54.000 | 53.438 | 52.854 | 54.027 | +0.027 |
|  | MedSeg | 29.249 | 31.960 | 30.212 | 29.719 | 30.005 | 29.170 | −1.042 |
|  | HAM10000 | 14.493 | 19.706 | 16.444 | 16.414 | 15.679 | 15.819 | −0.625 |
|  | PH2 | 34.039 | 27.607 | 22.196 | 22.670 | 20.910 | 21.191 | −1.005 |
| NACLIP | Covid-QU-Ex | 58.442 | 58.478 | 53.731 | 51.206 | 53.094 | 54.130 | +0.399 |
|  | MedSeg | 24.520 | 28.844 | 28.898 | 27.408 | 28.899 | 26.956 | −1.942 |
|  | HAM10000 | 10.958 | 20.904 | 15.650 | 13.489 | 15.455 | 14.745 | −0.905 |
|  | PH2 | 29.947 | 35.597 | 24.251 | 20.175 | 23.668 | 19.335 | −4.916 |

Covid에서는 EXP2 Conditional이 Static을 네 backbone 모두 조금 회복했지만, MedSeg CT에서는 네 backbone 모두 하락했다. 따라서 source에 CT가 있었다는 modality 유사성만으로 controller 전이가 보장되지 않는다. Anatomy, class prompt, acquisition과 score calibration이 함께 달라지기 때문이다.

---

## 7. 학습 이상 여부와 연구적 판정

EXP2 final 24개 run은 모두 500 updates를 완료했고 loss, reward, KL, gradient와 PPO ratio가 finite였다. Corrected PPO의 최대 `|ratio−1|`은 Conditional에서 약 0.12–0.19로 실제 변화했으며, 실행 실패는 아니었다.

다만 다음 과적합·drift 신호가 있었다.

- SCLIP Conditional reward는 `0.5511→0.5307`로 하락했고 reference KL은 평균 `7.166`까지 증가했다.
- SCLIP Conditional의 best validation update는 seed별 `25/1/1`로 매우 빨랐다.
- NACLIP Conditional은 seed별 best 시점이 불안정했고 BUSI에서 `−8.654%p` 하락했다.

그렇지만 과적합만이 원인은 아니다. Global Beta, corrected PPO, smooth reward와 LODO 모두 Static 대비 일관된 개선을 만들지 못했다. Pair-reward Oracle의 여지는 존재하므로 action 공간보다 **현재 label-free state가 action 방향을 구분하지 못하는 것**, 그리고 **pair reward와 최종 multiclass mIoU가 완전히 정렬되지 않는 것**이 핵심 병목이다.

동료 연구원과 다음 실험을 논의할 때는 우선순위를 다음 두 가지로 두는 것이 타당하다.

1. Oracle을 native-resolution foreground/background mIoU 기준으로 다시 측정해 실제 action upper bound를 확정한다.
2. Action이나 reward 항을 더 늘리기 전에, 약한 test-time augmentation 전후 patch-score 순위와 foreground mass 안정성을 나타내는 class-agnostic state 하나가 LODO에서 일관된 이득을 만드는지 검증한다.

---

## 8. 주요 산출물

- EXP1 상세 보고서: `reports/exp1/EXP1_FINAL_REPORT_KO.md`
- EXP1 결과: `reports/exp1/exp1_final_results.csv`
- EXP2 상세 보고서: `reports/exp2/EXP2_FINAL_REPORT_KO.md`
- EXP2 전체 결과: `reports/exp2/exp2_final_results.csv`
- EXP2 학습 진단: `reports/exp2/exp2_final_training_diagnostics.csv`
- Oracle 결과: `outputs/grpo_ngsc_exp2/oracle/full4_hard/`
- 실행 코드: `scripts/exp1/`, `scripts/exp2/`
