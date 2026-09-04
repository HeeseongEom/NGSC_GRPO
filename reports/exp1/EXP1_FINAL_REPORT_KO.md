# EXP1: GRPO-NGSC Feasibility 최종 정리

## 1. 한눈에 보는 결론

EXP1의 핵심 가설은 **source 전체에서 찾은 하나의 static action보다 image-class state에 조건화된 GRPO action이 internal 및 완전 held-out target에서 더 높은 mIoU를 보이는가**였다.

최종 결과는 이 가설을 지지하지 않는다.

- train/val에 사용되지 않은 source internal 1,306장에서 `GRPO − Source-Static` macro mIoU는 MaskCLIP `+0.007`, SCLIP `−0.264`, ClearCLIP `+0.045`, NACLIP `−0.655`%p였다.
- 완전 held-out external target 4종에서 `GRPO − Source-Static`은 MaskCLIP `−0.502`, SCLIP `+0.896`, ClearCLIP `−0.153`, NACLIP `−2.563`%p였다.
- Source-Static은 internal에서 Original보다 모든 backbone에서 높았지만 external에서는 대체로 낮았다. 따라서 **source calibration의 target transfer 문제**와 **conditional controller가 Static을 개선하지 못하는 문제**가 동시에 존재한다.
- 모든 학습은 500 updates를 정상 완료했고 reward도 소폭 상승했다. 실행 실패라기보다 state/action/reward의 식별성과 평가 정렬 문제로 판단한다.

## 2. 데이터와 평가 구분

### Source calibration

| Dataset | Modality | Train | Validation | Internal 미사용 |
|---|---|---:|---:|---:|
| BrainMRI | MRI | 48 | 16 | 536 |
| BUSI | Ultrasound | 48 | 16 | 49 |
| KiTS | CT | 48 | 16 | 425 |
| ColonDB | Endoscopy | 48 | 16 | 296 |
| 합계 | 4 modalities | 192 | 64 | 1,306 |

Internal test는 source pool에서 train/validation manifest에 한 번도 포함되지 않은 이미지만 사용했으며 경로 중복은 0이다. 동일 source dataset/class의 sample generalization을 진단하며 unseen class/modality 평가는 아니다.

### External target

| Dataset | Modality | 역할 |
|---|---|---|
| Covid-QU-Ex | X-ray | unseen target |
| MedSeg | CT | unseen target |
| HAM10000 | Dermoscopy | unseen target |
| PH2 | Dermoscopy | unseen target |

Target image, mask, state 통계 및 threshold sweep은 학습과 model selection에 사용하지 않았다.

## 3. 네 가지 setting

### Original NGSC

- legacy image-level normalization: 한 이미지의 모든 foreground class와 patch를 함께 정규화
- radiology `η=1.4`, non-radiology `η=0.8`
- affinity threshold `τ=0.6`
- suppression `γ=0.5`
- spatial term `κ_sp=0`
- 학습 없음

### Core-Fixed

- frozen BiomedCLIP, prompt ensemble, SCM, CDAM 유지
- class별 patch z-normalization
- class별 argmax seed
- action은 Original과 동일한 고정값
- 학습 및 source search 없음

Core-Fixed는 새로운 normalization/core correction의 효과와 hyperparameter optimization 효과를 분리하기 위한 baseline이다.

### Source-Static

Source train pair에서 256개 Sobol action을 평가해 backbone마다 하나의 global action을 선택했다.

| Backbone | η | τ | γ | κ_sp |
|---|---:|---:|---:|---:|
| MaskCLIP | 1.295 | 0.718 | 0.402 | 2.664 |
| SCLIP | 1.222 | 0.780 | 0.677 | 1.877 |
| ClearCLIP | 1.222 | 0.780 | 0.677 | 1.877 |
| NACLIP | 0.827 | 0.951 | 0.571 | 3.580 |

### Conditional GRPO

Source-Static action을 reference/warm start로 사용하고 image-class state에 따라 네 continuous action의 Beta distribution을 예측했다. Target에서는 학습·reward 평가·sampling 없이 Beta mean action을 한 번 적용했다.

## 4. State, action, reward와 policy

### 4.1 State

각 `(image I, class c)` pair에 대해 action과 무관한 11차원 state를 사용했다.

1. raw contrast score mean
2. raw score standard deviation
3. raw score 90% quantile
4. raw score 99% quantile
5. raw score maximum
6. normalized score maximum
7. normalized score top-10% mean
8. normalized score가 0보다 큰 patch 비율
9. normalized patch distribution entropy
10. base CDAM affinity mean
11. base affinity 90% quantile

Source-train에서만 dimension별 mean/std를 계산하고 다음과 같이 표준화했다.

\[
h=\operatorname{clip}\left(\frac{s-\mu_{train}}{\sigma_{train}+10^{-6}},-5,5\right)
\]

Dataset ID, modality ID, class ID, class embedding 및 raw patch embedding은 입력하지 않았다.

### 4.2 Continuous action

\[
a=(\eta,\tau,\gamma,\kappa_{sp})
\]

| Action | 범위 | 역할 |
|---|---:|---|
| `η` | `[0,3]` | refined score의 foreground threshold |
| `τ` | `[0,1]` | affinity가 낮다고 판단하는 cutoff |
| `γ` | `[0,1]` | low-affinity patch suppression strength |
| `κ_sp` | `[0,4]` | seed에서 멀어질수록 affinity를 감쇠하는 강도 |

공간 affinity와 refined score는 다음과 같다.

\[
A_j=R_j\exp\left(-\kappa_{sp}\lVert p_j-p_{seed}\rVert_2^2\right)
\]

\[
\widetilde\Lambda_j=
\widehat\Lambda_j\left[1-\gamma\mathbf{1}(A_j<\tau)\right]
\]

\[
\widehat M_j=\mathbf{1}(\widetilde\Lambda_j\ge\eta)
\]

### 4.3 Beta policy

96-parameter Linear-Beta controller를 사용했다.

\[
(\alpha(h),\beta(h))=\operatorname{Linear}_{11\rightarrow8}(h)
\]

각 action의 normalized variable은 독립 Beta distribution에서 sampling했다.

\[
z_k\sim\operatorname{Beta}(\alpha_k(h),\beta_k(h)),\qquad k=1,\ldots,4
\]

\[
a_k=a_{k,min}+(a_{k,max}-a_{k,min})z_k
\]

Linear weight를 0으로, bias를 Source-Static reference Beta에 맞춰 초기화했기 때문에 update 0에서는 모든 state가 동일한 Source-Static 중심 분포를 갖는다.

### 4.4 Reward

GT-positive pair에서는 hard Dice를 사용했다.

\[
R_{pos}=\frac{2\lvert\widehat M\cap Y\rvert}
{\lvert\widehat M\rvert+\lvert Y\rvert}
\]

GT-empty pair에서는 predicted foreground area를 벌점으로 사용했다.

\[
R_{neg}=1-\frac{\lvert\widehat M\rvert}{\lvert\Omega\rvert}
\]

Reward 해상도는 `224×224`였다. Hard threshold 때문에 서로 다른 sampled action이 같은 mask와 reward를 만드는 flat region이 존재한다.

### 4.5 GRPO update

- group size: 8
- states per update: 16
- updates: 500
- optimizer: Adam, learning rate `1e-3`
- reference KL coefficient: `1e-3`
- clip epsilon: `0.2`
- seeds: 0, 1, 2

같은 state에서 8개 action을 sampling하고 group reward를 표준화했다.

\[
\widehat A_g=\frac{R_g-\overline R}{\operatorname{Std}(R)+10^{-6}}
\]

구현상 `ppo_epochs=1`이고 optimizer step 전에 old/new distribution을 같은 parameter로 계산했다. 따라서 loss 평가 시 ratio가 1이어서 PPO clipping은 실질적으로 활성화되지 않았다. Gradient는 존재하지만 실제 동작은 one-step group-relative policy gradient와 reference KL에 가깝다.

## 5. 최종 macro mIoU

단위는 `%`이며 GRPO는 3 seeds 평균이다.

| Backbone | Internal Original | Internal Core | Internal Static | Internal GRPO | GRPO−Static | External Original | External Core | External Static | External GRPO | GRPO−Static |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MaskCLIP | 53.895 | 54.130 | 56.008 | 56.016 | **+0.007** | 35.296 | 34.534 | 30.930 | 30.428 | **−0.502** |
| SCLIP | 52.767 | 53.056 | 55.456 | 55.192 | **−0.264** | 32.114 | 34.044 | 29.902 | 30.798 | **+0.896** |
| ClearCLIP | 53.182 | 52.930 | 55.437 | 55.482 | **+0.045** | 34.370 | 34.751 | 30.713 | 30.560 | **−0.153** |
| NACLIP | 46.586 | 49.686 | 53.089 | 52.434 | **−0.655** | 30.967 | 35.956 | 30.632 | 28.070 | **−2.563** |

## 6. Internal dataset별 GRPO 결과

괄호는 `GRPO − Source-Static` mIoU %p다.

| Backbone | BrainMRI | BUSI | KiTS | ColonDB |
|---|---:|---:|---:|---:|
| MaskCLIP | 39.263 (+1.398) | 66.979 (−0.158) | 58.379 (−1.223) | 59.440 (+0.012) |
| SCLIP | 38.101 (−0.099) | 67.318 (−0.295) | 56.998 (−0.222) | 58.351 (−0.438) |
| ClearCLIP | 33.986 (+0.514) | 69.238 (−0.587) | 59.838 (+0.029) | 58.866 (+0.224) |
| NACLIP | 35.454 (+0.385) | 60.239 (−4.185) | 58.921 (+0.542) | 55.122 (+0.639) |

MaskCLIP처럼 dataset별 이득 방향이 충돌하거나 NACLIP BUSI처럼 특정 source domain을 크게 손상하는 현상이 확인됐다.

## 7. 가까운 external modality 결과

| Backbone | Covid X-ray Static → GRPO | MedSeg CT Static → GRPO |
|---|---:|---:|
| MaskCLIP | 55.430 → 53.464 (−1.966) | 32.335 → 31.469 (−0.867) |
| SCLIP | 49.819 → 49.373 (−0.446) | 30.007 → 29.725 (−0.282) |
| ClearCLIP | 54.000 → 53.438 (−0.563) | 30.212 → 29.719 (−0.493) |
| NACLIP | 53.731 → 51.206 (−2.525) | 28.898 → 27.408 (−1.490) |

Seen modality인 CT도 anatomy/class/dataset이 달라지면 이전되지 않았으며 X-ray에서도 네 backbone 모두 Static보다 낮았다.

## 8. 학습 신호

| Backbone | Train reward 처음 100 → 마지막 100 | Source-val 시작 → 최고 → 최종 | 최종 reference KL |
|---|---:|---:|---:|
| MaskCLIP | 0.512 → 0.524 | 0.653 → 0.654 → 0.651 | 0.256 |
| SCLIP | 0.480 → 0.491 | 0.605 → 0.606 → 0.603 | 0.174 |
| ClearCLIP | 0.515 → 0.521 | 0.647 → 0.649 → 0.648 | 0.127 |
| NACLIP | 0.429 → 0.434 | 0.614 → 0.622 → 0.619 | 0.170 |

- loss, reward, KL, gradient는 모두 finite였으며 12개 run이 전부 500 updates를 완료했다.
- MaskCLIP/SCLIP은 early validation peak 후 하락했다.
- ClearCLIP은 거의 plateau였다.
- NACLIP은 source-val reward가 올랐지만 internal mIoU는 Static보다 낮아 reward/metric mismatch를 보여준다.
- zero-advantage group 비율은 backbone 평균 약 2.8%에서 19.3%였다.

## 9. 최종 판정

1. **Source calibration은 internal sample에는 효과가 있다.** Source-Static은 Original보다 internal macro mIoU가 높다.
2. **Conditional GRPO의 추가 가치는 입증되지 않았다.** Internal에서도 Static 대비 개선이 사실상 0이거나 음수다.
3. **Target transfer도 별도 문제다.** Internal에서 좋은 Source-Static이 external target에서는 Original보다 대체로 낮다.
4. **학습 실행 자체의 실패는 아니다.** Reward와 일부 validation signal은 최적화됐지만 최종 metric으로 이어지지 않았다.
5. 가장 가능성 높은 원인은 hard reward의 flat region, state의 action 식별성 부족, 상호 중복된 action, 실제로 작동하지 않는 PPO clipping, random source-val의 약한 domain-generalization 선택력이다.

## 10. 산출물과 완결성

- 전체 최종 결과: `reports/exp1/exp1_final_results.csv`
- 모바일 macro 요약: `reports/exp1/exp1_macro_summary.csv`
- 모바일 캡처: `reports/exp1/exp1_macro_summary_mobile.png`
- internal manifest: `outputs/grpo_ngsc_feasibility_v1/internal_test/manifest.json`
- external summary: `outputs/grpo_ngsc_feasibility_v1/<METHOD>/results/summary.csv`
- internal summary: `outputs/grpo_ngsc_feasibility_v1/internal_test/<METHOD>/summary.csv`

통합 CSV는 internal/external, 4 backbones, Original/Core-Fixed/Source-Static/GRPO를 포함한다. GRPO seed별 raw row 대신 3-seed mean과 seed standard deviation만 남겨 최종 결과만 정리했다. 총 160개 aggregate row이며 각 internal run의 2,427 image-class pair와 4개 backbone의 6개 setting 파일이 모두 존재한다.
