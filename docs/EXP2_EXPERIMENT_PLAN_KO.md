# EXP2: TRIPS-faithful Continuous Beta GRPO-NGSC

## 목적

EXP2는 EXP1 실패 원인을 action headroom, GRPO optimizer, reward, action set, conditional state, domain generalization으로 분리한다. 모든 action은 continuous Beta distribution으로 정의하며 categorical grid policy를 사용하지 않는다.

External target은 모든 설계와 checkpoint 선택이 끝난 8단계에서만 평가한다.

## 공통 원칙

- Frozen BiomedCLIP/SCM/CDAM feature는 EXP1 cache를 재사용한다.
- Target image, target label, target state 통계는 train/validation/model selection에 사용하지 않는다.
- Source train에서만 policy와 state standardizer를 학습한다.
- Source validation 또는 LODO held-out source domain의 mIoU로 best checkpoint를 선택한다.
- Old Beta policy를 snapshot한 뒤 같은 sampled group을 여러 PPO epoch 재사용해 probability ratio와 clipping이 실제로 작동하게 한다.
- 두 RTX 3090에서 backbone queue를 분리하고 CPU thread는 프로세스당 1개로 제한한다.

## Continuous Beta action

각 normalized action은 다음과 같다.

\[
z_j\sim\operatorname{Beta}(\alpha_j,\beta_j),\qquad z_j\in(0,1)
\]

\[
a_j=a_{j,min}+(a_{j,max}-a_{j,min})z_j
\]

| 이름 | 범위 | 의미 |
|---|---:|---|
| `η` | `[0,3]` | foreground threshold |
| `τ` | `[0,1]` | affinity cutoff |
| `γ` | `[0,1]` | PAF suppression strength |
| `κ_sp` | `[0,4]` | spatial affinity decay |
| `q` | `[0.90,0.999]` | source null-evidence quantile |

Presence action은 source-train absent pair의 evidence 분포만 사용한다.

\[
e_c=\frac{1}{\rho}\log\left(\frac1N\sum_i e^{\rho\widehat\Lambda_{c,i}}\right)
\]

\[
\delta(q)=F_{0,source}^{-1}(q),\qquad \text{eligible}(c)=\mathbf1[e_c\ge\delta(q)]
\]

## Smooth reward

Soft mask는 다음과 같다.

\[
p_i=\sigma\left(\frac{s_i-\eta}{T_\eta}\right)
\]

Positive pair의 soft IoU와 soft Youden은:

\[
IoU_{soft}=\frac{\sum_i p_i y_i}{\sum_i p_i+\sum_i y_i-\sum_i p_i y_i+\epsilon}
\]

\[
J_{soft}=TPR_{soft}-FPR_{soft}
\]

\[
R_{pixel}=0.6IoU_{soft}+0.4\frac{J_{soft}+1}{2}
\]

`q`가 포함되면 presence gate의 Brier reward를 결합한다.

\[
R=0.75R_{pixel}+0.25\left[1-(g_c-y_c)^2\right]
\]

## 8단계

### 1. 현재 4-action oracle

- action: `(η,τ,γ,κ_sp)`
- reward: EXP1 hard Dice/FP
- internal 미사용 source 전체에서 per-image-class continuous oracle 측정
- 2,048 Sobol action을 patch grid에서 선별하고 상위 32개를 224 해상도에서 재평가
- 선택 action을 원본 해상도 multiclass mIoU로 평가

### 2. TRIPS-faithful Global Beta-GRPO

- state 없음
- global `(α,β)`를 직접 최적화
- Source-Static Beta reference에서 초기화
- TRIPS 설정 arm: LR `1e-2`, group/batch `4`, KL `1e-3`, 200 updates
- corrected PPO 4 epochs

### 3. Reward ablation

- hard Dice/FP
- Soft IoU
- Soft IoU + Soft Youden
- source-val mIoU, reward tie, zero-advantage, KL, probability ratio를 비교

### 4. Action ablation

- `η`
- `η+q`
- `η+τ+γ+κ_sp`
- `η+τ+γ+κ_sp+q`

각 action set에서 continuous oracle과 Global Beta-GRPO를 비교하고 4-backbone source-val macro mIoU로 하나를 선택한다.

### 5. Conditional Beta

- EXP1 11-D class-agnostic state
- 선택된 action/reward 사용
- global reference에서 시작하는 conditional mapping 검증

### 6. LODO

각 fold에서 3 source domain으로 학습하고 나머지 1 source domain으로 best checkpoint를 선택한다. 동일 fold에서 Global과 Conditional을 직접 비교한다.

### 7. State 하나 추가

50개 prompt template별 patch localization distribution의 Jensen–Shannon disagreement를 하나의 scalar로 추가한다.

\[
U_{prompt}=\frac1M\sum_m KL(P_m\|\overline P)
\]

11-D와 12-D policy를 LODO macro mIoU로 비교한다. Raw embedding, class ID, dataset ID는 사용하지 않는다.

### 8. 최종 internal/external

- 선택된 action/reward/state를 고정
- Global 및 Conditional을 source 전체에서 3 seeds 재학습
- train/val 미사용 source internal 평가
- 마지막으로 Covid-QU-Ex, MedSeg, HAM10000, PH2 external zero-shot 평가
- EXP1 Original/Core/Source-Static과 함께 최종 CSV로 취합

## GPU/서버 안전

GPU별로 동시에 하나의 backbone process만 실행한다. Prompt disagreement 추출은 큰 image batch를 사용하되 OOM이 발생할 때만 자동으로 batch를 절반으로 줄인다. Oracle과 policy training은 모델 자체가 작아 인위적으로 GPU memory를 점유하지 않으며, CPU/BLAS thread는 1로 제한해 다른 서버 작업과의 충돌을 줄인다.
