# EXP0: dataset별 기초 NGSC-GRPO 실험

## 목적

EXP0는 여러 source dataset을 섞지 않고, 한 번에 한 dataset만 학습해 다음 세 질문을 순서대로 확인한다.

1. Dataset별 고정 NGSC action을 grid search하면 fixed NGSC보다 얼마나 좋아질 수 있는가?
2. Image 정보를 보지 않는 Global GRPO가 그 고정 action을 학습할 수 있는가?
3. SCM 이전의 prompt-conditioned image representation을 보는 작은 CNN-GRPO가 image-class별 action을 더 잘 선택하는가?

Backbone은 `ClearCLIP` 하나로 고정한다. EXP0에서 backbone ablation까지 섞지 않는 이유는 data 수와 group size의 효과를 먼저 분리하기 위해서다.

## Data split

- 대상: BrainMRI, BUSI, KiTS, ColonDB, Covid-QU-Ex, MedSeg, HAM10000, PH2
- 한 run은 이 중 한 dataset만 학습한다.
- Train 크기: 32 또는 128 `(image, class prompt)` pair
- Prompt별 pair 수는 최대 1개 차이만 나도록 균등 배분한다.
- 가능한 경우 각 prompt 안에서 positive/absent pair도 균형화한다.
- Train에 한 번이라도 포함된 patient/image는 해당 run의 internal test에서 전부 제외한다.
- Internal: 같은 dataset의 나머지 image
- External: 해당 run이 학습하지 않은 나머지 7개 dataset 전체

MedSeg처럼 image 수가 128보다 적어도 prompt가 여러 개이므로 128 pair 구성이 가능하다. 같은 image가 서로 다른 prompt와 train pair를 만들 수 있지만, 그 image 전체는 internal에서 제외해 image leakage를 막는다.

실제 생성된 internal image 수는 다음과 같다.

| Dataset | n=32 internal | n=128 internal |
|---|---:|---:|
| BrainMRI | 569 | 483 |
| BUSI | 83 | 31 |
| KiTS | 457 | 361 |
| ColonDB | 328 | 232 |
| Covid-QU-Ex | 551 | 455 |
| MedSeg | 69 | 16 |
| HAM10000 | 9,947 | 9,796 |
| PH2 | 170 | 97 |

따라서 n=128의 MedSeg와 BUSI internal 결과는 표본 수가 작다는 점을 함께 보고해야 한다.

## Controller 입력과 reward

CNN controller는 기존 11-D SCM state를 사용하지 않는다. Dense patch image embedding에 foreground text와 normal text의 차이 방향을 곱해 class-conditioned spatial tensor를 만든 뒤, `1x1 conv → depthwise 3x3 conv → 1x1 conv → mean/max pooling`으로 네 Beta action을 출력한다.

Reward는 두 경우만 사용한다.

- GT에 class가 있으면 hard Dice
- GT에 class가 없으면 `1 - predicted foreground area`

KiTS, ColonDB, Covid-QU-Ex는 foreground prompt가 하나이고 모든 image에 그 class가 있으므로 absent-pair reward가 발생하지 않는다. 이 세 dataset에서는 positive Dice만으로 학습되며, absent-FP reward의 효과는 다중 prompt dataset에서만 관찰할 수 있다.

Action은 기존 `eta`, `tau`, `gamma`, `kappa_sp` 네 개이고 inference에서는 Beta mean을 사용한다.

## 세 실험과 실행 수

| Job 종류 | 조합 | Job 수 | 실제 policy 수 |
|---|---:|---:|---:|
| Dataset grid upper bound | 8 datasets | 8 | 0 |
| Ablation job | 8 datasets × 2 train sizes × 3 group sizes | 48 | Global + CNN 각 1개 |
| 합계 |  | 56 jobs | 96 policies |

Upper-bound는 n=128 split의 internal set을 사용한다. 625개 full4 grid를 patch resolution에서 모두 평가한 후 상위 8개와 fixed baseline을 최종 평가와 동일한 224×224 dataset mIoU로 다시 비교한다. 따라서 이전 EXP2 pair-Dice Oracle과 달리 최종 foreground/background mIoU를 직접 선택 기준으로 사용한다. 다만 finite grid 안에서의 상한이지 continuous action의 수학적 상한은 아니다.

## 실행

```bash
bash scripts/exp0/run_all.sh
```

단계별 실행은 다음과 같다.

```bash
python scripts/exp0/make_splits.py
python scripts/exp0/build_matrix.py
python scripts/exp0/build_cache.py --dataset BrainMRI --device cuda:0
python scripts/exp0/run_job.py --job-id ub_BrainMRI --device cuda:0
python scripts/exp0/run_job.py --job-id abl_BrainMRI_n32_g4 --device cuda:0
python scripts/exp0/summarize.py
```

모든 평가 결과는 dataset별 행으로 저장하며 internal/external 평균만으로 결론 내리지 않는다.
