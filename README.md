# NGSC-GRPO

Frozen NGSC(Non-Gradient Semantic Correction)의 **inference hyperparameter**를 GRPO로 보정하는 프로젝트다.
Dense CLIP backbone과 BiomedCLIP checkpoint는 전부 freeze한 채, NGSC의 연속 action
`(eta, tau, gamma, kappa_sp)`만 policy로 예측한다. 8개 medical segmentation dataset에서
dataset별 학습 → internal/external 평가를 수행한다.

---

## 1. 저장소 구조

```text
src/ngsc_grpo/      핵심 라이브러리
  core.py             연속 NGSC 연산(정규화·seed·CDAM·spatial prior)
  policy.py           Beta 분포 policy (Global / CNN 두 arm)
  training.py         GRPO 학습 루프 (group sampling, PPO-clip, KL)
  evaluation.py       mIoU / Dice / absent-FP 지표
  cache.py            frozen feature·score 캐시
  splits.py           dataset split 생성
  model_adapter.py    dense CLIP(ClearCLIP/MaskCLIP/NACLIP/SCLIP) 어댑터
  config.py registry.py provenance.py reporting.py cli.py

configs/            실험별 YAML — exp0.yaml, exp0_1.yaml, exp2.yaml, feasibility.yaml, smoke.yaml
scripts/            실험 실행 파이프라인 (아래 2절)
data/               8개 dataset의 protocol-required subset
assets/             prompt template, label/prompt map, provenance
docs/               실험 계획 및 전체 결과 문서
tests/              core/policy/split/evaluation/exp0 단위 테스트
reports/            EXP1·EXP2 최종 리포트(문서·CSV·그림)
outputs/            실행 산출물 — 저장소에 포함되지 않음(.gitignore)
checkpoints/        BiomedCLIP checkpoint — 용량 때문에 저장소에 미포함
references/         참고 논문 PDF — 저장소에 미포함
```

`checkpoints/`, `references/`, `outputs/`는 git에 올라가 있지 않으므로, clone 후에는
`configs/*.yaml`의 `paths.checkpoint_dir`가 가리키는 위치에 BiomedCLIP checkpoint를 직접 배치해야 한다.

### 환경

PyTorch/Transformers가 설치된 환경이면 된다. 검증 환경은 Python 3.9.12, PyTorch 2.8.0+cu128,
Transformers 4.42.4, NumPy 1.26.4다. 스크립트는 기본적으로
`/data2/hseum/.conda/envs/medclipsamv2/bin/python`을 사용하며, `PYTHON_BIN` 환경변수로 바꿀 수 있다.
모든 스크립트가 `PYTHONPATH=src`를 스스로 설정하므로 별도 설치는 선택 사항이다(`python -m pip install -e .`).

```bash
python -m pytest tests -q      # 단위 테스트
```

---

## 2. `scripts/` 구성

| 디렉터리 | 내용 |
|---|---|
| `exp0/` | **주 실험.** Dataset별 기초 NGSC-GRPO — split·cache·upper bound·ablation 전 과정 |
| `exp0_1/` | **reward ablation.** EXP0의 산출물을 재사용하고 reward 함수만 교체 |
| `exp0_dense_upper/` | EXP0 upper bound를 더 촘촘한 0.1 간격 grid로 재계산 |
| `exp0_reports/` | EXP0 리포트 부가 산출물(global action value CSV, reward 함수 PDF) |
| `exp1/` | 초기 feasibility 검증 (4 dense CLIP × 3 seed) |
| `exp2/` | TRIPS-faithful 후속 실험 (oracle, LODO, prompt-disagreement state) |

`exp0`와 `exp0_1`은 동일한 파일 구성을 가지며, 각 스크립트의 역할은 다음과 같다.

| 스크립트 | 역할 |
|---|---|
| `make_splits.py` | dataset별 train/internal/external split 생성 |
| `build_matrix.py` | 실행할 job 목록(job matrix) CSV 생성 |
| `build_cache_all.py` / `build_cache.py` | frozen dense feature·NGSC score 캐시 구축 (GPU 병렬) |
| `run_matrix.py` | job matrix를 GPU 큐에 분배해 병렬 실행 |
| `run_job.py` | job 하나 실행 (`--job-id`, `--device`) |
| `train.py` | GRPO 학습 (Global / CNN arm) |
| `upper_bound.py` | grid search 기반 oracle upper bound |
| `evaluate.py` | internal/external 평가 |
| `summarize.py` | 전체 결과를 리포트 CSV로 집계 |
| `smoke_test.py` | 최소 경로 동작 확인 |
| `common.py` / `common.sh` | config 로딩, 경로, 환경변수 설정 |

---

## 3. EXP0 — dataset별 기초 실험

Dataset을 섞지 않고 **dataset 하나로 학습해 그 dataset(internal)과 나머지(external)에서 평가**한다.

- **Dataset (8종)**: BrainMRI, BUSI, KiTS, ColonDB, Covid-QU-Ex, MedSeg, HAM10000, PH2
- **Backbone**: ClearCLIP 고정 (`method: ClearCLIP`)
- **Job 56개** = upper bound 8개(dataset당 1개) + ablation 48개
  - ablation = 8 dataset × train pair {32, 128} × group size {4, 8, 16}
  - 각 ablation job은 **Global**·**CNN** 두 policy arm을 함께 학습
- **비교 대상 3종**: fixed NGSC(baseline) / GRPO(Global·CNN) / grid upper bound
- **Reward**: present → `hard_dice`, absent → `1 - foreground_area`

설정은 `configs/exp0.yaml`, 프로토콜 상세는 `docs/EXP0_EXPERIMENT_PLAN_KO.md`에 있다.

### 실행

```bash
bash scripts/exp0/run_all.sh
```

`run_all.sh`는 아래 6단계를 순서대로 수행한다.

```text
make_splits.py  →  build_matrix.py  →  build_cache_all.py (GPU 0,1)
   →  run_matrix.py --job-type upper_bound   (8 jobs)
   →  run_matrix.py --job-type ablation      (48 jobs)
   →  summarize.py
```

단계별·부분 실행:

```bash
# 다른 GPU / 워커 수로 ablation만 재실행
python scripts/exp0/run_matrix.py --config configs/exp0.yaml \
  --job-type ablation --gpus 0 1 --workers-per-gpu 3

# job 하나만 실행 (job-id는 build_matrix.py가 생성)
python scripts/exp0/run_job.py --job-id abl_BUSI_n128_g8 --device cuda:0

# 스모크 테스트
python scripts/exp0/smoke_test.py --config configs/exp0.yaml
```

산출물이 config fingerprint와 일치하면 자동 재사용되고, 다시 만들려면 `--force`를 붙인다.
`PYTHON_BIN`, `CONFIG` 환경변수로 인터프리터와 설정 파일을 바꿀 수 있다.

```bash
PYTHON_BIN=/path/to/python CONFIG=configs/exp0.yaml bash scripts/exp0/run_all.sh
```

### 더 촘촘한 upper bound

```bash
python scripts/exp0_dense_upper/run_all.py --config configs/exp0.yaml --gpus 0 1
python scripts/exp0_dense_upper/summarize_dense.py --config configs/exp0.yaml
```

---

## 4. EXP0_1 — reward ablation

**EXP0에서 reward 함수만 바꾼 실험이다.** split·frozen feature cache·fixed NGSC score·upper bound를
EXP0 산출물에서 그대로 재사용하므로(`artifacts.reuse_experiment: ngsc_grpo_exp0`),
캐시 구축과 upper bound 단계가 없고 ablation 48개만 다시 학습한다.

| | EXP0 | EXP0_1 |
|---|---|---|
| present reward | `hard_dice` | `0.6·Dice + 0.3·lesion F1 + 0.1·boundary F1` |
| absent reward | `1 - foreground_area` | `empty_margin_then_one_minus_foreground_area` (margin 0.2) |
| 그 외 설정 | — | EXP0와 동일 |

lesion F1은 8-connectivity 연결성분 기준, boundary F1은 2 px 허용오차 기준이다.
설정은 `configs/exp0_1.yaml`에 있다.

### 실행

```bash
bash scripts/exp0_1/run_all.sh
```

```text
build_matrix.py  →  run_matrix.py --job-type ablation (48 jobs)  →  summarize.py
```

EXP0를 먼저 완료해 두어야 한다. reward 구현만 따로 검증하려면:

```bash
python scripts/exp0_1/test_reward.py
```

---

## 5. 결과 위치

모든 산출물은 `outputs/<experiment.name>/` 아래에 쌓인다
(EXP0 → `outputs/ngsc_grpo_exp0/`, EXP0_1 → `outputs/ngsc_grpo_exp0_1/`).

```text
outputs/ngsc_grpo_exp0/
  splits/       n32/, n128/                      dataset split
  cache/        ClearCLIP/                       frozen feature·score 캐시
  runs/         <dataset>/n{32,128}/g{4,8,16}/{global,cnn}/
                  policy_final.pt, training_log.csv, metadata.json
  upper_bound/  <dataset>/grid_results.csv, internal_summary.csv, metadata.json
  evaluations/  <dataset>/n{32,128}/g{4,8,16}/{global,cnn}/     GRPO 평가
                <dataset>/n{32,128}/no_group/fixed_ngsc/        baseline 평가
  runner_logs/                                   job별 실행 로그
  reports/                                       ← 최종 집계 결과
```

`summarize.py`가 만드는 **`reports/`가 결과의 최종 정리본**이다.

| 파일 | 내용 |
|---|---|
| `exp0_all_results.csv` | 모든 job·arm의 원본 지표 |
| `exp0_internal_comparison.csv` | dataset×n×g별 `NGSC / Global / CNN / Upper Bound` mIoU와 baseline 대비 차이 |
| `exp0_external_comparison.csv` | 위와 동일하되 학습 dataset → 나머지 dataset 일반화 성능 |
| `exp0_training_diagnostics.csv` | 학습 곡선 요약(reward, KL, clip 비율 등) |
| `exp0_dense_upper_bound_0p1.csv` | 0.1 간격 dense grid upper bound |
| `exp0_global_grpo_action_values.csv` | 학습된 Global policy의 action 값 |
| `status.json` | job 완료 상태 |

EXP0_1도 같은 구조이며 파일명 접두사만 `exp0_1_`이다
(`exp0_1_all_results.csv`, `exp0_1_internal_comparison.csv`, `exp0_1_external_comparison.csv`,
`exp0_1_training_diagnostics.csv`).

비교 CSV의 컬럼은 다음과 같다.

```text
train_dataset, train_pairs, group_size, [eval_dataset,]
NGSC_mIoU, Global_optim_mIoU, Global-NGSC, CNN_optim_mIoU, CNN-NGSC, Upper_Bound
```

`outputs/`는 `.gitignore` 대상이라 저장소에 포함되지 않는다. EXP1·EXP2의 확정 리포트는
저장소에 포함된 `reports/exp1/`, `reports/exp2/`에서 볼 수 있다.

---

## 6. 그 외 실험

```bash
bash scripts/exp1/run_smoke.sh                # CPU 최소 경로 검증
bash scripts/exp1/run_feasibility_all.sh      # 4 dense CLIP × 3 seed
bash scripts/exp1/run_method.sh MaskCLIP      # 단일 backbone
bash scripts/exp2/run_all.sh                  # TRIPS-faithful 후속 실험
```

## 7. 문서

| 문서 | 내용 |
|---|---|
| `docs/EXP0_EXPERIMENT_PLAN_KO.md` | EXP0 프로토콜 |
| `docs/EXP2_EXPERIMENT_PLAN_KO.md` | EXP2 프로토콜 |
| `docs/FEASIBILITY_EXPERIMENT_KO.md` | EXP1 feasibility 프로토콜 |
| `docs/NGSC_GRPO_ALL_EXPERIMENTS_KO_v2.md` | 초기 NGSC부터 EXP2까지 전체 설정·결과 종합 |
