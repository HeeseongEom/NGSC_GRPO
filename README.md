# NGSC-GRPO feasibility

Frozen NGSC의 inference hyperparameter를 source-only GRPO로 보정하고, 완전히 held-out인 4개 target dataset에서 평가하는 독립 프로젝트다. 기존 레포에서 필요한 BiomedCLIP checkpoint·remote-code 파일·prompt·8개 dataset subset·세 참고문헌을 로컬에 복제하므로 원본 레포나 Hugging Face 네트워크에 의존하지 않는다.

## 빠른 실행

기존 `medclipsamv2` 환경처럼 PyTorch/Transformers가 설치된 환경에서 실행한다. 현재 서버에서는 `/data2/hseum/.conda/envs/medclipsamv2/bin/python`을 자동 선택하고, 해당 경로가 없으면 활성화된 환경의 `python`을 사용한다.

통합 검증 환경은 Python 3.9.12, PyTorch 2.8.0+cu128, Transformers 4.42.4, NumPy 1.26.4다.

```bash
cd /data2/hseum/NGSC_GRPO
bash scripts/exp1/00_validate_project.sh
bash scripts/exp1/run_smoke.sh                  # CPU 최소 경로 검증(각 target 2장)
bash scripts/exp1/run_feasibility_all.sh        # 4 CLIP × 3 seeds 전체 학습·평가
```

한 dense CLIP만 실행하려면 다음을 사용한다.

```bash
bash scripts/exp1/run_method.sh MaskCLIP
```

다른 Python 또는 GPU를 지정할 수 있다.

```bash
PYTHON_BIN=/path/to/python CONFIG=configs/feasibility.yaml \
  bash scripts/exp1/run_method.sh NACLIP --device cuda:1
```

중간 산출물이 config fingerprint와 일치하면 자동 재사용한다. 다시 만들 때만 `--force`를 붙인다. 전체 프로토콜과 해석 기준은 [docs/FEASIBILITY_EXPERIMENT_KO.md](docs/FEASIBILITY_EXPERIMENT_KO.md)에 요약되어 있다.

## 구조

```text
assets/       prompt, label map, provenance
checkpoints/  완전 로컬 BiomedCLIP checkpoint/processor/tokenizer
configs/      본 실험과 smoke 설정
data/         8개 dataset의 protocol-required subset
docs/         실험 요약 및 주의사항
references/   NGSC, TRIPS, 상세 설계 문서
scripts/exp1/ 기존 feasibility 검증 → split/cache → train → test → summary
scripts/exp2/ TRIPS-faithful 후속 실험
src/          feature adapter, continuous NGSC, GRPO, evaluation
tests/        action/reward/policy/metric 단위 테스트
outputs/      method별 cache·policy·결과(실행 후 생성)
```

EXP2 전체 실행:

```bash
bash scripts/exp2/run_all.sh
```

EXP2는 EXP1 frozen feature cache를 재사용하며 continuous oracle, state-free Global Beta-GRPO,
reward/action ablation, LODO, prompt-disagreement state, 최종 internal/external 평가 순으로 실행된다.
상세 프로토콜은 `docs/EXP2_EXPERIMENT_PLAN_KO.md`에 정리되어 있다.
초기 NGSC부터 EXP1·EXP2까지의 데이터 구성, 설정 정의, dataset별 전체 결과는
`docs/NGSC_GRPO_ALL_EXPERIMENTS_KO_v2.md`에서 확인할 수 있다. 기존 압축본은
`docs/NGSC_GRPO_ALL_EXPERIMENTS_KO.md`에 보존되어 있다.

설치가 필요하면 `python -m pip install -e .`를 사용한다. 소스 checkout 설치 없이도 모든 script가 `PYTHONPATH=src`를 설정한다.

## EXP0: dataset별 기초 실험

Dataset을 섞지 않고 32/128 prompt-pair, group size 4/8/16으로 실행하는 EXP0는
`docs/EXP0_EXPERIMENT_PLAN_KO.md`에 정의되어 있다. Grid upper bound 8개와
Global/CNN-GRPO ablation 48개를 합친 56개 job은 다음 명령으로 실행한다.

```bash
bash scripts/exp0/run_all.sh
```
