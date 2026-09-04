# NGSC hyperparameter GRPO feasibility 실험 요약

## 무엇을 검증하는가

이 실험은 BiomedCLIP, prompt ensemble, SCM과 CDAM을 모두 고정한 채 NGSC inference의 네 연속값 `a=(eta, tau, gamma, kappa_sp)`만 image-class 상태에 맞춰 선택한다. 핵심 가설은 source 전체에 공통인 최적 static action보다 96-parameter Linear-Beta GRPO controller가 held-out target의 mIoU를 높인다는 것이다. TRIPS에서 bounded Beta action과 group-relative optimization 개념만 가져오며 diffusion schedule이나 Bernstein basis는 사용하지 않는다.

기존 shell에 있던 MaskCLIP, SCLIP, ClearCLIP, NACLIP 네 방법을 모두 실행한다. 기존 복제 모델은 방법 인자를 실제 forward에 전달하지 않고 한 TSA branch가 hard-code되어 있었으므로, 이번 코드는 frozen weight를 공유하면서 마지막 vision block의 dense rule을 `model_adapter.py`에서 명시적으로 선택한다.

## 데이터 처리와 누수 방지

Source는 BrainMRI, BUSI, KiTS, ColonDB이며 각 dataset의 기존 labeled test pool에서 seed 2027로 최대 64장을 class-stratified 선택한다. 그중 48장은 `source_train`, 16장은 `source_val`이다. patient/lesion ID가 제공되면 같은 ID가 split을 넘지 못하게 검사하고, ID가 없는 기존 2-D pool은 image를 unit으로 취급한다. 학습 sampler는 dataset → class → positive/negative 순서로 균형 sampling한다.

Target은 Covid-QU-Ex, MedSeg, HAM10000, PH2 전체다. source/target dataset, 파일, patient unit, 정규화한 class name의 교집합을 assert한다. Target manifest와 feature cache에는 label 및 present class를 저장하지 않는다. target mask는 최종 `evaluate` 단계에서만 읽으며 action search, state 표준화, checkpoint 선택에는 절대 사용하지 않는다. 이번 결과는 zero-shot만 유효하다. `one_normal_shot`은 진짜 정상 1장을 test와 분리한 manifest가 없으므로 의도적으로 실행 오류를 내는 API hook만 남겼다.

새 프로젝트에는 8개 dataset에서 실제 protocol이 읽는 `test_images`, `test_masks`와 HAM metadata만 복제했다. 원본 129GB 전체나 학습에 쓰지 않는 중복 폴더는 가져오지 않았다. 수정 BiomedCLIP checkpoint, processor/tokenizer, 50개 BiomedCoOp prompt, Brain/BUSI label map과 세 설계 참고자료는 로컬 복제했고 `assets/PROVENANCE.json`에 출처와 SHA-256을 기록했다. 디스크가 99% 사용 중이어서 큰 불변 payload 일부는 동일 filesystem hard link로 복제했다. 원본 경로 삭제와 무관하게 유지되지만 in-place 수정은 두 경로에 함께 반영되므로 data/checkpoint는 read-only input으로 취급한다.

## 실험 정의

SCM은 class별 patch z-normalization 후 seed를 argmax로 정한다. 상태는 raw score 5개, normalized score 4개, base CDAM affinity 2개인 11차원이며 source-train 통계만으로 표준화한 뒤 `[-5,5]`로 clip한다. `kappa_sp`는 seed와 patch의 정규화 좌표 거리로 affinity를 연속 감쇠한다. `gamma=0`이면 PAF가 꺼지고, `gamma=1`이면 기존 hard suppression endpoint가 된다.

먼저 source-train에서 Sobol 256 actions × 균형 추출 1024 pairs를 평가해 Source-Static `a0`를 찾는다. 이를 concentration 20 Beta reference로 삼아 Linear-Beta controller(11→8, 정확히 96 parameters)를 초기화한다. GRPO는 B=16 states, G=8 actions, 500 updates, Adam 1e-3, PPO clip 0.2, reference KL 1e-3, grad clip 1.0이며 seeds 0/1/2를 실행한다. positive reward는 hard Dice, class-absent reward는 `1-foreground area`다. 마지막 update만 평가하고 source-val로 checkpoint를 고르지 않는다. Test action은 Beta mean으로 결정하며 sampling·reward·update·target adaptation은 없다.

네 비교군은 (B0) 기존 image-global normalization과 고정 NGSC, (B1) class normalization만 바꾼 Core-Fixed, (B2) Source-Static, (B3) Conditional-GRPO다. Multi-class에서는 threshold를 넘은 eligible class 중 refined score가 가장 큰 하나만 선택해 mask가 겹치지 않게 한다. 평가는 target별 foreground class IoU와 background IoU의 평균인 기존 mIoU가 primary이며 Dice, AUROC, absent FP area를 함께 저장한다. 결과 CSV는 `[0,1]` 값과 `_percent` 열을 함께 기록한다. action 평균/표준편차, boundary hit, Beta concentration, state-action correlation도 저장한다.

## 실행과 결과 확인

```bash
cd /data2/hseum/NGSC_GRPO
bash scripts/exp1/00_validate_project.sh
bash scripts/exp1/run_smoke.sh
bash scripts/exp1/run_feasibility_all.sh
```

단계별로는 `01_make_splits.sh` → `02_build_cache_all.sh` → `03_train_all.sh` → `04_test_all.sh` → `05_summarize.sh` 순서다. feature cache가 가장 비싸며 같은 method의 static search·세 seed·네 baseline에서 재사용한다. 공통 split은 `outputs/grpo_ngsc_feasibility_v1/splits`, method별 산출물은 그 아래 `<METHOD>/{cache,static_search,seeds,results,diagnostics}`에 기록된다. 전체 표는 `summary_all_methods.csv`, 판정 근거는 method별 `feasibility_decision.json`에 저장된다.

GO는 seed 평균 GRPO가 Source-Static보다 macro mIoU `+1.0` point 이상이고 Original보다 높으며, 각 seed에서 3/4 target이 Static 대비 `-0.5` point 이내, 개선 방향이 seed 간 일관되고 boundary collapse가 없을 때다. 이 판정은 정식 통계 결론이 아니라 full LODO 진행 여부다.

## 반드시 주의할 점

- 첫 full run 전에 B0가 기존 저장 NGSC 결과와 맞는지 소수 표본의 raw/refined map까지 비교한다. 새 4-method adapter는 숨은 수동 model edit를 제거했지만, 과거 실행 당시 실제 branch가 기록되지 않았다면 숫자가 달라질 수 있다.
- config를 바꾸면 fingerprint가 달라져 cache/policy 재사용이 차단된다. 임의로 `.pt`를 섞지 않는다.
- target metric을 보고 action bound, prompt, update 수, checkpoint를 고르면 feasibility 자체가 target-tuned가 된다. 변경은 source train/val에서만 결정한다.
- class-absent pair를 Dice=1로 처리하지 않는다. 학습은 `1-FP area`, 보고용 foreground mIoU는 GT-present class만 평균하고 absent FP는 별도 기록한다.
- `gamma→0`은 PAF 불필요, `kappa_sp→0`은 spatial term 불필요라는 유효한 ablation 신호다. 반면 action std≈0 또는 95% 이상 boundary hit는 controller collapse로 해석한다.
- `run_smoke.sh`는 코드 경로 검증용으로 target당 2장만 쓰며 연구 결과가 아니다. 전체 target cache는 디스크와 시간이 크므로 먼저 여유 공간/GPU를 확인한다.
