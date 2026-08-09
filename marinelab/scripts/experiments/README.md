# Competitor 실험 실행 가이드

`docs/experiments/experiments_plan.md`의 E1–E4를 실행하는 프레임워크. 설계 근거는
`docs/experiments/competitor_framework_plan.md`, 방법(controller) 구현은 `marinelab/marinelab/control/`
(SSI-MPC는 `marinelab/marinelab/third_party/ssi_mpc_gpl/` — **GPL-3.0**, 배포 시 주의).

## 사전 준비 (Linux sim 호스트, 컨테이너)

```bash
git lfs pull                                          # 메시 없으면 USD 로드 실패
./docker/run.sh '/isaac-sim/python.sh -m pip install optuna'   # tune.py 전용, 1회
# PPO exported policy 생성 (1회): play.py가 로드 직후 checkpoints/exported/policy.{pt,onnx}를 씀
./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/play.py \
    --task Isaac-PKRC-WallScan-Eval-Direct-v0 --num_envs 1 --headless \
    --checkpoint ../checkpoints/rb_train_model_7998.pt --log_traj --eval_steps 10 --no_plot'
```

## 실행 순서 (의존성)

튜닝 산출물(`best_params.json`)을 E1~E3 설정이 참조하므로 순서가 있다:

```bash
# ① BO-static 가중치 튜닝 (~100 trial, 앱/솔버 각 1회 빌드로 전 trial 실행)
./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/experiments/tune.py \
    --config ../marinelab/scripts/experiments/configs/tune_bo_nmpc.yaml'

# ② SSI-MPC 하이퍼파라미터 튜닝 (①의 가중치를 승계 — ① 완료 후)
./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/experiments/tune.py \
    --config ../marinelab/scripts/experiments/configs/tune_ssi_mpc.yaml'
#    끝나면 experimental_results/tuning/ssi_mpc/best_params.json의 lr/kernel_std를
#    e1_nominal.yaml / e2_dr_sweep.yaml / e3_current.yaml의 ssi_lr/ssi_kernel_std에 반영

# ③ E1 본 비교 — 방법 지정이 기본 단위 (전체 실행은 명시적으로 'all')
./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/experiments/run_experiment.py nominal \
    --config ../marinelab/scripts/experiments/configs/e1_nominal.yaml'
...                                                                    diff --config ...
...                                                                    all  --config ...   # 5방법 순차 (수 시간)

# ④ E2 강건성 스윕 / E3 조류  (E1 상위 방법 확정 후)
... run_experiment.py ssi --config .../e2_dr_sweep.yaml
... run_experiment.py all --config .../e3_current.yaml
```

단일 셀만 실행(디버그/재실행): `run_experiment.py nominal --cond dr50 --seed 0 --config ...`.
방법 키: `nominal`(고정 가중치 NMPC) / `bo` / `ppo` / `ssi` / `diff`(제안, Diff-WMPC).
방법별 결과 파일이 독립이라 방법 단위로 나눠 돌려도 집계는 동일하다.
`tune.py`는 SQLite storage 기반이라 중단 후 같은 커맨드로 재개된다.

## 결과 저장 위치

모든 실험 산출물은 `<repo>/experimental_results/` 아래에 실험별 디렉토리로.
(기존 `results/`는 이 프레임워크 이전의 legacy 산출물 전용 — 여기서는 읽지도 쓰지도 않는다.)

```
experimental_results/
├── e1/                                   # exp 이름 (yaml의 exp: 키), 종류별 서브디렉토리
│   ├── raw/      trajectory_<method>_<cond>_s<seed>.npz   # raw 궤적 (전 스텝, 전 env)
│   ├── metrics/  metrics_<method>_<cond>_s<seed>.json     # 지표 + score + controller_cost
│   ├── plots/    trajectory_<method>_<cond>_s<seed>.png   # 셀별 진단 플롯
│   ├── tables/   table.csv, table.tex                     # aggregate.py 산출
│   └── figures/  fig_f1..f6.{png,pdf}                     # plot_figures.py 산출
├── e2/  e3/  ...                          # 동일 규약
├── tuning/
│   ├── bo_nmpc/   {study.db, trials.csv, best_params.json, budget.json}
│   └── ssi_mpc/   {study.db, trials.csv, best_params.json, budget.json}
└── _synthetic_demo/                       # GPU-free 파이프라인 데모 (재생성 가능, 미커밋)
```

`aggregate.py`/`plot_figures.py`에는 실험 디렉토리(`experimental_results/e1`)만 주면 서브디렉토리를
자동 인식한다 (flat 디렉토리도 하위호환).

- **최신 실행이 곧 현재 상태**: 같은 (방법, 조건, seed) 셀을 재실행하면 동일 파일명을
  덮어쓴다. 버전 이력이 필요하면 커밋으로 남긴다.
- `metrics_*.json`에는 표 지표 외에 실행 옵션 스냅샷(`options`), 채점 스칼라
  (`score.objective` — 충돌 시 Infinity), 방법별 계산 비용(`controller_cost`,
  E4(c)의 소스)이 함께 기록된다.
- raw npz는 기존 `scripts/rescore_trajectories.py`와 호환 — 지표 정의가 바뀌어도
  재실행 없이 재채점 가능.

## 집계 → 논문 표 (자동)

시뮬 불필요, 아무 PC에서나 (개발 Windows PC 포함):

```bash
python marinelab/scripts/experiments/aggregate.py experimental_results/e1
# -> experimental_results/e1/table.csv  +  experimental_results/e1/table.tex
python marinelab/scripts/experiments/aggregate.py experimental_results/e2 --metrics cycles_mean score.objective
```

- `table.tex`: 방법(행, 고정 순서 Fixed→BO→PPO→SSI→Diff) x 지표(열)의 booktabs 표,
  셀은 `mean ± sd`(trial = seed[×env]), 다중 조건이면 조건별 블록. 충돌로 objective가
  inf인 그룹은 `--`로 표기. `\input{table.tex}`로 논문에 바로 삽입.
- per-trial 원값(오버레이 점 찍기용)은 `table.csv`의 `values` 열에 세미콜론 구분으로
  보존된다 — experiments_plan.md의 "min/max 막대 금지, per-trial 점 오버레이" 요건용.
- 기본 지표 셋은 `aggregate.py`의 `DEFAULT_METRICS` (Table 1 열 구성). `--metrics`로
  `metrics_*.json`의 아무 dotted 경로나 지정 가능.
- E4(b) 튜닝 비용 열은 `experimental_results/tuning/*/budget.json`에서 (`collect_budgets`).

## 논문 그림 (자동, 계획 §10의 F1–F6)

역시 시뮬 불필요 — png + pdf 동시 생성:

```bash
python marinelab/scripts/experiments/plot_figures.py f1 experimental_results/e1                      # 본 비교 오버레이 (mean±SD + per-trial 점)
python marinelab/scripts/experiments/plot_figures.py f2 experimental_results/e1 --cond nominal --seed 0   # 대표 궤적 (s–z 전개도)
python marinelab/scripts/experiments/plot_figures.py f3 experimental_results/e2 --metric score.objective  # 강건성 곡선 (vs 섭동 강도)
python marinelab/scripts/experiments/plot_figures.py f4 experimental_results/e3 --cond step --seed 0 --t-event 90  # 외란 응답 시계열
python marinelab/scripts/experiments/plot_figures.py f5 experimental_results/e2 experimental_results/e2b --names zero-shot fine-tuned  # E2(b) 후
python marinelab/scripts/experiments/plot_figures.py f6 experimental_results/e1 --tuning experimental_results/tuning   # 비용 비교
```

출력은 기본 `<results_dir>/fig_<이름>.{png,pdf}` (`--out`으로 변경). f2/f4는 해당
조건·seed의 `trajectory_*.npz`에서 방법별 패널을 자동 구성한다.

## 방법/조건 추가

- **조건 추가**: 실험 yaml의 `conditions:`에 항목 추가 (`dr_fluid_scale`, `current:` 프로파일,
  `task`, `state: gt|ekf` 등 — 셀 옵션은 defaults < condition < method 순으로 병합).
- **방법 추가**: `marinelab/control/`에 `WallScanController` 구현 →
  `_sim_loop.build_controller`에 분기 → yaml `methods:`에 등록.

## 개발 PC (sim 없는 환경)

순수 계층은 네이티브로 테스트한다 (torch/numpy/pyyaml/scipy/gymnasium만 필요):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest marinelab/tests -q
```
