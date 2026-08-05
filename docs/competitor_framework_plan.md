# Competitor/Baseline 실험 프레임워크 구현 계획

**목적**: `experiments_plan.md`의 E1–E4를 "방법 × 조건 × seed" 조합으로 반복 실행할 수 있는
구조를 repo에 추가한다. E5(하드웨어)로의 이식이 쉬운 형태로 컨트롤러 계층을 설계한다.

**원칙**

1. **기존 코드는 수정하지 않는다.** 모든 신규 코드는 새 파일/새 패키지로 추가하고, 기존
   모듈은 import 해서 감싼다(어댑터). 기존 스크립트(`run_wallscan_mpc.py`, `play.py`,
   `train_diff_wmpc_wallscan.py`)는 회귀 기준선으로 그대로 남긴다.
2. **컨트롤러는 순수 계층에 둔다.** isaaclab/pxr import 금지 — 기존 `geometry`/`sensors`/
   `scan_state_machine`/`mpc_reference`/`wall_frame_ekf`가 지키는 규칙과 동일. 이것이
   sim2real seam이자, Windows 개발 PC에서 네이티브 유닛테스트가 도는 조건이다.
3. **방법(controller)과 조건(env/외란/상태소스)과 프로토콜(steps/seed/집계)을 직교로 분리**
   한다. E1~E3의 모든 셀은 이 세 축의 조합으로 표현되어야 한다.

---

## 1. 현재 자산 (2026-08-03 코드 확인 결과)

| 자산 | 위치 | 상태 |
|---|---|---|
| Fixed-W NMPC 폐루프 | `marinelab/scripts/run_wallscan_mpc.py` | 보유. `--state gt/ekf`, `--seed`, solve-time·saturation 통계 내장 |
| NMPC 솔버 쌍 (nominal+sensitivity) | `tasks/pkrc_wallscan/mpc_controller.py` | 보유. 가중치 파라메트릭, acados |
| 오차벡터·reference preview | `tasks/pkrc_wallscan/mpc_reference.py` | 보유. 순수 torch, 단위테스트 있음 |
| Diff-WMPC 학습 코어 | `algorithms/diff_wmpc.py` + `scripts/train_diff_wmpc_wallscan.py` | 보유. `--ckpt`는 eval 전용(learning OFF) |
| PPO 정책 | `checkpoints/rb_train_model_7998.pt` + `play.py`(onnx export 포함) | 보유 |
| Wall-frame EKF | `tasks/pkrc_wallscan/wall_frame_ekf.py` | 보유. 순수 계층 |
| 평가 지표·재채점 | `eval_metrics.py`, `scripts/rescore_trajectories.py` | 보유 |
| stress DR | `wallscan_env_cfg.py`의 Eval cfg | **고정 1단계** — ±25/50/75% 스윕 없음 |
| 조류 | `core/ocean_current.py` | 프리미티브만 존재. **wallscan env에 미연결** (`wallscan_env.py:77` 명시) |
| SSI-MPC / BO 튜닝 / seed 배치·집계 | — | **없음** (신규 구현 대상) |

---

## 2. 신규 디렉토리 구조 (전부 추가, 수정 없음)

```
marinelab/marinelab/control/              # [NEW] 순수 컨트롤러 계층 (isaaclab import 금지)
    __init__.py
    types.py            # VehicleState / ScanReference / ControlOutput dataclass
    base.py             # WallScanController 추상 인터페이스 + 타이밍 계측
    fixed_nmpc.py       # ① Fixed-W NMPC 어댑터 (mpc_controller 재사용, 가중치 주입 가능 → ②BO도 커버)
    ppo_policy.py       # ③ PPO 러너 (TorchScript/ONNX 로드 — Jetson과 동일 코드)
    ssi_mpc.py          # ④ SSI-MPC (신규 구현: 동일 OCP + 온라인 모델 적응)
    diff_wmpc_ctrl.py   # ⑤ Diff-WMPC 어댑터 (WeightPolicy → 가중치 → mpc_controller)
    estimator.py        # 상태소스 추상화: GT | EKF (wall_frame_ekf 조립) — sim/hw 공용

marinelab/marinelab/experiments/          # [NEW] 실험 프로토콜의 순수 코어 (네이티브 테스트 대상)
    protocol.py         # yaml -> 셀 행렬 전개, 결과 파일 명명 규약
    aggregate.py        # 수집/통계(mean±SD+per-trial)/표 생성, budget.json 집계

marinelab/scripts/experiments/            # [NEW] 실험 실행 계층 (sim 전용, 위 코어의 CLI)
    run_experiment.py   # 단일 진입점: --config <yaml> [--method ... --cond ... --seed ...]
    env_variants.py     # E2/E3용 파생 env cfg + 신규 gym ID 등록 (기존 __init__.py 무수정)
    tune.py             # ②④ 공용 자동 튜닝 파이프라인 (Optuna) — §6 프로토콜
    aggregate.py        # results/<exp>/ 수집 → mean±SD + per-trial 오버레이 표·그림 + E4(b) 비용표
    bench_inference.py  # E4(c): solve+forward 마이크로벤치, isaaclab 무의존 (Jetson에서 그대로 실행)
    configs/
        e1_nominal.yaml e2_dr_sweep.yaml e2_finetune.yaml e3_current.yaml e4_ablation.yaml
        tune_bo_nmpc.yaml tune_ssi_mpc.yaml   # 탐색 공간 선언 (방법별 차이는 이 파일뿐)

marinelab/tests/control/                  # [NEW] 네이티브 테스트 (Windows 개발 PC에서 실행 가능)
```

- `control/`이 marinelab 패키지 안에 있어도 기존 파일 수정은 없다 (서브패키지 추가는 어떤
  기존 파일도 건드리지 않는다). 단 `marinelab/__init__.py`는 `from . import tasks`로 무거운
  env 등록을 트리거하므로(PEP 562 lazy는 `tasks/pkrc_wallscan/__init__`에만 해당), isaaclab이
  없는 환경에서는 `tests/conftest.py`의 패키지 shim 패턴으로 우회한다 — 테스트는 이미 자동
  적용되고, 하드웨어 부트스트랩용 동일 shim은 P4(E5 준비)에서 standalone 모듈로 제공한다.
- 신규 gym ID(DR 스윕·조류 변형)는 `env_variants.py` import 시점에 `gym.register`로 추가 등록
  — 기존 `tasks/pkrc_wallscan/__init__.py`는 건드리지 않는다.

---

## 3. 컨트롤러 인터페이스 (하드웨어 이식의 핵심)

```python
# control/types.py — 전부 plain numpy/torch. ROS 노드가 그대로 채울 수 있는 형태.
@dataclass
class VehicleState:      # 순서·프레임은 mpc_controller의 x(13)과 일치시킴
    pos_w: np.ndarray    # (3,)
    quat_wb: np.ndarray  # (4,) wxyz
    lin_vel_b: np.ndarray
    ang_vel_b: np.ndarray
    wall_range: float    # 소나 측정 (sim: ray-cast / hw: Ping1D)
    stamp: float

@dataclass
class ScanReference:     # scan_state_machine + mpc_reference.reference_preview 산출물
    ref_nodes: np.ndarray   # (N+1, NP_REF) preview, 또는 frozen setpoint(ablation)
    phase: int
    ...

@dataclass
class ControlOutput:
    u: np.ndarray        # (6,) 정규화 스러스터 명령 [-1,1] — env.step과 하드웨어 ESC 공용 규약
    solve_ms: float      # E4(c) 계측: sim/Jetson 동일 코드로 수집
    aux: dict            # 방법별 진단 (가중치 벡터, saturation 여부, 적응 파라미터 등)
```

```python
# control/base.py
class WallScanController(ABC):
    def reset(self, state: VehicleState) -> None: ...
    def step(self, state: VehicleState, ref: ScanReference) -> ControlOutput: ...
```

- **estimator 체인도 동일 규약**: `estimator.py`가 (raw 센서 → `VehicleState`)를 담당.
  sim에서는 `sensors.apply_sensors` 출력 + `wall_frame_ekf`, 실기체에서는 드라이버 토픽 +
  동일 `wall_frame_ekf`. E5에서 바뀌는 것은 estimator 입력원뿐, controller는 무변경.
- scan state machine·reference 생성은 러너(사이드) 책임으로 두되, 순수 모듈이므로
  하드웨어 노드에서도 동일 코드가 돈다 (기존 `run_wallscan_mpc.py`가 이미 이 패턴).
- PPO는 `checkpoints/exported/policy.onnx`(play.py가 생성)를 로드하는 러너로 구현 —
  Jetson 추론 경로와 코드가 갈라지지 않는다.

---

## 4. 방법별 구현 계획 (E1의 5개 방법)

| # | 방법 | 구현 | 규모 |
|---|---|---|---|
| ① | Fixed-W NMPC | `fixed_nmpc.py`: `mpc_controller` 호출 어댑터. 가중치 벡터를 ctor 인자로 (기본값 = 기존 DEFAULT_WERR) | 소 |
| ② | BO-tuned NMPC | ①과 동일 클래스 + §6 공용 튜닝(`tune.py --method bo_nmpc`)이 만든 `best_params.json` 로드 | 중 |
| ③ | PPO | `ppo_policy.py`: onnx/torchscript 로드, obs 31-D 조립은 estimator+러너가 제공 | 소 |
| ④ | SSI-MPC | `ssi_mpc.py`: **이식 (직접 구현 아님).** 공식 오픈소스 [UM-iRaL/SSI-MPC](https://github.com/UM-iRaL/SSI-MPC) (Zhou & Tzoumas, "Simultaneous System Identification and MPC with No Dynamic Regret", 쿼드로터+acados)에서 온라인 SysID 모듈과 OCP 결합부를 가져온다. **적응 법칙은 원 코드 그대로**(베이스라인 공정성 방어), 교체는 플랜트 모델(→`mpc_controller.py`의 UUV 모델)과 레퍼런스(→wallscan)뿐. ROS Noetic 래퍼·쿼드로터 모델은 버림. 적응 모듈은 acados 없이 단위테스트 가능하게 분리 유지. 하이퍼파라미터는 수동 튜닝 금지 — §6 공용 프로토콜로만 | 중 (이식) |
| ⑤ | Diff-WMPC | `diff_wmpc_ctrl.py`: 학습된 `WeightPolicy` 로드 → 매 스텝 가중치 → `mpc_controller`. `run_wallscan_mpc.py --policy_ckpt` 경로의 어댑터화 | 소 |

**신규 작성 vs 이식 구분**: 실질적 이식은 ④뿐이다. ②는 Optuna 글루, ①③⑤는 보유 코드
어댑터, 러너·집계·env variants는 우리 고유 인프라. Diff-WMPC는 이미 이식 완료 상태
(`algorithms/diff_wmpc.py` = 마린짐 참조 구현의 포트; TUM 원 코드는 골격만 공개라
논문에는 "자체 구현" 명시 — experiments_plan.md 제외표 참조).

**④ 착수 전 확인 필요**: 원 repo의 라이선스(재배포·수정 허용 범위), 준거 커밋 고정,
이식 범위(SysID 모듈 + 솔버 결합부) 확정. 경쟁 방법을 불리하게 약식 구현하면 E1 결과
전체가 공격받으므로 적응 법칙은 원 코드를 보존하고, 튜닝 노력도 제안 방법과 동급으로
기록해 둘 것(리뷰 대응).

---

## 5. 실험 러너와 프로토콜

- `run_experiment.py` 한 개가 E1/E2/E3/E4(a)를 전부 커버한다:
  `방법(nominal|bo|ppo|ssi|diff — 위치 인자, 전체 실행은 'all')` × `조건(--cond, env variant/상태소스/외란 프로파일)`
  × `--seed`. yaml이 셀 목록을 정의하고, CLI는 단일 셀 오버라이드용.
- acados는 순차 solve이므로 MPC 계열은 env 1개 × (8 trial × 5 seed) 루프,
  PPO는 8 env 병렬 — **trial 수(40)와 에피소드 길이(180 s)를 맞추는 것**이 프로토콜의 불변량.
  `--eval_steps`/`--score_episode` 규약은 기존 CLAUDE.md 함정(첫 에피소드 절반 길이)을 따른다.
- 출력 규약: `results/<exp>/<method>_<cond>_s<seed>.{npz,json}` — 기존 `eval_metrics`/
  `rescore_trajectories.py` 포맷을 그대로 사용해 재채점 호환 유지.
- `aggregate.py`: 위 규약을 스캔해 Table 1/2/3 (csv+latex)과 Fig(성능 vs 섭동 강도 곡선,
  zero-shot/fine-tuned 막대)를 생성. **mean±SD + per-trial 점 오버레이** (min/max 막대 금지
  — experiments_plan.md의 통계 요구).

## 6. 자동 튜닝 프로토콜 — BO-static(②)·SSI-MPC(④) 공용

튜닝이 필요한 두 베이스라인은 **동일한 파이프라인 · 동일한 예산 · 동일한 로그 스키마**로
튜닝한다. 목적은 두 가지: "베이스라인을 덜 튜닝해놓고 이겼다"는 리뷰 공격 봉쇄
(부모 논문이 MOBO-MPC 비용을 460k 샘플/1071 s로 공개한 선례를 따름), 그리고
E4(b) 비용표를 사람 손 없이 자동 생성.

### 파이프라인 (`tune.py`)

```
tune.py --method {bo_nmpc, ssi_mpc} --config configs/tune_<method>.yaml
  └─ Optuna study (TPE, sampler seed 고정, SQLite storage → 중단/재개 가능)
       └─ trial마다: run_experiment.py 단축 프로토콜(1 seed × 단축 에피소드) 호출 → 스칼라 점수
```

- **탐색 공간은 yaml 선언** — 방법별 차이는 이 파일뿐, 파이프라인 코드는 공용:
  - `bo_nmpc`: w_err(NE)·w_u(nu), log-scale, 상하한은 `WeightPolicy` 기본 범위와 동일
  - `ssi_mpc`: 적응 하이퍼파라미터 (RFF 피처 수, 망각계수/정규화, 업데이트 주기 등 —
    이식 시 원 repo의 튜닝 노브 목록으로 확정)
- **출발선 통일**: SSI-MPC의 MPC 비용 가중치는 ②의 최적 결과를 승계한다. 가중치까지
  다시 탐색하면 "모델 적응 vs 가중치 적응" 축이 섞이므로, SSI-MPC 튜닝은 적응
  하이퍼파라미터에만 집중한다.
- **objective 동일**: E1 채점 스칼라(§10-2에서 확정)를 두 방법에 그대로 사용. 충돌은
  실패 처리. 탐색은 단축 프로토콜, **상위 k개는 full 프로토콜로 재채점** 후 최종 채택.
- **예산 동일**: 두 방법 모두 동일 trial 수(~100) × 동일 단축 프로토콜 — "같은 예산을
  줬다"가 수치로 성립.

### 튜닝 로그 (trial마다 자동 기록)

`results/tuning/<method>/`:

| 파일 | 내용 |
|---|---|
| `study.db` | Optuna storage — 재현·중단 후 재개 |
| `trials.csv` | trial별 파라미터 · 점수 · 에피소드 수 · 시뮬 스텝 · wall-clock |
| `best_params.json` | 채택 파라미터 + 채택 근거(full 재채점 점수) — 컨트롤러가 직접 로드 |
| `budget.json` | 총합: trials / episodes / env_steps / wall_clock_s |

`aggregate.py`가 `budget.json`을 읽어 E4(b) 표의 "튜닝 비용" 열을 부모 논문과 동일
단위(샘플 수·초)로 자동 생성한다. 수동 튜닝이 남는 곳은 이 프로토콜 대상이 아니다:
Fixed-W(①)는 개발 이력(코드 주석의 실험 로그)이 곧 기록이고, PPO(③)는 학습 비용이
W&B에 이미 기록되어 있다.

## 7. 조건(env) 측 확장 — 전부 `env_variants.py`에서 파생으로

| 조건 | 방식 |
|---|---|
| E2 DR 스윕 ±25/50/75% | 기존 Eval cfg를 상속한 3개 cfg(유체계수 scale 범위만 오버라이드) + 신규 gym ID 등록 |
| E2 온라인 fine-tuning | `experiments/` 쪽에 `finetune_diff_wmpc.py` 신설: `DiffWMPCLearner` state_dict 로드 후 learning ON으로 단기 계속. 기존 train 스크립트의 eval-only `--ckpt` 의미는 건드리지 않음. optimizer state 저장이 필요하면 learner를 **서브클래스**로 확장 |
| E3 조류 step/sine | base env가 조류 미연결이므로 `WallScanEnv` **서브클래스**(신규 파일)에서 `OceanCurrent` 인스턴스를 hydrodynamics에 주입 + 시간 프로파일(step 급전환, sine) 드라이브. MPC 평가 시 모델은 공칭 유지(외란 강건성 시험이므로) |
| E4(a) ablation | preview on/off = 러너에서 frozen setpoint 생성으로 처리. `werr_ub` 500/5000 = WeightPolicy ctor 인자 노출(학습 변형은 `finetune`/신규 학습 스크립트 쪽에서). saturation-skip on/off = learner 서브클래스로 토글 |

## 8. 하드웨어 이식(E5) 대비 체크리스트

- [ ] `control/` + `estimator.py` + `scan_state_machine`/`mpc_reference`/`wall_frame_ekf`만으로
      폐루프 한 스텝이 도는 것을 네이티브 테스트로 보장 (isaaclab 없이)
- [ ] `ControlOutput.u`는 정규화 6-스러스터 — ESC 매핑만 하드웨어 쪽 책임
- [ ] PPO=onnx, WeightPolicy=torchscript export 스크립트 포함 (Jetson 배포물)
- [ ] acados 생성 C 코드의 Jetson(aarch64) 크로스 빌드 절차를 `bench_inference.py`와 함께 문서화
- [ ] `bench_inference.py`는 isaaclab 무의존 → Jetson에서 그대로 실행해 E4(c) 표 채움
- [ ] 코드 공개 경계: `control/`(정식화·EKF·태스크)는 공개 가능, `algorithms/diff_wmpc.py`
      학습 코어는 비공개 — 패키지 경계가 곧 공개 경계가 되도록 유지

## 9. 단계별 로드맵 (선행 순서 = experiments_plan.md 의존성 순서)

| 단계 | 내용 | 실행 환경 | 완료 기준 |
|---|---|---|---|
| P0 | `control/` types·base·estimator + ①③⑤ 어댑터 + 네이티브 테스트 | Windows PC | 테스트 통과. 기존 111개 테스트 무손상 |
| P1 | `run_experiment.py` + `configs/e1` + `aggregate.py` | Linux 호스트 | fixed/ppo/diff 3개 방법으로 E1 미니런(1 seed) → 기존 `run_wallscan_mpc.py`·`play.py` 결과와 지표 일치(회귀 검증) |
| P2 | §6 `tune.py` 공용 튜닝 파이프라인 / ④ SSI-MPC 이식 (적응 로직은 P0처럼 네이티브 테스트 먼저) → ② BO 튜닝 실행 → ④ 튜닝 실행 | 양쪽 | E1 5개 방법 × 5 seed 완주 → Table 1 + `budget.json` 2종 |
| P3 | E4(a) 토글 + `bench_inference.py` | 양쪽 | Table 2·3 (Jetson 실측은 후순위 가능) |
| P4 | `env_variants.py`: DR 스윕 → fine-tune → 조류 | Linux 호스트 | E2 곡선 + E3 |

**분업**: 이 Windows PC(iGPU, sim 불가)는 순수 계층 개발+테스트 전담, sim 실행은 기존
Linux 호스트(RTX 4080, Docker). P0 테스트는 기존 `tests/conftest.py` 스텁 패턴을 따르므로
torch만 있으면 된다 (현재 이 PC의 Python 3.13에는 torch 미설치 — P0 착수 시 설치 필요).

## 10. 그림(Figure) 계획 — 일반 비교 그림만 (2026-08-04 확정)

방법 특화 그림은 강조할 컨트리뷰션이 확정될 때까지 보류하고, 어떤 강조 방향에서도
유효한 **일반 비교 그림**만 기본 플랜으로 한다. 전부 `results/` 산출물에서 자동 생성
가능한 것들이며, 추가 로깅이 필요 없다.

| # | Figure | 내용 | 데이터 소스 | 대응 실험 |
|---|---|---|---|---|
| F1 | 본 비교 통계 | 방법별 mean±SD + **per-trial 점 오버레이** (min/max 막대 금지) | `table.csv` values 열 | E1 |
| F2 | 대표 궤적 | 벽면 전개도(s–z 평면) 스캔 경로 vs 레퍼런스, 방법별 | `trajectory_*.npz` | E1 |
| F3 | 강건성 곡선 | 성능 vs 섭동 강도(±25/50/75%), 방법별 라인+에러바+점 | `results/e2/metrics_*.json` | E2(a) |
| F4 | 외란 응답 시계열 | 추종오차·기울기 vs 시간, 조류 급전환 시점 수직선, 방법별 | `results/e3/trajectory_*.npz` | E3 |
| F5 | zero-shot vs fine-tuned | 막대 비교 (fine-tune은 제안 방법만) | E2(b) 실행 후 metrics | E2(b) |
| F6 | 비용 비교 | 학습/튜닝/추론 비용 (E4b·c 표의 그림판) | `budget.json` + `controller_cost` + W&B | E4(b,c) |

**보류 목록** (컨트리뷰션 확정 후 재도입 판단; 필요 시 스텝별 aux 로깅 선행):
- SSI-MPC 특화: 예측오차 감소 곡선, n_rf/lr 민감도 스윕(`e4_sensitivity_ssi.yaml`은
  부록/여력 시 전용으로 남김 — 기본 실행 플랜에서 제외)
- 제안 방법 특화: 가중치 적응 가시화(phase별 w 변화)
- 내부 진단: 튜닝 trial 산점도 (리뷰 대응 자료로는 보관, 본문 그림 아님)

SSI-MPC 논문에서 계승하는 것은 그림이 아니라 **관례**만: 조건당 trial 수(5회)·mean±SD
통계·계산시간 mean±SD 표, 그리고 하드웨어에서 하이퍼파라미터 축소·고정(그들은 시뮬
M=50/η=0.25 → 실기체 M=25/η=0.05) 관행. 원 논문이 검증해주는 구현 선택(feature에서
절대 위치 제외 = `INPUT_MASK_DEFAULT`, kernel_std는 feature 스케일 의존이라 튜닝 대상)은
유지한다.

## 11. 미결정 사항 (구현 지시 전 확정 필요)

1. **SSI-MPC 이식 조건** — 원 repo(UM-iRaL/SSI-MPC) 라이선스 확인·준거 커밋 고정·이식 범위 확정 (§4 ④)
2. E1 채점 스칼라 확정 — Table 1 지표 중 무엇의 가중합인지 (제안: 사이클 달성 + 추종 RMSE 역수, 충돌 시 실패 처리). §6 튜닝 objective로도 공용되므로 튜닝 착수 전 확정 필수
3. E2 fine-tuning 예산 — "quick online fine-tuning"의 스텝 수/세그먼트 정의
4. 조류 프로파일 수치 — step 크기·전환 시점, sine 진폭·주기 (부모 논문 세팅 대응)
5. Python 버전 정합 — 컨테이너는 3.11, 이 PC는 3.13; `control/`은 3.11 문법 기준으로 작성
