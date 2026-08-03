# PKRC Wall-Scan 학습 코드 분석 가이드

작성: 2026-07-24 (Claude Code 세션). 이 문서는 학습 파이프라인의 각 코드가 **어디에 있고, 무엇을 하고, 왜 그렇게 설계됐는지**를 실행 순서대로 따라간다. 경로는 레포 루트(`marinelab/`) 기준.

---

## 0. 30초 요약: 데이터가 도는 길

```
scripts/train.py                       ← 진입점: env 생성 + PPO 러너 실행
  └─ gym.make("Isaac-PKRC-WallScan-…") ← tasks/pkrc_wallscan/__init__.py 의 등록으로 해석
       └─ WallScanEnv (wallscan_env.py) ← 물리·센서·보상·종료의 본체
  └─ OnPolicyRunner.learn()            ← rsl_rl 라이브러리 (PPO 루프)
       매 iteration:
         [수집] 2048 env × 48스텝 롤아웃 → (관측31, 행동6, 보상) 98,304개
         [학습] GAE 어드밴티지 → PPO clip 업데이트 (8 epoch × 4 minibatch)
       리셋마다: DORAEMON이 동역학 DR 파라미터 재샘플 (Train 스테이지만)
```

---

## 1. 진입점 — `scripts/train.py`

Isaac Lab 상류의 표준 학습 스크립트를 marinelab이 그대로 쓴다. 핵심 5줄:

| 하는 일 | 코드 위치(함수) |
|---|---|
| CLI 파싱 (`--task --num_envs --max_iterations --resume --load_run` 등) | 파일 상단, `cli_args.add_rsl_rl_args` |
| env 생성 | `main()` 내 `gym.make(args_cli.task, cfg=env_cfg)` |
| rsl_rl 래핑 | `RslRlVecEnvWrapper(env)` — gym API를 rsl_rl이 먹는 텐서 배치 API로 변환 |
| 체크포인트 resume | `get_checkpoint_path(...)` → `runner.load(resume_path)` |
| 학습 실행 | `runner.learn(num_learning_iterations=…)` — 이후 제어권은 rsl_rl로 |

로그·체크포인트는 실행 CWD 기준 `logs/rsl_rl/<experiment_name>/<타임스탬프_runname>/` (우리 운용에선 isaaclab 루트에서 실행하므로 `isaaclab/logs/...`).

---

## 2. 태스크 등록 — `marinelab/tasks/pkrc_wallscan/__init__.py`

`gym.register` 5개(Stage1/Stage2/Stage3/Train/Eval)가 **문자열 entry-point**로 env·cfg를 가리킨다. 같은 `WallScanEnv`에 cfg만 달리 물려 커리큘럼을 만든다. 상단의 `_LAZY`(PEP 562)는 Isaac 미설치 환경에서도 geometry/sensors 같은 순수-torch 모듈을 임포트·테스트 가능하게 하는 장치.

---

## 3. 환경 본체 — `marinelab/tasks/pkrc_wallscan/wallscan_env.py`

`WallScanEnv(DirectRLEnv)`. DirectRLEnv의 스텝 순서는 고정이다:

```
_pre_physics_step(actions) → [물리 dt×decimation] → _get_dones → _get_rewards
→ (done env만) _reset_idx → _get_observations
```

### 3.1 물리 배선 (bluerov와 동일 패턴)
- `__init__`: `HydrodynamicsModel`(core/hydrodynamics.py — Fossen 모델: 부가질량·감쇠·부력·코리올리)과 `ThrusterModel`(core/thruster.py — TAM 6×6으로 6개 추력→렌치 변환, 1차 지연) 생성.
- `_pre_physics_step`: 액션 clamp(-1,1) → 스러스터 다이내믹스 적용.
- `_apply_action`: 추력 렌치 + 유체력 렌치를 합산해 PhysX에 인가.

### 3.2 상태 읽기 — `_read_state()` : **GT와 센서의 분리점 (sim2real의 심장)**
- GT: 위치·자세·속도·**명목 마운트 기준 소나 거리**(보상용).
- 센서: `sensors.py::apply_sensors`로 노이즈 + per-episode bias(DR)를 입힌 관측용 값. 소나는 **DR된 마운트 포즈**에서 측정(`geometry.sonar_wall_distance`).
- 원칙: **보상은 GT, 관측은 센서.** 정책이 관측 불능인 오차로 처벌받으면 수렴이 깨진다(이 세션에서 두 번 실증).

### 3.3 스핀 서치 게이트 — `_get_rewards()` 상단
리셋 직후 `search_omega`(0.63rad/s)로 yaw 레퍼런스를 한 바퀴 스위프하며 `scan_state_machine.search_step`으로 소나 최소값 방위(=최근접 벽)를 기록, 완료 시 per-env `_yaw_ref`에 잠금. 서치 중엔 z 레퍼런스를 수면(9.5)에 고정.

### 3.4 스캔 상태기계 — `scan_state_machine.py`
DESCEND(0)→SWAY_A(1)→ASCEND(2)→SWAY_B(3)→랩. `step()`은 reach 판정(|오차|<reach_eps 0.45가 reach_hold 10스텝 연속)으로 페이즈를 전진시키고, SWAY 진입 때 `z_hold`를 **GT z로** 래치(z_latch 인자 — 센서 bias가 보상 레퍼런스를 오염 못 하게). 타이밍용 sway 입력도 GT(`_s_gt`) — DVL bias 적분 드리프트 때문.

### 3.5 보상 — `_compute_reward_terms()` (12항)

| 항 | 식 | 가중치(cfg) | 근거 |
|---|---|---|---|
| wall_dist | exp(−\|GT벽거리−1.5\|/0.5) | ×5 | 임무 핵심 |
| heading | exp(−\|GT헤딩−yaw_ref_cur\|/0.3) | ×5 | 서치 중엔 스위프 추종, 이후 잠긴 방위 |
| depth | exp(−\|GTz−z_ref\|/0.5) | ×5 | Z_SCALE 0.5 = 평형오차를 밴드 안으로 |
| sway | exp(−\|GTs−s_ref\|/0.5) | ×5 | GT(_s_gt) — DVL 드리프트 격리 |
| progress | 25×(직전 목표거리−현재 목표거리) | — | potential-based shaping(Ng 1999): 이동 속도에 dense 보상, 최적성 보존 |
| upright | GT up_vec의 z | ×2 | 수직 기립 |
| waypoint | 페이즈 전진 1회당 | +200 | 스캔 진행 인센티브 |
| collision | 벽 0.4m 이내 스텝당 | −50 | |
| action_rate / action_mag / ang_vel | 제곱합 | 소액 − | 부드러움·에너지 |
| alive | 스텝당 | +0.1 | |

### 3.6 종료 — `_get_dones()`
충돌(0.4m, hard 스테이지만) / 경계(z<0.15 또는 z>10.2 — **물리 폭주 가드만**; 실제 상·하한은 수면과 바닥) / 전복(up_z<0.3) / 성공(3사이클) / 타임아웃(180s). 원인별 마스크는 `Scan/term_*` 텔레메트리로 분해 기록.

### 3.7 리셋 — `_reset_idx()`
수면(9.5) 스폰 + 디스크 xy 샘플 → DORAEMON 기록·재샘플(아래 §5) 또는 고정 DR → 센서 bias 재샘플 → 서치 암 → 텔레메트리(`Scan/end_phase·end_cycles·term_*`) 로깅. **순서 제약**: DORAEMON 성공 판정이 `_cycles` 0-리셋보다 먼저.

---

## 4. PPO — 하이퍼는 marinelab, 구현은 rsl_rl

### 4.1 하이퍼파라미터 — `tasks/pkrc_wallscan/agents/rsl_rl_ppo_cfg.py`
bluerov cfg(`tasks/bluerov/agents/rsl_rl_ppo_cfg.py`) 상속:

| 항목 | 값 | 의미 |
|---|---|---|
| 망 | actor·critic 각 [128,128,64] ELU | 관측31→행동6 가우시안(μ), σ 별도 파라미터 |
| num_steps_per_env | 48 | 롤아웃 길이 → 배치 98,304 |
| lr / schedule | 3e-4 / adaptive(desired_kl 0.01) | KL 초과 시 자동 인하 |
| clip / γ / λ | 0.2 / 0.99 / 0.95 | PPO 표준 |
| epochs × minibatch | 8 × 4 | |
| **entropy_coef** | **0 (오버라이드)** | `_kill_entropy_bonus` — clamp 액션에선 entropy 보너스가 σ 폭주(1→251 실측)를 부름 |

### 4.2 구현체 — Isaac 번들 `site-packages/rsl_rl/`
- `runners/on_policy_runner.py`: 수집↔학습 루프, 체크포인트 저장(50 iter마다), extras["log"] → TensorBoard.
- `algorithms/ppo.py`: GAE 계산, clip surrogate, clipped value loss, adaptive LR.
- `modules/actor_critic.py`: 가우시안 정책. `Mean action noise std` 로그 = σ (1.0 초과 상승은 적신호).

---

## 5. DORAEMON (Train 스테이지 전용 적응형 DR)

| 파일 | 역할 |
|---|---|
| `marinelab/algorithms/doraemon.py` | 엔진(ICLR 2024): 13차원 Beta 분포, IS 성공률 추정, KL 신뢰영역 내 엔트로피 최대화 |
| `tasks/pkrc_wallscan/doraemon_dr.py` | glue: `PARAM_DEFS` 13개(added_mass…time_constant), `build_scheduler`, `apply_xi`(코어 텐서 경로로 per-env 물리값 주입) |
| env `_reset_idx` | 에피소드 종료 시 (xi, return, 성공=사이클≥1, logp) 기록 → 새 xi 샘플·적용 → 주기적 `step()` → `DORAEMON/*` 메트릭 |

동작: 성공률≥α(0.5)면 DR 분포를 넓히고(로봇이 견디는 한 최대 다양성), 아니면 좁힘. **사이클이 완성돼야 비로소 가동된다** — 그래서 Stage3에서 사이클부터 완성하는 것.

---

## 6. 이 세션에서 실측으로 확정한 함정 (재발 방지 체크리스트)

1. **σ 폭주**: entropy_coef>0 + 액션 clamp = σ 무한 팽창. `Mean action noise std` 상시 감시.
2. **평형 vs 판정밴드 경합(3회 실측)**: 트래킹 보상 평형(~0.3m), 수면 부유 평형(9.5+)이 reach 밴드 바로 밖이면 정책은 그 자리에 주차한다. 목표·밴드를 정할 때 "노력 0으로 도달하는 상태"를 먼저 계산.
3. **킬 경계-밴드 간격**: 스폰·밴드·목표를 움직이면 상·하 킬 경계와의 잔여 간격을 반드시 검산(두 번 당함). 물리 한계(수면·바닥)가 실제 구속이면 킬 경계는 폭주 가드로만.
4. **원거리 무보상**: exp-tracking은 멀면 0 → 이동 속도를 보상하는 shaping 필요.
5. **진단은 계측으로**: `Scan/end_phase·end_cycles·term_*` 텔레메트리 + 결정론 정책 궤적 트레이스(스텝별 z/s/phase 덤프)가 blind 노브 조정보다 압도적으로 빠르다.

## 7. 실행 치트시트

```bash
# 학습 (isaaclab 루트에서)
./isaaclab.sh -p /root/home/rl_ws/marinelab/scripts/train.py \
  --task Isaac-PKRC-WallScan-Stage3-Direct-v0 --num_envs 2048 \
  --max_iterations 3000 --headless --run_name <이름> [--resume --load_run <run폴더>]
# 재생/영상:  scripts/play.py  (--video --video_length N | --real-time)
# 실시간 곡선: TensorBoard http://localhost:6006 (Settings→Reload data 켜기)
# 테스트:     PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./isaaclab.sh -p -m pytest marinelab/tests/ -q
```
