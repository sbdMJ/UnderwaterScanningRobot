# PKRC UUV 벽면 스캔 강화학습 프로젝트 보고서

작성일: 2026-07-24
대상: marinelab / Isaac Lab, PKRC UUV
브랜치: `feat/pkrc-wallscan-rl` (로컬, 미푸시 — 커밋 약 94개 중 본 프로젝트분 약 30개)

---

## 1. 개요

PKRC UUV(질량 22.8kg, 양성부력 +0.5kgf, T200 추력기 6개)가 지름 12m·높이 10m 원통형 물탱크 안에서 다음 임무를 수행하도록 end-to-end 강화학습(PPO)으로 단일 신경망 정책을 학습한다.

- 최근접 벽과 거리 1.5m 유지
- 최근접 벽 방향으로 헤딩 고정
- 수직 지그재그 스캔 반복: 하강 → sway 1m → 상승 → sway 1m

기존 Stonefish 고전 제어기(depth/wall/stability 개별 컨트롤러)를 대체하는 접근으로, 관측→6추력기 명령을 정책이 직접 출력한다(residual/보정 방식은 설계 단계에서 기각). 알고리즘은 rsl_rl의 PPO, 기존 hover 태스크(BlueROV) 하이퍼파라미터를 상속한다. sim-to-real을 전제로 도메인 랜덤화(DR)를 필수 요소로 설계했다.

## 2. 시스템 구성

### 2.1 로봇 임포트 및 자산

PKRC 로봇 자산(`assets/pkrc/`)은 기존 Stonefish 정의를 USD로 변환해 재사용했으며(커밋 `5cb1eff feat(pkrc): PKRC robot asset + hover task + convert/screenshot tools`), 벽면 스캔 태스크는 이 자산을 그대로 사용하고 물리 모델(`HydrodynamicsModel`, `ThrusterModel`)도 marinelab core에서 재사용했다. 신규 구현은 관측 조립·보상·웨이포인트 상태기계·종료 조건에 한정된다.

### 2.2 환경 설계

`WallScanEnv(DirectRLEnv)`가 환경 본체다. DirectRLEnv 표준 스텝 순서(`_pre_physics_step → _get_dones → _get_rewards → _reset_idx → _get_observations`)를 따른다.

**관측(31차원)** — 실기 센서 유래 값만 사용(카메라 원본 미포함):

| 블록 | 값 | 차원 | 출처 |
|---|---|---|---|
| 자세 | 상방벡터 + 헤딩 sin/cos | 3+2 | 3DM-GV7 INS |
| 각속도 | ω_body | 3 | INS |
| 선속도 | v_body | 3 | Water Linked DVL-A50 |
| 벽거리 | d | 1 | Ping Sonar |
| 수심 | z | 1 | Bar02 압력 |
| 위치추정 | XY(+yaw) + 유효플래그 | 3+1 | UKF-M(수면마커), 마커 미가시 시 valid=0 |
| 명령 | 벽거리·수심·sway 오차 + 스캔위상 sin/cos | 3+2 | 태스크(웨이포인트) |
| 헤딩오차 | 현재 yaw 레퍼런스 대비 sin/cos | 2 | 스핀서치의 시변 레퍼런스 관측용 |
| 서치플래그 | 스핀서치 진행 여부 | 1 | 모드 스위치 |
| 이전 행동 | prev thrust | 6 | 부드러운 제어용 |

설계 원칙은 **GT(ground truth)와 센서의 분리**다. 보상은 GT 값으로 계산하고, 관측은 노이즈·바이어스가 실린 센서 값으로 조립한다. 정책이 관측 불능인 오차로 처벌받으면 수렴이 깨진다는 것이 이 세션에서 두 차례 실증됐다(코드 가이드 §3.2).

관측 차원은 구현 도중 28→31로 확장됐다. Task6 최초 구현·`final-fix-report.md` 검증 시점(스모크 obs shape `[4,28]`)에는 28차원이었고, 스핀서치 도입(커밋 `708748e` — 수면 스폰 + 최근접벽 탐색)으로 헤딩오차 sin/cos(2)와 서치플래그(1)가 추가되어 현재 `wallscan_env_cfg.py`의 `observation_space: int = 31`(위 표 구성)로 안정화됐다.

**보상(12항)** — 벽거리·헤딩·수심·sway는 `exp(−|오차|/scale)` 트래킹 항(각 가중 ×5), potential-based progress shaping(목표거리 감소분 ×25), upright(×2), waypoint 도달 보너스(+200), 충돌 페널티(−50), action_rate/action_mag/ang_vel 소액 페널티, alive(+0.1)로 구성된다. 상세는 §4 표 참조.

**종료 조건**: 충돌(0.4m 이내, hard 스테이지만) / 경계(z<0.15 또는 z>10.2, 물리 폭주 가드) / 전복(up_z<0.3) / 성공(3사이클 완주) / 타임아웃(180초).

### 2.3 커리큘럼 (4단계)

풀 지그재그를 처음부터 던지면 벽충돌·발산으로 학습이 깨진다는 설계 판단에 따라 단계적으로 확장했다.

| 스테이지 | 내용 |
|---|---|
| Stage1 | 제자리 + 벽거리 + 헤딩 유지(station-keeping), 벽충돌 종료 완화 |
| Stage2 | + 수직 추종(하강·상승), sway 미적용 |
| Stage3 | + sway 지그재그 웨이포인트 스캔(풀 태스크, no DR) — 사이클 완성 확인용 |
| Train | 풀 태스크 + DR(DORAEMON 적응형) |
| Eval | 풀 태스크 + 고정 넓은 DR |

Stage1/Stage2/Stage3/Train/Eval 5개 태스크는 모두 동일한 `WallScanEnv`에 cfg만 달리 물려(`gym.register`의 문자열 entry-point로 env·cfg 지정) 구현했다 — 커리큘럼 단계마다 별도 env 클래스를 만들지 않는 설계다.

### 2.4 DORAEMON 적응형 DR

Train 스테이지에서 고정-균등 DR 대신 DORAEMON(Tiboni et al., ICLR 2024) 적응형 스케줄러를 사용한다. 13차원 Beta 분포로 동역학 파라미터(added_mass, linear/quadratic damping, volume, CoB/CoG offset×3, inertia, thrust_coefficient, time_constant)를 관리하며, 중요도샘플링(IS) 기반 성공률 추정으로 성공률≥α(0.5)면 KL 신뢰영역 내에서 분포를 넓히고(엔트로피 최대화), 실패가 늘면 좁힌다. 센서 바이어스 DR(6개 노브)은 DORAEMON 범위 밖으로 두고 기존 고정-균등 방식을 유지한다.

통합 설계의 핵심 결정:
- 스케줄러는 env-owned(`WallScanEnv` 내부 `_reset_idx`/`_get_rewards`에서 구동)
- 에피소드 성공 판정은 스캔사이클 기반: `success = (_cycles >= doraemon_success_cycles)`
- per-env 값 적용은 core API를 튜플-또는-텐서 겸용으로 확장(`HydrodynamicsModel.scale_parameters`, `ThrusterModel.randomize_parameters`) — 기존 BlueROV 튜플 경로는 무변경(하위호환)
- DORAEMON 성공 판정에 쓰는 `doraemon.success_cycles`(기본 1)는 태스크 자체의 성공 종료 기준(3사이클)과 별개 노브다 — 커리큘럼 성공 bar를 태스크 성공보다 낮게 잡아 학습 초기 success_rate가 0에 고착되는 것을 방지하는 의도된 설계(Plan2 최종 리뷰에서 조정)

MVP 한계로 명시된 항목: 스케줄러 체크포인트는 `train.py --resume` 간 재개되지 않음(DR 분포가 초기 집중도로 재시작, 정책 가중치는 정상 재개 — 후속 작업으로 이연), Eval 스테이지에서 커리큘럼 리플레이 미지원(`export_recording` 존재, 후속 작업), 타임아웃이면서 `cycles >= success_cycles`인 에피소드도 성공으로 집계(의도된 단순화).

### 2.5 테스트 전략

설계 스펙(§11)이 규정한 검증 레벨:

| 레벨 | 방법 |
|---|---|
| 유닛 | 각 센서 변환 함수: 참값 입력 → 기대 노이즈/드롭아웃 통계 assert(자체 self-check) |
| 유닛 | 웨이포인트 상태기계: 합성 궤적으로 phase 전진·s_ref 증가 검증 |
| 유닛 | 소나 레이캐스트: 원통 중심에서 알려진 거리 assert |
| 스모크 | 각 커리큘럼 스테이지 2-iter 학습(에러 없음 + 보상 상승) |
| 수렴 | 4096 env 전체 학습 후 평가 지표(충돌율·추종 RMSE) 임계 이하 확인 |

DORAEMON 통합 스펙(§9)은 추가로 param_defs/nominal 구성 검증, core 텐서 경로의 통계적 경계 검증, 성공 계산 로직 검증을 유닛 레벨에, Train 태스크 16 envs/2 iters 스모크(DORAEMON 메트릭 출력 + 기존 BlueROV hover 스모크 하위호환)를 요구했다. 두 스펙 모두 §3.1·§3.3에서 보고한 대로 전량 그린으로 완료됐다.

## 3. 실험 캠페인 요약

### 3.1 Plan 1 — 태스크 구축 (Tasks 1-8)

`.sp/plans/2026-07-22-pkrc-wall-scan-rl.md`를 `subagent-driven-development`(태스크마다 신규 구현자 + 스펙준수 리뷰어 + 코드품질 리뷰어)로 실행. 순서: 벽거리 지오메트리(Task1) → 센서 추상화(노이즈+드롭아웃, Task2) → 웨이포인트 스캔 상태기계(Task3, z_ref latch 버그 수정 포함) → 원통 탱크 spawn(Task4) → env cfg(Task5) → `WallScanEnv`(Task6, 관측28·보상 GT 기반·N사이클 성공 종료) → agents+gym 등록(Task7) → 커리큘럼 스모크(Task8, Stage1/Stage2/Train 전부 EXIT=0).

각 태스크는 독립 리뷰(스펙 준수 + 코드 품질)를 거쳐 clean/Approved 판정을 받았다. 리뷰 과정에서 추적된 사소 항목(minor, track): Task2에서 up_vec 노이즈 적용 후 미정규화, 노이즈 기본값이 placeholder(설계 스펙 §13 미해결 항목), INS drift/DVL 바텀락 드롭아웃 미구현(스펙 §7·§10) — 모두 "최종리뷰/DR 튜닝에서 판단" 태그로 이연. Task6 초기 구현에서 관측 28차원·소나 마운트 미보정 상태였던 것을 이후 별도 커밋으로 보강(§2.2 참조).

최종 리뷰(opus): READY-WITH-FOLLOWUPS, must-fix 2건은 커밋 `90fb189`에서 수정(탱크 콜리전 비활성화, 사용되지 않는 spawn-position DR 오버라이드 제거). 나머지(INS drift/DVL 드롭아웃/소나 빔 모델, exp-scale·노이즈·보상가중치 크기, up_vec 재정규화)는 sim2real 튜닝 단계로 이연.

`final-fix-report.md` 상세: 탱크 콜리전을 끈 이유는 GPU PhysX에서 로봇이 solid volume 안에 스폰되며 depenetration으로 인한 스퓨리어스 이젝션이 발생했기 때문(벽거리/충돌 판정은 별도로 analytic 레이캐스트를 쓰므로 물리 콜리전이 불필요). 검증: `AppLauncher` 헤드리스 스크립트로 obs shape `[4,28]`, 5스텝 보상 전부 finite, 스퓨리어스 종료 0건 확인.

### 3.2 소나 마운트 + DR 보강 (개별 커밋)

Plan1 완료 후 Train launch 전 보강: 소나 마운트를 실측 지오메트리 기준으로 보정(`b5e49ee`, `dfba593`), 센서별 per-episode DR 바이어스 추가(`d8b4c91`), DR 바이어스가 GT 비교 보상(depth/sway)에 새는 버그 수정(`0b279ff` — 보상은 GT, 관측만 센서라는 원칙을 재확인한 수정).

### 3.3 Plan 2 — DORAEMON 통합 (Tasks 1-5)

`.sp/plans/2026-07-22-doraemon-wallscan-integration.md`를 동일하게 `subagent-driven-development`로 실행. core 텐서 경로 확장(Task1: `scale_parameters`, Task2: `ThrusterModel.randomize_parameters`, dtype 캐스트 버그 수정 커밋 `469bb5c` 포함) → glue 모듈(Task3: 13개 param_defs, `build_scheduler`, `apply_xi`) → env lifecycle 배선(Task4: record→sample→apply→step, record-before-cycles-zero 순서 검증됨) → 스모크(Task5: DORAEMON 메트릭 10건 출력 EXIT=0, BlueROV 하위호환 스모크 EXIT=0, pytest 81 passed).

최종 리뷰(opus): READY-WITH-FOLLOWUPS, must-fix 0건. 이연 항목: `success_cycles` 명칭 혼동 우려(선택적 rename), 중복 `.to(device)` 호출, `PARAM_DEFS` 주석 명확화. 관찰 지침(watch-the-log): `DORAEMON/entropy` 상승 및 `success_rate>=alpha(0.5)`가 수용 신호, `min_ess_ratio=0.01` 완화값은 ESS 이상 시 상향 필요.

### 3.4 수렴 캠페인 (개별 커밋, iteration별 증상→진단→수정)

Train launch(사용자 승인) 이후 실제 학습 로그를 보며 다음 이터레이션을 거쳤다(커밋 순, 오래된 것부터):

| # | 증상 | 수정 | 커밋 |
|---|---|---|---|
| 1 | 40초 에피소드로는 풀 사이클 도달 불가 | 에피소드 40s→120s | `a56688e` |
| 2 | sway 페이즈 진행이 막힘(로봇이 판정선 밖에서 호버링), DVL 바이어스가 비현실적 | GT sway로 페이즈 타이밍 전환, 현실적 DVL 바이어스, waypoint 보너스 강화, reach_eps 완화 | `778ed69` |
| 3 | 상방 스폰(z=9.5)이 킬 경계선(9.5) 위에 정확히 걸침, 스캔 진행 텔레메트리 부재 | z 상한 9.5→9.8, `Scan/*` 진행 텔레메트리 추가 | `574fb99` |
| 4 | 사이클 시간 예산 부족(120s로도 서치+스캔 여유 없음) | 에피소드 120s→180s, 스핀서치 20s→10s | `e2dc798` |
| 5 | 원거리 무보상(exp-tracking이 목표에서 멀면 ≈0) | potential-based progress shaping(z/s 목표거리 기반) 추가 | `8a59341` |
| 6 | σ(행동 노이즈 표준편차) 무한 팽창(1→251 실측), 제어가 bang-bang으로 퇴화 | `entropy_coef=0`으로 오버라이드(`_kill_entropy_bonus`) | `8760401` |
| 7 | reach 밴드가 트래킹 평형(≈0.3m)보다 좁아 페이즈 전진이 안 터짐 | reach_eps 0.2→0.3, Z_SCALE 1.0→0.5 | `557ddf2` |
| 8 | Stage3(풀 사이클, DR 없음)로 사이클 완성 자체를 먼저 검증할 필요 | Stage3 커리큘럼 단계 신규 추가 | `5eed662` |
| 9 | 상단 킬 경계가 탱크 rim보다 낮고, reach 밴드가 관측된 파킹 지점을 못 덮음 | 천장 킬 경계를 rim 위로, reach 밴드를 관측 평형 위로 재산정 | `585d4da` |
| 10 | 바닥 킬 경계(0.5)가 상단 수정과 비대칭 | 바닥 킬 경계 0.5→0.15(천장 수정의 대칭 미러) | `c7e9b02` |
| 11 | ascend 밴드(z_top 9.0)가 자유부양 평형보다 낮음 | z_top 9.0→8.5 | `2447242` |

각 수정은 §4의 "킬 경계-밴드 간격" 함정을 반복적으로 노출했다(이 세션에서 두 번 실측 — 항목 9, 10, 11이 그 반복). 진단은 매번 `Scan/end_phase·end_cycles·term_*` 종료-원인 분해 텔레메트리와 결정론 정책 궤적 트레이스(스텝별 z/s/phase 덤프)로 수행했으며, blind 노브 조정보다 빠르게 근본 원인을 특정했다.

캠페인 종료 시점에 코드 구조·데이터 흐름·함정을 정리한 분석 가이드를 작성(`254a884 docs: wall-scan training-code analysis guide`).

## 4. 핵심 발견 (함정 5개)

이 세션에서 실측으로 확정한 함정과 각각의 재발 방지책이다(`docs/wallscan-training-code-guide.md` §6 기준).

| # | 함정 | 메커니즘 | 재발 방지책 |
|---|---|---|---|
| 1 | σ 폭주 | `entropy_coef>0` + 액션 clamp(-1,1) 조합에서 σ가 1을 넘어도 행동이 안 변해 entropy 보너스가 공짜 보상이 됨 → σ 1.0→251까지 단조 폭주, 제어가 bang-bang으로 퇴화. 짧은 런(~600 iter)에선 안 보이고 수천 iter에서 폭발 | `Mean action noise std` 로그를 상시 감시(1.0 초과 상승 = 적신호), 클램프 액션 태스크는 `entropy_coef=0` 기본값으로 시작 |
| 2 | 평형 vs 판정밴드 경합(3회 실측) | 트래킹 보상 평형(≈0.3m), 수면 부유 평형(9.5+)이 reach 밴드 바로 밖이면 정책이 그 자리에 주차 | 목표·밴드 설계 시 "노력 0으로 도달하는 평형 상태"를 먼저 계산하고 밴드를 그보다 넓게 설정 |
| 3 | 킬 경계-밴드 간격 | 스폰·밴드·목표를 움직이면 상·하 킬 경계와의 잔여 간격이 깨짐(세션에서 두 번 당함) | 스폰 위치·reach 밴드·목표(z_top/z_bottom) 수정 시 반드시 상·하 킬 경계와의 잔여 간격을 함께 검산. 물리적 한계(수면·바닥)가 실제 구속이면 킬 경계는 폭주 가드로만 둘 것 |
| 4 | 원거리 무보상 | exp-tracking 보상은 목표에서 멀면 ≈0이라 이동 속도를 보상하지 않음 | potential-based progress shaping(`k*(prev_dist-curr_dist)`) 추가, ref 전환 시 공짜 보상이 나지 않도록 양측 거리를 현재 ref 기준으로 계산 |
| 5 | 진단은 계측으로 | blind 노브 조정 3연속 실패 후에야 계측 전환이 돌파구가 됨 | `Scan/end_phase·end_cycles·term_*` 텔레메트리 + 결정론 정책 궤적 트레이스(스텝별 z/s/phase 덤프)를 처음부터 extras log에 넣을 것 |

## 5. 현재 상태 및 남은 일

최신 커밋(`2447242 fix(wallscan): z_top 9.0 -> 8.5`) 기준 stage3_fix3 학습이 진행 중이다. 판정 기준은 waypoint 800(1사이클) 돌파 여부 — 지금까지의 iteration은 사이클 완성 자체가 막혀 있었으므로, 이 기준을 넘는지가 다음 확인 지점이다.

사이클 완성이 확인되면 남은 절차는 다음과 같다(코드 가이드 §5 근거, DORAEMON은 "사이클이 완성돼야 비로소 가동" — Stage3에서 사이클부터 완성시키는 순서 설계였다):

1. Train 스테이지 재개 — DR(DORAEMON 적응형) 활성화 상태에서 수렴 확인. `DORAEMON/entropy` 상승 & `DORAEMON/success_rate>=alpha(0.5)`가 수용 신호(3.3절 관찰 지침).
2. Eval 스테이지 검증 — 고정 넓은 DR에서의 일반화 확인. 평가 지표는 벽거리 RMSE, 헤딩 RMSE, sway/depth 추종오차, 사이클 완주율, 충돌율(설계 스펙 §9 기준, 구체 임계값은 미확정).

로컬 커밋은 총 94개 중 wall-scan 프로젝트 관련이 약 30개이며 원격에 푸시되지 않은 상태다. 브랜치 정리(머지/PR) 여부는 사용자 결정 대기.

## 부록

### A. 실행 커맨드 치트시트

```bash
# 학습 (isaaclab 루트에서)
./isaaclab.sh -p /root/home/rl_ws/marinelab/scripts/train.py \
  --task Isaac-PKRC-WallScan-Stage3-Direct-v0 --num_envs 2048 \
  --max_iterations 3000 --headless --run_name <이름> [--resume --load_run <run폴더>]

# 재생/영상
scripts/play.py  (--video --video_length N | --real-time)

# 실시간 곡선
TensorBoard http://localhost:6006 (Settings→Reload data 켜기)

# 테스트
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./isaaclab.sh -p -m pytest marinelab/tests/ -q
```

로그·체크포인트 경로: 실행 CWD 기준 `logs/rsl_rl/<experiment_name>/<타임스탬프_runname>/` (isaaclab 루트에서 실행 시 `isaaclab/logs/...`).

### B. 주요 파일 경로

| 경로 | 역할 |
|---|---|
| `scripts/train.py` | 학습 진입점(env 생성 + PPO 러너 실행) |
| `marinelab/tasks/pkrc_wallscan/__init__.py` | `gym.register` 5개(Stage1/Stage2/Stage3/Train/Eval) |
| `marinelab/tasks/pkrc_wallscan/wallscan_env.py` | 환경 본체(`WallScanEnv(DirectRLEnv)`) |
| `marinelab/tasks/pkrc_wallscan/wallscan_env_cfg.py` | 관측/행동 공간, 보상 가중, 커리큘럼 cfg |
| `marinelab/tasks/pkrc_wallscan/sensors.py` | GT→센서 변환(노이즈·바이어스·드롭아웃) |
| `marinelab/tasks/pkrc_wallscan/scan_state_machine.py` | 웨이포인트 phase 전진 + z_ref/s_ref 생성 |
| `marinelab/tasks/pkrc_wallscan/tank.py` | 원통 탱크 spawn cfg(콜리전 비활성화, 시각용) |
| `marinelab/tasks/pkrc_wallscan/doraemon_dr.py` | DORAEMON glue(param_defs, build_scheduler, apply_xi) |
| `marinelab/algorithms/doraemon.py` | DORAEMON 엔진(Beta 분포, IS 성공률 추정) |
| `marinelab/tasks/pkrc_wallscan/agents/rsl_rl_ppo_cfg.py` | PPO 하이퍼파라미터(entropy_coef=0 오버라이드 포함) |
| `marinelab/core/hydrodynamics.py` | Fossen 유체동역학 모델(`scale_parameters` 텐서 경로) |
| `marinelab/core/thruster.py` | 추력기 TAM·1차 지연 모델(`randomize_parameters` 텐서 경로) |
| `docs/wallscan-training-code-guide.md` | 코드 구조·데이터 흐름·함정 분석 가이드(1차 소스) |
| `.sp/specs/2026-07-22-pkrc-wall-scan-rl-design.md` | 태스크 설계 스펙 |
| `.sp/specs/2026-07-22-doraemon-wallscan-integration-design.md` | DORAEMON 통합 설계 스펙 |
| `.superpowers/sdd/progress.md` | SDD 태스크별 진행 레저 |
