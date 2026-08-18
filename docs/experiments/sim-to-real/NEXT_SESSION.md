# ⚠ NEXT_SESSION — sim-to-real 후속 세션 메타 프롬프트 (일회용)

> **이 문서의 규약 (반드시 따를 것):**
> 1. 이 문서는 sim-to-real 작업을 이어받는 **새 세션이 가장 먼저 읽는 문서**다.
>    세션 메모리 없이 이 문서만으로 전체 여정과 다음 할 일을 100% 복원할 수 있게
>    작성되었다.
> 2. **일회용이다.** 읽고 후속 작업에 착수한 직후(첫 실질 작업 커밋에 포함해서),
>    ① 이 파일을 `git rm`으로 삭제하고 ② `CLAUDE.md` 안의 "sim-to-real 시작 전
>    필수" 절(이 파일을 가리키는 절)도 함께 삭제하라. 두 번 세 번 반복해 읽는
>    낭비를 막기 위함이다. 삭제해도 정보는 사라지지 않는다 — 여기 적힌 모든
>    영구 정보의 정본은 `README.md`(레저)·`thruster_mapping.md`·
>    `../../../marinelab/ros/mj_ws_README.md`(런북)에 이미 있고, 이 문서는 그
>    포인터와 "지금 어디까지 왔나"의 스냅샷일 뿐이다.
> 3. 작업이 다시 세션 경계에 걸리면(미완인 채 세션이 끝나면), 같은 규약으로 이
>    문서를 **그 시점 기준으로 새로 작성**해 남기고 CLAUDE.md 포인터를 복원하라.

## 1. 프로젝트 한 문단 (컨텍스트 제로에서)

ICRA 논문용 비교 실험: 원통 수조(R=6 m) 벽면 스캔 UUV(PKRC, 22.8 kg, T200×6)에
대해 SSI-MPC(온라인 적응 NMPC)를 Fixed-W NMPC/BO-NMPC/PPO와 비교한다. 시뮬 실험
E1–E4는 사실상 완료(diff-WMPC만 타 머신 대기), **이 브랜치(`feature/sim-to-real`)는
E5(실기체) 축**: 실제 로봇의 하드웨어 식별 → Jetson 배포 → 수조 폐루프 검증 →
본실험(Table 4)이 여정이다. 주장 한정: "시뮬 순위가 실기체에서 유지된다".

## 2. 여정 전체 지도 (완료 표기 포함)

정본 레저: [`README.md`](README.md) §1 (커밋 해시까지 전부 기록). 요약:

| 단계 | 내용 | 상태 |
|---|---|---|
| A/B | 센서 실측 특성화 → EKF-in-loop 시뮬 게이트 (s-보정 수정, vis7 통과, E5 go) | **DONE** |
| C | HW 식별: 추력맵 k/I₀ 실측, 항력 실측(시뮬이 5–7× 과대), heave k=0.99 N/A, 트림 +0.24 N, 배포 plant `pkrc_plant_hw2026.json` | **DONE** |
| C-② | 공유 폐루프 코어(`WallScanControlLoop`) + ROS 브리지 3노드 (estimator/controller/mapper) | **DONE** |
| D | Jetson: acados aarch64 빌드, E4(c) 벤치 완성 — **배포 설정 h20/rti4 = 15.6–16.2 ms** (기본값 반영), mj_ws 배포 워크스페이스 | **DONE** |
| E-① | 시나리오 ② 체인 라이브니스 (소형 수조, enable OFF) | **DONE — PASS** (2026-08-16) |
| E-② | 시나리오 ③ depth-hold 폐루프 (소형 수조) | **진행 중 ← 지금 여기** |
| E-③ | 본 탱크: 마커 측량 → 단계적 폐루프 (depth-hold → wall-align → wallscan) | 미착수 |
| E-④ | E5 본실험 (방법별 3–5회, bag → `eval_metrics` → Table 4) | 미착수 |

## 3. 지금 여기 — 시나리오 ③의 정확한 상태 (2026-08-18 03:41 bag까지)

2026-08-18 밤 현장 세션에서 depth-hold 폐루프를 6개 bag에 걸쳐 디버깅했고,
**근본원인 7개를 순차 규명·수정했다** (전부 커밋됨; 레저 E-② 행과 각
`marinelab/scripts/experiments/hw_bag_depthhold*_20260818.py` docstring이 판정 기록):

1. 허구 수평 오차(마커리스 블라인드 앵커 드리프트)가 heave까지 지배 → 바닥 고착
   → **`depth_only:=true`** (수평 werr 6채널 0)
2. IMU가 NED/마운트 규약으로 롤 ±180° 도착 → MPC가 뒤집힌 줄 알고 심도 반전 제어
   → **`imu_mount_rpy_deg:="[180.0,0.0,0.0]"`**
3. 매퍼가 리스트 파라미터 셸 인용 실패로 2세션 미캘리브레이션 구동 → **실측
   캘리브레이션을 노드 기본값으로 굽고** 기동 로그에 k/I₀ 출력
4. `reach_eps` 기본 0.6이 소형 수조에서 z_ref를 enable 심도에 래치 → **0.05**
5. 아크릴 수조에서 DVL 완전 사망(valid 0%) → EKF vz 동결 → MPC 감쇠 상실
   → **`vz_from_depth:=true`** (압력심도 미분 + LPF; 03_41 bag에서 corr 0.88 검증)
6. **위상머신 순환** (03_41 bag의 마지막 원인): z_top=z_bottom이면 도달 조건이
   항상 참 → 34 s에 49회 위상 전환, SWAY 진입마다 z_ref를 현재 심도로 재래치
   → ±4 cm 사각파 주입 → 15 cm 리밋사이클 → **`hold_z` 모드 신설** (상태머신
   완전 우회; `scan_loop.py` + 컨트롤러 `-p hold_z:=`, 커밋 `ce09488`)
7. (운영 함정) `depth_only` 플래그가 현장에서 두 번 누락됨 → 런북에 enable 후
   체크리스트 추가 (u[0..3]≈0 확인)

**미검증**: `hold_z` + `depth_only` 조합의 현장 합격 bag이 아직 없다.
로컬 `~/mj_ws`는 hold_z 포함 상태로 재조립 완료 — **Jetson에는 아직 rsync 안 됨**
(사용자가 함). GitHub PR은 사용자 요청으로 본문까지 준비했으나 `gh` 미인증으로
생성 못 함 — 본문은 웹에서 만들라고 전달됨 (아래 §6).

## 4. 바로 다음 할 일 (우선순위 순)

### 4-1. 시나리오 ③ 재시도 결과 대기/분석 ← 최우선

사용자가 수조에서 재시도하고 bag 경로를 준다. 절차는 런북
[`mj_ws_README.md`](../../../marinelab/ros/mj_ws_README.md) §9 "터미널별 상세"가
정본 (rsync → **T1–T3 전부 재시작** → T1 first-IMU roll≈0 + T2 WARN 2개
`DEPTH-ONLY`/`DEPTH-HOLD` + T3 `[CALIBRATED]` 확인 → enable → bag).

bag 분석 방법: `marinelab/scripts/experiments/hw_bag_depthhold_0341_20260818.py`를
템플릿으로 (rosbags 파이썬 분석; dvl_msgs는 `~/PKRC로봇_코드_및_데이터/hero_ws`에서
등록). **합격 판정**: 잔여 진동 ±2–3 cm 수준(데드존 릴레이 한계 — 필요 정적힘
0.24 N < 최소 실현힘 0.37 N이라 완전 정지는 구조적으로 불가), heave u 비포화,
phase 0 고정, 10 cm 눌렀다 놓기 복원. 합격이면: 레저 E-② 행 DONE 갱신 + 분석
스크립트 커밋. **불합격이면 knob 돌리기 금지 — bag에서 원인 규명부터**
(이 캠페인의 전 과정이 그 규율로 진행됐다).

### 4-2. 합격 후 같은 수조에서

- `method:=ssi` 재시도 (런북 §9 T2 주석 참조; **바닥 접촉 있었던 세션에서는
  재-enable 전 접촉 잔차 학습 오염 주의**)
- 마커 리깅 가능해지면 ③ 본문(마커 있는 버전, 수평 1.5 A) 승격

### 4-3. 본 탱크 (Phase E-③/④, 레저 §3 Phase E 체크리스트)

탱크-마커 측량(`marker_x/y/yaw`) → 7 m 가시성 현장 확인 → 단계적 폐루프
(depth-hold → wall-align → wallscan; 하강 ≥0.2 m/s는 본 탱크에서만 시연 가능 —
소형 수조는 1.7 s 구간 하한 0.14 m/s까지 확인됨, 레저 C-⑥) → 본실험 3–5회 ×
{Fixed-W, SSI, PPO onnx} → bag을 `eval_metrics`로 시뮬 동일 지표화 → Table 4.

### 4-4. 전부 끝나면

PR(§6) 체크리스트 채우고 main 머지 (사용자 합의: 다 끝나기 전 머지 금지).

## 5. 이 축의 함정 (하나라도 무시하면 세션 하나가 날아간다)

- **rsync는 실행 중 노드에 반영 안 됨** — 코드 배포 후 T1–T3 전부 재시작
- `/teleop/thruster_currents`는 **manual 모드에서만 실측** — auto 중 값은 참고만
- cmd pub 공백 >0.5 s → teleop이 stale로 manual depth-hold(목표 0.6 m PID) 복귀
  — "auto인데 이상하게 움직인다"의 단골 원인
- bar10xt가 이 소형 수조에서 ~+0.5 m 오프셋: state z 바닥 0.016 / 부유 ~0.25,
  Z_HOLD는 **현장에서 읽어서** 0.10–0.15 근방
- Ping1D는 소형 수조에서 멀티패스(0.8–5.3 m 난사) — wall 입력으로 쓰지 말 것
- 창 평균 u↔I 비율은 뱅뱅 구간에서 왜곡 — **포인트와이즈로 대조** (이걸로 오진
  1회: 매퍼 stale로 잘못 판정했다 철회, `hw_bag_depthhold_20260818.py` 부록)
- 테더 개입 가능성 있는 bag으로 힘 평형 식별 금지 (§4e 철회 사례)
- 컨테이너 산출물 root 소유 → 커밋 전 chown; ≥30분 작업은 CronCreate 모니터링
  (CLAUDE.md 규칙)

## 6. 걸려 있는 것들 (코드 외)

- **PR 미생성**: `feature/sim-to-real` → `main`. 제목/본문 준비돼 사용자에게 전달됨
  (체크포인트 PR, "do not merge yet" + 머지 전 체크리스트 4항목). 사용자가 웹에서
  만들거나 `gh auth login` 후 요청할 것.
- 남은 시뮬 축(이 브랜치 무관): diff-WMPC 이식(`docs/diff_wmpc_port_todo.md`,
  타 머신 대기), E2(b)/F5, E4(a).

## 7. 정본 문서 지도

| 무엇 | 어디 |
|---|---|
| 진행 레저 (커밋 추적) | `docs/experiments/sim-to-real/README.md` |
| 현장 런북 (터미널별 명령) | `marinelab/ros/mj_ws_README.md` (= mj_ws의 README) |
| HW 식별 데이터·프로토콜 | `docs/experiments/sim-to-real/thruster_mapping.md` |
| Jetson acados + E4(c) 표 | `docs/experiments/sim-to-real/jetson_acados_build.md` |
| 현장 세션 판정 기록 | `marinelab/scripts/experiments/hw_bag_*.py` docstring |
| 배포 plant | `marinelab/config/pkrc_plant_hw2026.json` |
| 워크스페이스 조립 | `marinelab/ros/make_mj_ws.sh` (출력 `~/mj_ws`) |
| 테스트 (318개, sim 불필요) | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ~/.conda/envs/acmpc_sim/bin/python -m pytest marinelab/tests/ -q` |
