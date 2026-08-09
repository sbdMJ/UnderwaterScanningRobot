# Sim-to-Real (E5) — 진행 내역과 계획

> 브랜치 `feature/sim-to-real`. 마지막 갱신: 2026-08-09.
> 대상 방법: SSI-MPC 우선 (파이프라인 검증 목적 — 온라인 적응까지 있는 가장 까다로운
> 방법이 실기체에서 돌면 나머지는 부분집합), 이후 Fixed-W NMPC / PPO / diff.
> 상세 데이터·표의 정본: [`../hw_sensor_characterization.md`](../hw_sensor_characterization.md).

## 0. 요약 (한 문단)

시뮬 GT 상태로만 검증돼 있던 스택을 실기체 조건으로 가져가기 위해, ① 실제 로봇
bag 2개로 센서를 실측 특성화하고, ② 그 조건의 EKF-in-loop 시뮬에서 **상태추정이
유일한 블로커**(공통모드 5–7× 열화)임을 확인, ③ 원인을 "절대 fix의 방위각을 버려서
s(스캔 진행도)가 미보정 적분기로 남는 것"으로 코드 수준에서 특정하고 수정, ④ 실측
조건 전부(노이즈·주기·ArUco 1.3 Hz·가시성 7 m)를 적용한 재검증에서 **GT 대비
+5~13%로 게이트 통과 (E5 go)**. 이어서 하드웨어 배포물(estimator 브리지, 공유
폐루프 코어)을 구현 중이다.

## 1. 완료 내역 (커밋 추적)

| 단계 | 내용 | 커밋 |
|---|---|---|
| B-0 | EKF-in-loop 프리체크 (placeholder 센서): ssi 7.2× / nominal 5.5× 열화 — **red-flag** | `8e0699b` |
| A | 122731 bag 센서 특성화 — 노이즈는 placeholder가 5× 비관, **주기는 5–10× 낙관** (DVL 9.5 Hz, 깊이 5 Hz). 이 bag의 UKFM은 순수 추측항법이었음(ArUco 0건) | `324eb39` |
| B-1 | 실측 센서 모델: `SensorCfgHW2026Bag`(+Aruco), `SensorRateGate`, SimSensorStream rate-and-hold | `bd47e68` |
| B-2 | measured 조건 10셀: φ는 0.4°로 회복되나 objective 불변 — **근본원인 = `update_ukfm`의 H에 s행이 없음** (s가 유일한 미보정 적분기) | `c839b3d` |
| B-3 | **EKF s-보정**: fix 방위각 → s 의사측정 (innovation-form unwrap, lap-safe). 합성: 8 m 발산 → 0.31 m 유계 | `73e4f52` |
| B-4 | 122531 bag(마커 가시)으로 ArUco 실측: fix 1.3 Hz, 혁신 6.5 cm, 점프 없음. baseline/sfix 20셀: **red-flag 해제** (ssi 761=1.09×GT, nominal 999=1.03×GT) | `08f50ee`, `7752a27` |
| C-① | **estimator 브리지**: `TopicSampleAssembler`(순수, 테스트 6) + `pkrc_wallscan_bridge` ROS2 패키지 + bag 리플레이 검증 도구 (122531 PLUMBING OK). `/ukfm/odom_validated`만 절대 fix로 소비, θ 3-튜플 전달 | `a4e5fc2` |
| B-5 | 마커 가시성 실측 한계 **7 m** 반영 (vis7): +2~3% 비용, ŝ 최악 8.6 cm, 충돌 0 — **E5 go** | `a052da0` |

판정 총괄표 (5-시드 평균 objective, ↓):

| 조건 | ssi | nominal |
|---|--:|--:|
| GT (E1) | 697 | 969 |
| placeholder-EKF | 4,989 | 5,284 |
| measured (실측 노이즈+주기) | 5,105 | 5,039 |
| measured_aruco (1.3 Hz fix, s-보정 없음) | 5,699 | 5,206 |
| measured_aruco_sfix (s-보정) | 761 | 999 |
| **measured_aruco_sfix_vis7 (+7 m 한계)** | **786 (1.13×GT)** | **1,020 (1.05×GT)** |

## 2. 진행 중 — Phase C-② 컨트롤러 노드

- [x] 순수 코어 `WallScanControlLoop` 추출 (`control/scan_loop.py`): 상태머신 + 레퍼런스
      preview + 컨트롤러 step을 한 객체로 — **시뮬(`_sim_loop`)과 하드웨어 노드가 같은
      코드를 호출**하도록. §8 폐루프 네이티브 테스트 3개 (조립기→EKF→루프→컨트롤러
      한 틱이 시뮬레이터 없이 돈다).
- [x] `_sim_loop.run_mpc_cell`을 코어 호출로 교체 (동작 보존 리팩토링)
- [ ] **회귀 스모크 진행 중**: sfix s4 셀 재실행 → 기록치 96.7 재현 확인
- [ ] `wallscan_controller` ROS 노드: `/wallscan/state`(+estimator_debug의 s_hat) 구독
      → 코어 step → `/wallscan/u` 발행. 안전장치: 상태 stale 시 zero-thrust,
      `/wallscan/enable` 게이트 (teleop이 기본 권한 유지)

## 3. 남은 계획

### Phase C-③ — 구동계 연결 (로봇/실험실 필요)
- [ ] **u[-1,1] → VESC 전류 매핑**: `thruster_test.py`로 스러스터별 추력-전류 곡선 측정.
      Phase C의 마지막 미지수. teleop 노드에 auto 모드 추가 (키보드 = 안전 오버라이드).
- [ ] 시뮬 allocation matrix ↔ 실배치(surge×2/sway×2/heave×2, VESC 0x151–0x156) 대조
      (`_probe_plant.py` 파라미터 포함)

### Phase D — Jetson 실행 환경 + E4(c)
- [ ] acados aarch64 빌드 절차 문서화 + 실행 (Jetson)
- [ ] `bench_inference.py` 신설 (isaaclab 무의존): acados solve + SSI RFF 업데이트 실측
      — 데스크톱 먼저, Jetson에서 재실행 → E4(c) 표. 20 ms 제어 예산 판정.
- [ ] Jetson에 marinelab 체크아웃 + 브리지/컨트롤러 노드 colcon build
      (절차: [`marinelab/ros/pkrc_wallscan_bridge/README.md`](../../../marinelab/ros/pkrc_wallscan_bridge/README.md))

### Phase E — 수조 실험 (E5 본실험)
- [ ] 탱크-마커 캘리브레이션 측량 (`marker_x/y/yaw` 파라미터)
- [ ] 마커 가시성 스캔 심도 전 구간 확인 (실측 한계 7 m 전제의 현장 검증)
- [ ] 단계적 폐루프: depth-hold → wall-align → wallscan
- [ ] 본실험: 공칭 zero-shot × 3–5회 (Fixed-W → SSI → PPO onnx), bag 계측 →
      `eval_metrics`로 시뮬 동일 지표 → Table 4
- 주장 한정: "시뮬 순위가 실기체에서 유지된다" (experiments_plan §E5)

### 실기체 이식의 두 가지 하드 전제 (해소 상태)
1. ✅ ROS 체인이 fix의 절대 방위각 θ를 estimator에 전달 — `SensorSample.ukfm` 3-튜플,
   브리지 구현·검증 완료. **빠뜨리면 s-드리프트 재발** (실측: 나쁜 시드 67–112 cm).
2. ✅ 마커 가시성 7 m — vis7 시뮬 재검증 통과. 현장 확인만 남음.

## 4. 실행 규칙 (이 축 작업 시)

- 셀 실행은 per-cell 프로세스 + 워커별 사설 `isaaclab/logs` 마운트 (acados codegen 충돌).
- 컨테이너 산출물은 root 소유 → 커밋 전 chown (기록된 함정).
- 코드 수정은 **셀 실행 중 금지** (리포가 bind-mount — 셀 간 코드 불일치 발생).
  실행 중 개발은 git worktree에서.
- 기록된 조건의 의미를 바꾸는 수정은 새 조건명으로 (예: `measured_aruco` vs `_sfix`
  vs `_sfix_vis7`) — 결과 덮어쓰기 금지, config만으로 재현 가능하게.
