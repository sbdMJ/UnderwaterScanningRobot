# 실기체 센서 특성화 — 2026-08-06 teleop rosbag 분석 (Phase A)

> 소스: `~/PKRC로봇_코드_및_데이터/20260806_122731` (ROS2 bag, 34.4 s, 키보드 teleop,
> 수심 0.59–0.71 m). 분석 스크립트: `marinelab/scripts/experiments/hw_bag_characterize.py`
> (rosbags 패키지, ROS 설치 불필요). 실행 환경: 로봇은 Jetson + ROS2 Humble,
> hero_ws (`pkrc_control` teleop + `pkrc_controller` UKF-M localization).

## 1. 실측 결과 (34 s 구간)

hf-noise = std(diff)/√2 — 저주파 운동을 제거한 1차 차분 노이즈 추정 (이동 중이므로 상한값).

| 토픽 | 주기 | 최대 공백 | hf-noise | 시뮬 placeholder 대비 |
|---|--:|--:|---|---|
| `/ukfm/odom` xy | 31.5 Hz | 57 ms | 0.5 mm | ukfm_noise 0.03 m (60×) — 단 아래 §3 참고 |
| `/ukfm/odom` yaw | 31.5 Hz | 57 ms | 0.001 rad | — |
| `/ukfm/wall_distance` | 9.5 Hz | 416 ms | 3.4 cm | sonar_noise 0.05 m (1.5×) |
| `/dvl/data` vel | 9.5 Hz | 376 ms | 4.1 / 3.9 / 0.9 mm/s | dvl_noise 0.02 m/s (~5×) |
| `/bar10xt/depth` | 5.0 Hz | 602 ms | 4.2 mm | depth_noise 0.02 m (~5×) |
| `/imu/data` gyro | 100 Hz | 30 ms | 1.6–2.1 mrad/s | ins_noise 0.01 (~5×) |
| `/ekf/odometry_earth` (3DM EKF) | 25 Hz | 54 ms | — | — |

- UKFM z는 이 bag에서 상수 0 (±1 cm) — **깊이는 UKFM이 아니라 압력계가 소스**.
  `VehicleState` 조립 시 z는 `/bar10xt/depth`에서 가져와야 한다.
- 벽거리 0.68–1.27 m 범위에서 운용 (시뮬 d_ref는 1.5 m).

## 2. 시뮬 가정과의 핵심 괴리 — 노이즈보다 **주기**

시뮬 `SimSensorStream`은 모든 센서를 매 제어스텝(50 Hz)마다 신선하게 샘플한다.
실기체는 DVL 9.5 Hz, 깊이 5 Hz(공백 최대 0.6 s), 벽거리 9.5 Hz다. 노이즈 크기는
placeholder가 일관되게 ~5× 비관적이었지만, 갱신 주기는 반대로 5–10× 낙관적이다.
EKF-in-loop 재검증(Phase B) 전에 SimSensorStream에 **rate-and-hold 모델**(센서별
실측 주기로만 갱신)을 넣어야 시뮬 판정이 실기체를 대표한다.

## 3. 가장 중요한 발견 — 이 bag의 UKFM은 추측항법이었다

`ukfm_localization.py`의 발행 구조:

- `/ukfm/odom` — **무조건 발행** (ArUco 보정 없이 IMU+DVL 예측만으로도)
- `/ukfm/odom_validated` — ArUco fix가 `marker_timeout`(5 s) 이내일 때만 발행

이 bag에서 `/aruco/pose_array`·`/aruco/pose_6dof`·`/ukfm/odom_validated` 모두
**0건** → 34 s 내내 절대위치 보정이 없었다. 즉 §1의 매끈한 xy(0.5 mm)는 필터
예측의 스무스니스이지 절대 정확도가 아니다. 절대 정확도·보정 빈도·보정 시
점프 크기는 **이 bag으로는 특성화 불가**.

시사점:

1. e5_ekf 프리체크의 red-flag(ŝ 드리프트, 나쁜 시드 67–112 cm RMSE)는 실기체에서도
   그대로 유효한 위협이다 — 절대 보정이 없으면 ŝ는 실기체에서도 DVL 적분 드리프트를
   따라간다. 시뮬의 보수적 UKFM 게이팅이 오히려 현실적이었다.
2. 시뮬 `ukfm_valid` 게이트(깊이·기울기)와 실기체 `odom_validated` 게이트(마커
   타임아웃)는 설계가 일치한다 — 이식 시 `ukfm_valid := odom_validated 수신 여부`로
   매핑하면 된다.

## 4. 다음 액션

- [x] **(로봇 측 요청)** ArUco 마커가 보이는 bag → §5 (20260806_122531)
- [x] SimSensorStream rate-and-hold 모델 + 실측 노이즈 SensorCfg 추가 → e5_ekf 재실행
      (`SensorCfgHW2026Bag`/`SensorCfgHW2026BagAruco`, 커밋 bd47e68)
- [ ] bag 리플레이로 wall_frame_ekf 오프라인 검증 (nmpc-wallscan 브랜치의 EKF replay
      진입점 이식)

## 5. 마커 가시 bag (20260806_122531) — UKFM 절대 보정 특성화

48.5 s, 수심 0.59–3.81 m. `/aruco/pose_6dof` 37건, `/ukfm/odom_validated` 893건
(t=16.5 s부터 연속 발행, odom 틱 커버리지 64%). 분석:
`marinelab/scripts/experiments/hw_bag_aruco_analyze.py`.

| 항목 | 실측값 |
|---|---|
| ArUco fix 주기 | **~1.3 Hz** (간격 중앙값 0.51 s, p90 1.24 s, 최대 2.48 s) |
| fix 시점 혁신 \|odom−aruco\| | **중앙값 6.5 cm**, p90 10.8 cm, 최대 21.4 cm |
| 보정 점프 (odom 스텝) | 0.02 cm — 노드의 low-pass가 흡수, 점프 모델 불필요 |
| 마커 가시 심도 | 3.81 m에서도 유지 (스캔 심도 전 구간은 실험 수조에서 확인 필요) |

시사점: **절대 정보는 odom의 31.5 Hz가 아니라 ArUco의 1.3 Hz로 들어온다.** 시뮬의
UKFM 채널은 이 케이던스와 fix 정확도로 모델링해야 한다 → `SensorCfgHW2026BagAruco`
(`ukfm_period` 0.77 s, `ukfm_noise` 0.065).

## 6. e5_ekf 재판정 결과와 근본 원인 (2026-08-09)

| 조건 (ssi/nominal 5-시드 평균) | ssi | nominal |
|---|--:|--:|
| GT (E1) | 697 | 969 |
| placeholder-EKF | 4,989 | 5,284 |
| measured (실측 노이즈+주기, UKFM 31.5 Hz) | 5,105 | 5,039 |
| measured_aruco (UKFM 1.3 Hz, s-보정 없음) | 5,699 | 5,206 |
| **measured_aruco_sfix (s-보정)** | **761 (1.09×GT)** | **999 (1.03×GT)** |
| measured_aruco_sfix_vis7 (+가시성 7 m 한계) | 786 (1.13×GT) | 1,020 (1.05×GT) |

실측 센서 모델은 φ RMSE를 1.4–2.4°→0.4–0.6°로 잡았지만 objective는 그대로 —
**ŝ가 유일한 미보정 적분기**이기 때문이다 (`wall_frame_ekf.update_ukfm`의 H에 s행이
없어, 절대 fix가 알고 있는 방위각 θ를 버리고 있었다). 나쁜 시드의 DVL 바이어스 추첨이
ŝ에 무한 적분되어 67–112 cm RMSE → 스캔 레퍼런스 자체가 틀어져 어느 컨트롤러도 못
버틴다 (공통 모드 5–7×).

수정: fix 방위각에서 s 의사측정을 파생해 EKF에 추가 (`update_ukfm(s_meas)`,
`estimator.py`의 innovation-form unwrap). 합성 검증: 5 cm/s DVL 바이어스에서 s 오차
8 m 발산 → **0.31 m 유계** (60–360 s 평평).

**sim 재판정 (`measured_aruco_sfix`, 2026-08-09): red-flag 해제.** ŝ RMSE 전 시드
2.6–6.0 cm (수정 전 67–112 cm), 충돌 0, 순위 보존 (ssi 761 < nominal 999 — GT의
697 < 969와 동일 방향). EKF-in-loop 비용이 GT 대비 +3~9%로, 실측 UKFM 케이던스
(1.3 Hz) 하에서 하드웨어 진입 게이트를 통과한다. 개별 시드 최대치는 ssi s0의 6.0×
(18→109)이나 절대값이 작아 시드 분산 범위 안이다.

남는 하드웨어 전제였던 두 가지도 해소됨 (2026-08-09):

① **θ 전달 체인 구현 완료** — `control/hw_bridge.py`(순수, 네이티브 테스트) +
`ros/pkrc_wallscan_bridge`(hero_ws용 rclpy 노드). `/ukfm/odom_validated`만 절대
fix로 소비하고, fix를 (r, φ, θ) 3-튜플로 estimator에 넘긴다. 배선은
`hw_bag_replay_estimator.py`로 122531 bag에 대해 검증 (PLUMBING OK).

② **마커 가시성 실측 한계 7 m 반영 재검증** (`measured_aruco_sfix_vis7`,
`ukfm_max_depth: 7.0` — z < 3 m에서 fix 상실, 스캔 하단 2 m 블라인드): ssi
761→786 (+3%), nominal 999→1,020 (+2%), ŝ RMSE 최악 8.6 cm, 충돌 0. 블라인드
밴드에서의 코스팅은 재획득 시 θ 보정이 되잡는 유계 드리프트로 확인. **E5 go.**
