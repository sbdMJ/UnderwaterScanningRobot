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

- [ ] **(로봇 측 요청)** ArUco 마커가 실제로 보이는 상태의 bag 1개
      (`/ukfm/odom_validated`가 발행되는 구간 포함, 가능하면 스캔 심도까지 잠항 포함,
      2–3 분) — UKFM 절대 정확도·보정 점프·가시성 비율 특성화용. Phase B의
      상태소스 결정(wall_frame_ekf vs UKFM 직접)이 이것에 걸려 있다.
- [ ] SimSensorStream rate-and-hold 모델 + 실측 노이즈 SensorCfg 추가 → e5_ekf 재실행
- [ ] bag 리플레이로 wall_frame_ekf 오프라인 검증 (nmpc-wallscan 브랜치의 EKF replay
      진입점 이식)
