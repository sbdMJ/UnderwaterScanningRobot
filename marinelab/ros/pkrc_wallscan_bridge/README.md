# pkrc_wallscan_bridge — 센서 토픽 → 벽면좌표계 상태추정 브리지 (E5)

hero_ws의 드라이버 토픽을 `marinelab.control.WallFrameStateEstimator`(시뮬과 바이트 동일
코드)에 연결해 `/wallscan/state`(tank-frame Odometry)를 발행한다. 컨트롤러(NMPC/SSI)는
이 상태를 소비한다 — E5 아키텍처에서 "바뀌는 것은 estimator 입력원뿐"의 그 입력원.

## 핵심 설계 (e5_ekf 캠페인에서 실측으로 확정된 것들)

1. **`/ukfm/odom_validated`만 절대 fix로 취급** — 일반 `/ukfm/odom`은 ArUco가 안 보여도
   추측항법으로 계속 발행된다 (2026-08-06 bag에서 34초간 fix 0건 실측). validated는
   marker_timeout(5 s) 이내의 fix가 있을 때만 나온다.
2. **fix의 절대 방위각 θ를 estimator에 전달한다** (`SensorSample.ukfm` 3-튜플).
   이것이 s(스캔 진행도) 상태를 보정하는 유일한 절대 정보다 — 빠뜨리면 DVL 바이어스가
   s에 무한 적분되는 드리프트가 그대로 재발한다 (시뮬 실측: 나쁜 시드 s RMSE 67–112 cm,
   objective 5–7× 열화 → θ 보정 후 2.6–6 cm, GT 대비 +3~9%).
3. **측정은 한 번만 소비** — wall range/fix는 새 메시지가 있는 틱에만 필터를 보정한다.
   50 Hz 루프가 9.5 Hz 소나를 5번씩 재적용하는 과신을 막는다 (DVL/깊이는 hold).
4. 순수 로직(`TopicSampleAssembler`)은 marinelab 네이티브 테스트로 검증되며, 이 노드는
   구독/발행 배선만 가진다. 배선 자체는 `hw_bag_replay_estimator.py`로 실제 bag에 대해
   검증했다 (PLUMBING OK, 2026-08-09).

## Jetson 설치

```bash
# 1) 로봇에 marinelab 리포 체크아웃 (isaaclab/Isaac Sim 불필요 — 순수 모듈만 씀)
git clone <repo> ~/UnderwaterScanningRobot

# 2) 이 패키지를 hero_ws에 링크하고 빌드
ln -s ~/UnderwaterScanningRobot/marinelab/ros/pkrc_wallscan_bridge ~/hero_ws/src/
cd ~/hero_ws && colcon build --packages-select pkrc_wallscan_bridge

# 3) 실행 (marinelab 위치는 자동 탐색되나, 다른 경로면 MARINELAB_ROOT 지정)
export MARINELAB_ROOT=~/UnderwaterScanningRobot/marinelab
ros2 run pkrc_wallscan_bridge estimator_bridge --ros-args \
  -p tank_height:=10.0 -p tank_radius:=6.0 \
  -p marker_x:=0.0 -p marker_y:=0.0 -p marker_yaw:=0.0
```

`marinelab/__init__`가 isaaclab을 당기는 문제는 `marinelab_loader.py`가 conftest와 같은
방식(경로만 가진 빈 패키지 등록)으로 우회한다. numpy만 있으면 된다.

## 반드시 해야 하는 캘리브레이션

- **marker_x/y/yaw**: UKFM world(마커 앵커) → 탱크축 원점 좌표계의 2-D 강체 변환.
  마커를 탱크축 위 수면에 두면 identity. 아니면 측량해서 넣는다.
- **tank_height**: 압력 깊이 → 탱크 z(바닥 기준) 변환에 쓰인다.
- 마커 가시성: 실측 한계 **7 m** (2026-08-09 확인). 탱크 10 m 기준 z < 3 m 구간은
  fix가 끊긴다 — 그 구간에서 s는 DVL 추측항법으로 코스팅하며, 시뮬 재검증
  (`e5_ekf_precheck.yaml`의 vis7 조건)이 이 블라인드 구간 포함 성능을 판정한다.

## 컨트롤러 노드 (`wallscan_controller`)

`/wallscan/state`(+`estimator_debug`의 ŝ)를 구독해 순수 코어 `WallScanControlLoop`
(시뮬 러너와 **동일 객체** — 회귀 셀로 +0.015% 재현 확인)를 돌리고 `/wallscan/u`
(정규화 [-1,1] 6채널)를 발행한다.

```bash
# plant 파라미터는 sim이 export한 JSON (repo에 커밋됨)
ros2 run pkrc_wallscan_bridge wallscan_controller --ros-args \
  -p method:=nominal \
  -p plant_json:=$MARINELAB_ROOT/config/pkrc_plant_fixed_tam.json
# SSI: -p method:=ssi -p params_json:=<BO 가중치 JSON>  (채택 하이퍼파라미터가 기본값)
```

안전 설계:
- **`/wallscan/enable` (Bool)이 기본 OFF** — 켜기 전까지 u는 항상 0. 켜는 순간 현재
  심도에 스캔을 앵커링. teleop 노드가 CAN을 소유하므로 최종 권한은 항상 키보드.
- 상태 스트림이 0.5 s 이상 끊기면 zero-thrust + 경고. 컨트롤러 예외도 zero-thrust.
- NMPC(acados)는 Jetson에 acados/casadi 빌드가 선행돼야 한다 (Phase D).

## 남은 통합 단계

- `u` → VESC 전류 매핑 (thruster_test.py 곡선), teleop auto 모드에서 `/wallscan/u` 소비
