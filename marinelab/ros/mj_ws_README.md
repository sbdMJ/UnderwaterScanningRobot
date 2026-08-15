# mj_ws — Jetson 오버레이 워크스페이스 (wallscan sim-to-real)

`make_mj_ws.sh`가 조립한 워크스페이스. 구성:

- `src/pkrc_control` — hero_ws 복사본 + **패치 0001 적용 완료** (teleop wallscan
  auto 모드). hero_ws 원본은 건드리지 않는다.
- `src/pkrc_wallscan_bridge` — estimator/controller/mapper 노드.
- `marinelab/` — **순수 파이썬 marinelab 번들** (리포 클론 불필요):
  패키지 + `config/pkrc_plant_hw2026.json`(실측 확정 plant) +
  `scripts/experiments/{bench_inference,hw_bag_replay_estimator}.py`.
  ROS 노드용 `MARINELAB_ROOT=~/mj_ws/marinelab`.
- `experimental_results/tuning/bo_nmpc/best_params.json` — BO 튜닝 가중치
  (ssi/bo 컨트롤러와 벤치가 자동 로드).
- `jetson_acados_build.md` — acados aarch64 빌드 절차 (시나리오 ①의 전제).

캘리브레이션(§5–6)만 할 거면 ROS 빌드(§2)까지로 충분하고, **시나리오 ①~③**
(드라이런 벤치 / 체인 라이브니스 / depth-hold)은 §7–9를 따른다.

## 1. Jetson으로 복사

PC에서:

```bash
rsync -a ~/mj_ws/ <user>@<jetson-ip>:~/mj_ws/
```

## 2. 빌드 (Jetson에서, 1회)

```bash
source /opt/ros/humble/setup.bash
cd ~/mj_ws
colcon build --symlink-install
```

`--symlink-install`이면 이후 파이썬 파일 수정이 재빌드 없이 반영된다.
(pkrc_wallscan_bridge까지 같이 빌드되지만 무해 — 실행만 Phase D에서.)

## 3. teleop 실행 — 이 터미널은 hero_ws 대신 mj_ws를 source

```bash
source /opt/ros/humble/setup.bash
source ~/mj_ws/install/setup.bash        # mj_ws의 pkrc_control이 우선
ros2 run pkrc_control keyboard_control_teleop
```

**주의 — CAN 소유자는 teleop 하나뿐**: hero_ws 쪽 teleop이 이미 떠 있으면 먼저
종료할 것 (둘이 동시에 CAN에 쓰면 안 된다). 센서 노드 등 나머지 hero_ws
스택은 기존 터미널에서 그대로 운용 — 이 터미널만 mj_ws를 source한다.
(같은 터미널에서 hero_ws도 source해야 하면 순서는 hero_ws → mj_ws:
나중에 source한 쪽의 `pkrc_control`이 우선한다.)

## 4. auto 모드 사용법

- **`g` 키 = auto 토글.** 진입 시 로그 `WALLSCAN AUTO ON — 아무 키나 누르면
  수동 복귀` 확인.
- **그 외 아무 키 = 즉시 수동 복귀 (비상정지).** 로그 `WALLSCAN AUTO OFF (key …)`.
- `/wallscan/current_cmd`가 **0.5 s 이상 끊기면 자동으로 수동 복귀 + 정지**
  (발행 측 Ctrl+C가 곧 소프트 정지).
- auto 전류도 수동과 같은 안전 경로(극성·max_current 클램프·데드존·램프)를
  통과한다.

## 5. 전류 명령 발행 (다른 터미널 — Jetson이든 같은 ROS 도메인의 PC든)

```bash
source /opt/ros/humble/setup.bash
ros2 topic pub -r 10 /wallscan/current_cmd std_msgs/msg/Float64MultiArray \
  "{data: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]}"    # 예: surge 쌍 각 +1.0 A
```

- **10 Hz 연속 발행 필수** (`-r 10`) — 0.5 s staleness에 걸리면 자동 정지된다.
- 순서는 VESC T1..T6, 단위 A. **+1 A 명령당 추력 방향**
  (thruster_mapping.md §4b, heave는 수중 실측으로 확정):

  | T1 | T2 | T3 | T4 | T5 | T6 |
  |---|---|---|---|---|---|
  | 전진 | 전진 | 우현 | 우현 | **상승** | **하강** |

  surge/sway 쌍은 같은 부호 = 같은 방향. **heave 쌍은 반대 부호가 같은 방향**
  — 하강 `[0,0,0,0,-I,+I]`, 상승 `[0,0,0,0,+I,-I]`.
- 첫 스텝은 0.5 A로 방향·리깅부터 확인하고 올릴 것.

## 6. 캘리브레이션 절차

`<repo>/docs/experiments/sim-to-real/thruster_mapping.md` §4b(쌍 bollard pull +
요-영점) / §4c(heave 부력 평형). 기록 CSV(`thruster,amps,kgf`; 쌍은 `12`/`34`/`56`)
를 PC의 `marinelab/scripts/experiments/hw_thrust_calibrate.py`에 넣으면
`newton_per_amp`/`max_thrust`까지 산출된다.

**2026-08-15 확정치** (아래 시나리오들이 쓰는 값):
`newton_per_amp=[1.594,1.594,1.754,1.754,0.99,0.99]`,
`amps_offset=[0.694,0.694,0.764,0.764,0.729,0.729]`, `max_thrust=3.68`,
plant = `marinelab/config/pkrc_plant_hw2026.json` (납 0.5 kg 트림 상태 기준).

---

# 시나리오 ①~③ (Phase D/E-1단계)

## 7. 시나리오 ① — 드라이런 (물 불필요)

전제: `jetson_acados_build.md`대로 acados 빌드 + 그 §0의 파이썬 의존
(`torch` cpu, `gymnasium`, 리플레이용 `rosbags`).

```bash
# (a) E4(c) 추론 벤치 — 20 ms 예산 판정, 결과 json은 PC로 회수해 커밋
cd ~/mj_ws && python3 marinelab/scripts/experiments/bench_inference.py \
  --steps 1000 --label jetson
# 첫 실행은 OCP 코드젠 수십 초. 산출: experimental_results/e4_inference/bench_jetson.json
# 판정: total p99 < 20 ms. 초과 시 --rti-iters 4 → --horizon 20 순으로 완화.

# (b) estimator 리플레이 — 배선 검증 (마커 보이는 bag 하나를 Jetson에 복사)
python3 marinelab/scripts/experiments/hw_bag_replay_estimator.py <bag-dir>
```

## 8. 시나리오 ② — 체인 라이브니스 (물, enable OFF = 추력 0, 리스크 없음)

hero_ws 센서 스택(+ 소형 아크릴 수조면 `ping1d_sonar` 드라이버도) 실행 후,
mj_ws 터미널 3개:

```bash
# T1: estimator (소형 수조 = 아크릴: DVL·/ukfm/wall_distance 사망 → Ping1D로 대체)
source /opt/ros/humble/setup.bash && source ~/mj_ws/install/setup.bash
export MARINELAB_ROOT=~/mj_ws/marinelab
ros2 run pkrc_wallscan_bridge estimator_bridge --ros-args \
  -p tank_height:=0.85 -p tank_radius:=6.0 \
  -p marker_x:=0.0 -p marker_y:=0.0 -p marker_yaw:=0.0 \
  -p wall_topic:=/sensor/sonar/ping1d/range -p wall_msg:=range
# 본 탱크에서는 wall_topic/wall_msg 생략(기본 = DVL 전방 고도) + tank_height:=10.0

# T2: 컨트롤러 (enable 기본 OFF — u는 계산되지만 0으로 게이트)
ros2 run pkrc_wallscan_bridge wallscan_controller --ros-args \
  -p method:=ssi -p plant_json:=$MARINELAB_ROOT/config/pkrc_plant_hw2026.json \
  -p params_json:=$HOME/mj_ws/experimental_results/tuning/bo_nmpc/best_params.json

# T3: 매퍼 (u→전류; enable OFF 동안은 0 A 명령이 나감)
ros2 run pkrc_wallscan_bridge thrust_mapper --ros-args \
  -p newton_per_amp:="[1.594,1.594,1.754,1.754,0.99,0.99]" \
  -p amps_offset:="[0.694,0.694,0.764,0.764,0.729,0.729]" -p max_thrust:=3.68
```

teleop(§3)을 auto로 두고 확인할 것: `/wallscan/state` 50 Hz /
`estimator_debug` / controller 로그의 solve_ms / `/wallscan/current_cmd`가
0으로 흐르는지 / 마커 인식 시 first-fix 앵커 로그 / **DVL 부재 시 EKF가
발산하는지 유계인지** (이게 이 시나리오의 수확). 전 과정 bag 녹화 권장:
`ros2 bag record /wallscan/state /wallscan/estimator_debug /wallscan/u
/wallscan/current_cmd /teleop/thruster_currents /bar10xt/depth /imu/data`.

## 9. 시나리오 ③ — depth-hold 폐루프 (물, 소형 수조 가능)

②의 T2만 파라미터를 바꿔 스테이션 키핑으로: 스캔 컬럼을 현재 심도 한 점으로
붕괴시키고(sway 0) 컨트롤러가 깊이+자세만 잡게 한다.

```bash
ros2 run pkrc_wallscan_bridge wallscan_controller --ros-args \
  -p method:=ssi -p plant_json:=$MARINELAB_ROOT/config/pkrc_plant_hw2026.json \
  -p params_json:=$HOME/mj_ws/experimental_results/tuning/bo_nmpc/best_params.json \
  -p z_top:=0.45 -p z_bottom:=0.45 -p sway_step:=0.0
```

절차: 로봇을 중앙 ~0.45 m 심도에 두고 → `ros2 topic pub -1 /wallscan/enable
std_msgs/msg/Bool "{data: true}"` (상승엣지에 현재 심도로 재앵커) → 관찰 →
`data: false` 또는 teleop 아무 키로 즉시 회수.

**안전 수칙 (소형 수조 필수)**:
- 컨트롤러는 벽거리 1.5 m도 잡으려 한다 — 마커 프레임을 현재 위치가 가상
  탱크의 r≈4.5 m가 되게 놓지 않으면 수평으로 기어간다. 마커를 못 놓으면
  **teleop max_current를 낮춰**(1–1.5 A) 수평 권한 자체를 줄이고 테더 대기.
- enable은 짧게 (첫 시도 ≤30 s), 심도 0.2 m 이탈·과도 수평 이동 시 즉시 OFF.
- z_top/z_bottom을 수조 심도(0.85 m)보다 깊게 두지 말 것 — 기본값 그대로
  켜면 z_bottom=1.0을 찾아 바닥으로 파고든다.
