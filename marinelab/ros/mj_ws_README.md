# mj_ws — Jetson 오버레이 워크스페이스 (wallscan sim-to-real)

`make_mj_ws.sh`가 조립한 워크스페이스. 구성:

- `src/pkrc_control` — hero_ws 복사본 + **패치 0001 적용 완료** (teleop wallscan
  auto 모드). hero_ws 원본은 건드리지 않는다.
- `src/pkrc_wallscan_bridge` — estimator/controller/mapper 노드 (이번 캘리브레이션
  에는 불필요 — 실행하려면 marinelab 체크아웃 + `MARINELAB_ROOT` 필요, Phase D).

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
