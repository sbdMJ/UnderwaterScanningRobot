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

⚠ **rsync로 코드를 갱신했으면 떠 있는 브리지 노드(T1~T3)를 전부 재시작**할 것
— symlink는 파일만 바꾸고 실행 중인 프로세스는 옛 코드로 계속 돈다 (2026-08-18:
mapper 재시작 누락으로 무캘리브레이션 리밋사이클이 한 세션 더 재발).

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
# 브리지 터미널 공통 (T1~T3): hero_ws를 먼저 source (dvl_msgs 런타임 import),
# 그 위에 mj_ws. acados 환경변수는 T2에 필수 (~/.bashrc에 넣었다면 생략).
source /opt/ros/humble/setup.bash
source ~/hero_ws/install/setup.bash && source ~/mj_ws/install/setup.bash
export MARINELAB_ROOT=~/mj_ws/marinelab
export ACADOS_SOURCE_DIR=$HOME/acados LD_LIBRARY_PATH=$HOME/acados/lib:$LD_LIBRARY_PATH

# T1: estimator (소형 수조 = 아크릴: DVL·/ukfm/wall_distance 사망 → Ping1D로 대체)
# 발행 전제 2가지 — 로그가 무엇을 기다리는지 말해준다:
#   (a) DVL·IMU·깊이 첫 메시지 (아크릴이라도 DVL "드라이버"는 떠 있어야 함 —
#       락 없이도 메시지는 나온다), (b) 첫 마커 fix 앵커. 마커 없이 라이브니스만
#       볼 때는 anchor_without_fix:=true (블라인드 앵커 — x/y/s는 fix 전까지 허구,
#       폐루프(③) 금지).
ros2 run pkrc_wallscan_bridge estimator_bridge --ros-args \
  -p imu_mount_rpy_deg:="[180.0,0.0,0.0]" -p vz_from_depth:=true \
  -p tank_height:=0.85 -p tank_radius:=6.0 \
  -p marker_x:=0.0 -p marker_y:=0.0 -p marker_yaw:=0.0 \
  -p wall_topic:=/sensor/sonar/ping1d/range -p wall_msg:=range \
  -p anchor_without_fix:=true
# 본 탱크에서는 wall_topic/wall_msg/anchor_without_fix 생략 + tank_height:=10.0

# T2: 컨트롤러 (enable 기본 OFF — u는 계산되지만 0으로 게이트; 기본 h20/rti4 =
#     E4c 배포 설정. 첫 실행은 acados 코드젠 수십 초 — ~/.cache/wallscan_acados)
ros2 run pkrc_wallscan_bridge wallscan_controller --ros-args \
  -p method:=ssi -p plant_json:=$MARINELAB_ROOT/config/pkrc_plant_hw2026.json \
  -p params_json:=$HOME/mj_ws/experimental_results/tuning/bo_nmpc/best_params.json

# T3: 매퍼 (u→전류; enable OFF 동안은 0 A 명령이 나감)
ros2 run pkrc_wallscan_bridge thrust_mapper --ros-args -p amps_limit:="[3.0,3.0,3.0,3.0,5.0,5.0]"   # 시나리오 ②: 기본 클램프
```

teleop(§3)을 auto로 두고 확인할 것: `/wallscan/state` 50 Hz /
`estimator_debug` / controller 로그의 solve_ms / `/wallscan/current_cmd`가
0으로 흐르는지 / 마커 인식 시 first-fix 앵커 로그 / **DVL 부재 시 EKF가
발산하는지 유계인지** (이게 이 시나리오의 수확). 전 과정 bag 녹화 권장:
`ros2 bag record /wallscan/state /wallscan/estimator_debug /wallscan/u
/wallscan/current_cmd /teleop/thruster_currents /bar10xt/depth /imu/data`.

## 9. 시나리오 ③ — depth-hold 폐루프 (물, 소형 수조 가능)

②에서 T1·T2만 바꾼다. 스캔 컬럼을 현재 심도 한 점으로 붕괴시키고(sway 0)
컨트롤러가 깊이+자세(+마커가 있으면 위치)만 잡게 한다.

```bash
# T1: estimator — ③에서는 wall_topic을 기본(사망 토픽)으로 되돌린다.
# 이유: 소형 수조의 실제 벽거리(≪1.5 m)를 소나로 먹이면 가상 원통(R=6,
# d_ref=1.5) 기하와 충돌해 혁신이 마커 fix와 싸운다. 소나 없이 마커+깊이로만.
ros2 run pkrc_wallscan_bridge estimator_bridge --ros-args \
  -p imu_mount_rpy_deg:="[180.0,0.0,0.0]" -p vz_from_depth:=true \
  -p tank_height:=0.85 -p tank_radius:=6.0 \
  -p marker_x:=4.5 -p marker_y:=0.0 -p marker_yaw:=0.0
# marker_x=4.5: 마커 바로 아래가 가상 탱크의 r=4.5(=R−d_ref) 지점이 되게 —
# 컨트롤러의 "벽거리 1.5 m" 목표가 현 위치에서 이미 충족되도록. 마커의 x축을
# 로봇 초기 기수 방향과 맞춰 붙이면 marker_yaw:=0으로 충분.

# T2: 컨트롤러 — depth-hold는 hold_z로 상태머신을 통째로 우회한다. Z_HOLD는
# 아래 절차에서 읽은 값. 첫 폐루프는 method:=nominal 권장 (ssi는 hold 확인 후).
# ★ hold_z 필수 — z_top=z_bottom 붕괴만으로는 안 된다 (2026-08-18 bag 03_41:
# 도달 조건이 항상 참이라 위상이 34 s에 49회 순환, SWAY 진입마다 z_ref를 현재
# 심도로 재래치 → ±4 cm 사각파 주입 → 15 cm 리밋사이클).
ros2 run pkrc_wallscan_bridge wallscan_controller --ros-args \
  -p method:=nominal -p plant_json:=$MARINELAB_ROOT/config/pkrc_plant_hw2026.json \
  -p params_json:=$HOME/mj_ws/experimental_results/tuning/bo_nmpc/best_params.json \
  -p hold_z:=Z_HOLD \
  -p z_top:=Z_HOLD -p z_bottom:=Z_HOLD -p sway_step:=0.0 -p reach_eps:=0.05

# T3: 매퍼 — 소형 수조에서는 amps_limit로 권한 자체를 줄인다 (안전 노브)
ros2 run pkrc_wallscan_bridge thrust_mapper --ros-args \
  -p amps_limit:="[1.5,1.5,1.5,1.5,3.0,3.0]"
```

절차:
1. 로봇을 수조 중앙, 목표 심도(~수면 아래 0.4 m)에 손으로 잡고
   `ros2 topic echo --once /wallscan/state --field pose.pose.position.z`
   → 그 값을 `Z_HOLD`로 T2 재시작 (enable 엣지에서 z_ref가 현재 심도에
   앵커된 뒤 0.2 m/s로 hold_z까지 램프하고 거기 고정된다).
2. teleop auto ON (`g`), 손 놓고 →
   `ros2 topic pub -1 /wallscan/enable std_msgs/msg/Bool "{data: true}"`.
3. 관찰 (첫 시도 ≤30 s): 심도 유지 ±10 cm, 전류 `/teleop/thruster_currents`가
   heave 쌍 위주로 0.7–1.5 A 근방, 수평 전류가 지속 편향이면 마커 배치 문제.
4. 종료: `"{data: false}"` 발행 또는 **teleop 아무 키 (최종 비상정지)**.

### ③-마커리스 변형 — depth-hold 전용 (마커 리깅 불가 시)

x/y/s가 허구(②에서 실측: 정지 중 1–2 cm/s 표류)라 컨트롤러가 수평으로
허깨비를 쫓는다 → **수평 4개의 amps_limit을 데드존 전류로 클램프해 수평
추력을 물리적으로 0으로** 만들고 (I ≤ I₀ ⇒ F=0), heave만 살린다. 깊이
루프(병목 축)는 온전히 시험된다. 바뀌는 것:

```bash
# T1: ②와 동일하게 anchor_without_fix:=true 유지 (wall/마커 없음)
ros2 run pkrc_wallscan_bridge estimator_bridge --ros-args \
  -p imu_mount_rpy_deg:="[180.0,0.0,0.0]" -p vz_from_depth:=true \
  -p tank_height:=0.85 -p tank_radius:=6.0 \
  -p marker_x:=4.5 -p marker_y:=0.0 -p marker_yaw:=0.0 \
  -p anchor_without_fix:=true
# T3: 수평 = 데드존 클램프 (추력 0), heave만 3 A
ros2 run pkrc_wallscan_bridge thrust_mapper --ros-args \
  -p amps_limit:="[0.69,0.69,0.76,0.76,3.0,3.0]"
```

T2는 아래 "터미널별 상세"의 명령을 그대로 쓴다 — **마커리스는
`depth_only:=true` + `reach_eps:=0.05`가 필수** (③ 본문 T2와 다름).
관찰 대상은 **z hold 품질만**: 유지 정밀도(±cm), heave 전류 패턴
(평형 ~0.85 A 근방 + 보정), 손으로 10 cm 눌렀다 놓을 때 복원. 수평/요는
전류가 데드존이라 아무 일도 안 일어나는 게 정상이고, current_cmd에 찍히는
수평 명령이 허구를 쫓아 커지는 모습 자체가 "마커가 왜 필요한가"의 기록이
된다. 마커 리깅이 되는 날 ③ 본문(수평 1.5 A)으로 승격.

#### 터미널별 상세 (마커리스 depth-hold 세션 전체, 순서대로)

모든 mj_ws 터미널(T1–T3)의 공통 서두 — 한 줄이라도 빠지면 각각 다른 방식으로
죽거나 침묵한다 (hero_ws 없으면 dvl_msgs ImportError, ACADOS 없으면 T2
ImportError):

```bash
source /opt/ros/humble/setup.bash
source ~/hero_ws/install/setup.bash
source ~/mj_ws/install/setup.bash
export MARINELAB_ROOT=~/mj_ws/marinelab
export ACADOS_SOURCE_DIR=$HOME/acados
export LD_LIBRARY_PATH=$HOME/acados/lib:$LD_LIBRARY_PATH
```

**T0 (hero_ws 센서, 평소 bringup 그대로)**: IMU + bar10xt + **DVL 드라이버
필수** (아크릴이라 락은 없어도 드라이버는 떠 있어야 estimator가 발행 시작).

```bash
ros2 topic hz /imu/data /bar10xt/depth /dvl/data   # dvl은 끊겼다 이어져도 OK
```

**T1 (estimator)** — 공통 서두 후:

```bash
ros2 run pkrc_wallscan_bridge estimator_bridge --ros-args \
  -p imu_mount_rpy_deg:="[180.0,0.0,0.0]" -p vz_from_depth:=true \
  -p tank_height:=0.85 -p tank_radius:=6.0 \
  -p marker_x:=4.5 -p marker_y:=0.0 -p marker_yaw:=0.0 \
  -p anchor_without_fix:=true
```

기대 로그: `waiting for first message on: ...` → `BLIND anchor at r=4.50`
(1회 WARN) → 이후 침묵 = 발행 중. 체크: `ros2 topic hz /wallscan/state` ≈ 50.

**T3 (mapper — T2보다 먼저: auto 유지용 하트비트)** — 공통 서두 후:

```bash
ros2 run pkrc_wallscan_bridge thrust_mapper --ros-args \
  -p amps_limit:="[0.69,0.69,0.76,0.76,3.0,3.0]"   # ★ 수평 = 데드존 = 추력 0
```

기대: `thrust mapper up [CALIBRATED], ... k=[1.594, ...], I0=[0.694, ...]` 즉시
+ `/wallscan/current_cmd` 5 Hz 0. **캘리브레이션 값은 노드 기본값** (2026-08-18:
리스트 파라미터가 셸 인용에서 깨져 두 세션이 UNCALIBRATED로 돌았던 사고 후
기본값으로 구움) — `[UNCALIBRATED]`가 뜨면 뭔가 잘못된 것.

**T2 (controller)** — 먼저 로봇을 **가용 컬럼의 중간쯤**에 손으로 잡고 T5에서
Z_HOLD를 읽는다. ⚠ **Z_HOLD는 매 세션 새로 읽는다 — 절대 지난 세션 값 재사용
금지.** state z 프레임은 estimator가 세션마다 재앵커하고 bar10xt 오프셋도
표류한다: 08-18 세션은 부유 0.25/바닥 0.016이었지만 08-19 세션은 부유
0.98/바닥 0.73이었고, 재사용한 0.13이 **바닥보다 0.6 m 아래**가 되어 캡 추력으로
바닥 고착 (bag 22_36_50, `hw_bag_depthhold_2236_20260819.py`). 절대값이 아니라
**방금 읽은 z에서 컬럼 중간으로 10 cm 안쪽**을 고른다 (바닥·수면 각각 ≥8 cm 여유).
이제 컨트롤러가 enable 시 |z−hold_z| > 0.3 m이면 `HOLD-Z SANITY` ERROR와 함께
enable을 거부한다 — 이 에러가 보이면 낡은 값을 쓴 것이니 다시 읽어라
(의도적 원거리 목표는 `-p hold_z_sanity_m:=`으로 완화, 0 = 비활성):

```bash
ros2 topic echo --once /wallscan/state --field pose.pose.position.z   # → Z_HOLD
```

공통 서두 후 (Z_HOLD 치환; **마커리스는 depth_only 필수 + 첫 시도는 nominal** —
2026-08-18 실측: depth_only 없이는 허구 수평 오차가 heave까지 지배해 바닥 고착):

```bash
ros2 run pkrc_wallscan_bridge wallscan_controller --ros-args \
  -p method:=nominal -p plant_json:=$MARINELAB_ROOT/config/pkrc_plant_hw2026.json \
  -p params_json:=$MARINELAB_ROOT/config/depthhold_rate_weights.json \
  -p thrust_limits:="[0.0,0.0,0.0,0.0,2.25,2.25]" \
  -p hold_z:=Z_HOLD -p depth_only:=true \
  -p z_top:=Z_HOLD -p z_bottom:=Z_HOLD -p sway_step:=0.0 -p reach_eps:=0.05
# hold_z: 상태머신 우회(고정 z_ref) — 2026-08-18 bag 03_41에서 위상 순환이
# z_ref를 0.13↔0.17로 튕겨 15 cm 리밋사이클을 만든 것의 근본 수정.
# 2026-08-19 (bag 04_15 → 근본원인 ⑧ 수정): plant JSON이 ACTUATOR-RATE 모델을
# 켠다 (force_rate_limit = teleop 램프 17 A/s × k; MPC가 전류 램프를 예측에 반영
# → 뱅뱅 relay 제거). params_json은 rate 모델 전용 depth-hold 가중치
# (z=8, v_z=15, wu=0.1 — 기본 z=40/wu=0.01은 vz LPF 0.5 s 지연과 공진해 7 cm
# 리밋사이클, _probe_rate_mpc.py 실측; BO best_params는 rate 모델에 이월 불가).
# thrust_limits = 이 세션의 실현 가능 힘 k(amps_limit−I₀): 수평 데드존 클램프 = 0,
# heave 3 A = 2.25 N — 모델이 없는 힘을 계획하거나 수평을 동원하는 것을 차단.
# depth-hold가 서면 method:=ssi로 재시도 — 절차는 아래 "SSI 재시도" 블록
```

**SSI 재시도 (nominal depth-hold 합격 후, 2026-08-19 bag 23_47 기준)** — 목표:
nominal에 남는 **+3~6 cm 편측 오프셋**(평형력이 마찰 데드존 내부라 생기는 구조
한계)을 SSI의 온라인 잔차 학습(d_world 주입)이 흡수하는지 확인. T2만 바꾼다
(Ctrl-C 후 재기동; T1/T3는 그대로):

```bash
ros2 run pkrc_wallscan_bridge wallscan_controller --ros-args \
  -p method:=ssi -p plant_json:=$MARINELAB_ROOT/config/pkrc_plant_hw2026.json \
  -p params_json:=$MARINELAB_ROOT/config/depthhold_rate_weights.json \
  -p thrust_limits:="[0.0,0.0,0.0,0.0,2.25,2.25]" \
  -p hold_z:=Z_HOLD -p depth_only:=true \
  -p z_top:=Z_HOLD -p z_bottom:=Z_HOLD -p sway_step:=0.0 -p reach_eps:=0.05
```

- method:=ssi 외 전부 nominal 시나리오-③과 동일 (SSI는 FixedWeightNMPC를 상속 —
  rate 모델·LATENCY PREDICTOR·가중치·캡이 그대로 탑재된다). SSI 하이퍼파라미터
  (`ssi_lr` 등)는 tuning attempt-2 trial 87 채택값이 노드 기본값.
- **원인 ⑩ 가드 (2026-08-20 bag 00_33)**: 1차 ssi 시도는 학습기의 회귀 쌍이
  0.4 s 데드타임에 오염돼 10–12 N 유령 외란을 주입, 13–15 cm 리밋사이클로 실패.
  수정 3종이 노드 기본값으로 탑재됨 — 지연 정렬 회귀 쌍(command_latency_s 사용),
  주입 저역통과 `ssi_d_tau`=3 s(안정성의 핵심), 클램프 `ssi_d_max`=5 N.
  이 값들을 끄지 말 것 (0으로 끄면 00_33 재현).
- 기동 확인: WARN 4개 동일 + `wallscan controller up: method='ssi'`.
- **enable 후 손 대지 말 것** (push 시험은 nominal bag에서만): 학습기는 모든
  미모델 힘을 잔차로 배우므로, 손/테더 개입은 오염이다.
- **접촉이 생기면 disable로 끝나지 않는다**: 학습기 가중치(alpha)는 re-enable에도
  유지되도록 설계돼 있다 (`reset_episode`는 alpha 보존). 바닥·수면·테더 접촉이
  있었으면 **T2 노드를 재시작**하고 다시 enable.
- 판정 (60–90 s bag): controller_debug가 ssi에서 4채널 연장된다 —
  `[7:10]=d_world`(N, world), `[10]=1스텝 예측오차 노름`. 합격 신호는
  ① d_world[2]가 수십 초 안에 준정상값으로 수렴 (기대: 실부력잔차 ~+0.2–0.5 N),
  ② |z−z_ref| 평균이 nominal의 +4 cm에서 유의미하게 감소, ③ 리플은 nominal과
  동급(±2–3 cm), heave 비포화 유지. 예측오차가 지속 증가하거나 d_world가 발산하면
  즉시 OFF — 잔차 학습이 지연/접촉과 얽힌 것 (bag 가져와 분석).

기대 로그: `building acados solver ...` (**rate 모델 첫 기동은 nx 변경으로 C 코드
재생성 — 수 분 소요가 정상**; 5분 초과 시 `rm -rf ~/.cache/wallscan_acados` 후
재시작) → `wallscan controller up ...`
→ **WARN 4개 필수 확인**: `ACTUATOR-RATE model: force slew ...` (없으면 plant
JSON이 구버전 — rsync 확인) + `LATENCY PREDICTOR: rolling state forward 0.40 s ...`
(없으면 plant JSON 구버전 — 없이는 ~0.4 s 왕복 지연이 16 cm/4.2 s 캡 스윙
리밋사이클을 만든다, bag 23_03 실측) + `DEPTH-ONLY mode: zeroed werr ...` (없으면
depth_only 누락 — 2026-08-18에 두 번 누락, 바닥 고착·리밋사이클 재발) +
`DEPTH-HOLD mode: phase machine bypassed, z_ref -> ...`
→ `/wallscan/u`·current_cmd 50 Hz.

enable 후 30초 안에 확인 (`ros2 topic echo /wallscan/u`):
- **u[0..3] ≈ 0** — 단, 이 규칙은 **heave가 비포화일 때만** 유효 (04_15 bag 실측:
  depth_only가 켜져 있어도 heave 포화 중엔 옵티마이저가 pitch 트림을 통해 수평
  스러스터를 cost-free로 동원해 u[0..3]가 u[4]와 동기 부호반전, |평균| 0.1–0.3).
  진짜 누락 시그니처는 03_41처럼 **준-DC 클램프 고착**(|u0| 평균 ≥0.5, 부호 고정)
  — 허구 벽 오차가 상수라서 흔들리지 않는다. 헷갈리면 T2의
  `DEPTH-ONLY mode:` WARN 로그가 정본.
- **u[4], u[5]가 ±1 포화 왕복이 아님** — 포화 왕복(03_41: 틱의 78%)이면
  z_ref가 튀고 있는 것 (hold_z 누락) → 즉시 OFF.
  ※ 04_15 실측: hold_z·depth_only 둘 다 정상이어도 **relay 리밋사이클**(근본원인
  ⑧, werr z=40 near-relay + teleop 램프 지연)로 81% 포화·±5 cm 진동이 남는다 —
  이 경우 hold_z 누락 오진 금지, 진단은 phase 고정 여부로 구분 (누락이면 위상
  순환, 원인 ⑧이면 phase 0 고정인데 포화 왕복). ACTUATOR-RATE 모델 + rate 전용
  가중치가 이 원인의 수정 — 그 구성에서 u[4] 포화 왕복이 다시 보이면 params_json
  누락(기본 z=40이 vz 지연과 공진)부터 의심하고, params_json도 정상인데 **~4 s
  주기 캡-투-캡 왕복**이면 원인 ⑨ = LATENCY PREDICTOR 미적용 (bag 23_03: 왕복
  지연 ~0.4 s가 모든 가중치 세트를 리밋사이클로 몰았다 — WARN 4개 중
  `LATENCY PREDICTOR` 확인).
- **u[4]/u[5]가 캡(−0.61 = −2.25/3.68)에 한 방향으로 고정된 채 z 접근이 멈추면
  즉시 OFF** — 도달 불가능한 z_ref다 (22_36 실측: 바닥 아래 목표를 향해 15 s간
  캡 추력으로 바닥 고착). rate 모델의 정상 정착은 u[4]가 캡 아래에서 완만히
  움직이는 모습이다 (틱당 변화 ≤ 0.09).
- controller_debug의 phase(2번째 값)가 **0에 고정** (위상 순환 = hold_z 누락)
- controller_debug의 solve_ms: nx 19로 **Jetson h20/rti4가 ~26 ms** (bag 22_36
  실측; 이전 15.6–16.2). 20 ms 틱을 넘는 soft overrun — teleop stale 문턱(0.5 s)
  대비 무해하고 f_act 부기는 보수 방향이지만, 더 커지면 `-p rti_iters:=3` 또는
  `-p horizon:=15`를 E4(c) 프로토콜로 벤치 후 적용.

**T4 (teleop — mj_ws 것, hero_ws teleop은 먼저 종료)**:

```bash
source /opt/ros/humble/setup.bash && source ~/mj_ws/install/setup.bash
ros2 run pkrc_control keyboard_control_teleop
```

`g` → `WALLSCAN AUTO ON` 로그, 10 s쯤 수동 복귀 없이 유지되는지 관찰.
**이 터미널이 비상정지 (아무 키)** — 세션 내내 손 닿는 곳에.

**T5 (기록/명령)**:

```bash
# ① bag 시작
ros2 bag record /wallscan/state /wallscan/estimator_debug /wallscan/u \
  /wallscan/current_cmd /wallscan/controller_debug /wallscan/enable \
  /teleop/thruster_currents /bar10xt/depth /imu/data /dvl/data
# ② 로봇을 목표 심도 근처에서 손 놓기 → ③ 폐루프 ON
ros2 topic pub -1 /wallscan/enable std_msgs/msg/Bool "{data: true}"
#    → T2 로그 "scan enabled: anchored at z=..." 확인
# ④ 30 s 관찰: 심도 ±10 cm, heave 전류 0.7–1.5 A 기대 (3 A 포화 왕복이면 즉시 OFF)
ros2 topic pub -1 /wallscan/enable std_msgs/msg/Bool "{data: false}"   # ⑤ OFF
# ⑥ 정상이면 재-enable 2–3분 + 손으로 10 cm 눌렀다 놓기(복원 응답) → bag 종료
```

이상 징후별 1차 대응: state 안 나옴 → T1 로그(무엇을 기다리는지 말해줌);
u 안 나옴 → T2가 아직 빌드 중이거나 state 미수신(`state stream stale` 경고);
auto가 자꾸 풀림 → current_cmd 발행 확인(T3 하트비트) + 키보드 접촉;
enable 후 무반응 → T2 `scan enabled` 로그 유무와 controller_debug의
enabled 플래그 확인.

**안전 수칙 (소형 수조 필수)**:
- enable은 짧게 (첫 시도 ≤30 s), 심도 0.2 m 이탈·과도 수평 이동 시 즉시 OFF.
- z_top/z_bottom을 수조 심도(0.85 m)보다 깊게 두지 말 것 — 기본값 그대로
  켜면 z_bottom=1.0을 찾아 바닥으로 파고든다.
- 트림 상태 확인: 이 plant JSON은 납 0.5 kg 장착(+0.24 N) 기준 — 납을 뗀
  로봇(+4.7 N)이면 heave 평형 전류가 3 A대로 올라가 hold가 더 거칠어진다.
