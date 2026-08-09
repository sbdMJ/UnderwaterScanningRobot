# 스러스터 대응표와 u→전류 매핑 (Phase C-③, Step 0)

> 2026-08-09 작성. 코드 근거: sim `PKRCThrusterCfgFixedTAM.allocation_matrix`
> (`marinelab/marinelab/assets/pkrc/pkrc.py:190`), 실기체
> `hero_ws/src/pkrc_control/pkrc_control/keyboard_control_teleop.py` (`self.tam` :513,
> `thruster_polarity` :526, `mix_thrusters` :757).
> **⚠ 아래 표의 heave 쌍은 실물 검증 전까지 가설이다 — §3의 프로토콜로 확정할 것.**

## 1. 대응표 (컨트롤러 u 인덱스 → VESC)

sim u는 MPC allocation matrix의 열 순서, u=±1 ↔ ±40 N (`PlantParams.max_thrust`).
실기체 전류 명령은 teleop의 `send_current(vesc_id, A)` 규약 (±5 A 하드클램프, 내부
스케일 ×1000).

| u idx | sim 기하 (TAM 열에서 유도) | 실기체 | VESC | teleop 극성 | 근거·상태 |
|--:|---|---|---|--:|---|
| 0 | surge, y=+0.15 (Mz −0.15) | T1 surge_left | 0x151 | +1 | yaw 부호 패턴 일치 (teleop T1 yaw −) — **신뢰** |
| 1 | surge, y=−0.15 (Mz +0.15) | T2 surge_right | 0x152 | +1 | yaw 부호 패턴 일치 — **신뢰** |
| 2 | sway, x=+0.15 (Mz +0.15), z=−0.09 | T3 sway_left | 0x153 | +1 | yaw 부호 패턴 일치 — **신뢰** |
| 3 | sway, x=−0.15 (Mz −0.15), z=−0.09 | T4 sway_right | 0x154 | **−1** | 모터 회전방향 반대 (teleop 주석) — **신뢰** |
| 4 | heave, y=+0.16 (Mx +0.16) | T5 "수직 상" | 0x155 | +1 | **미검증 — §2 불일치** |
| 5 | heave, y=−0.16 (Mx −0.16) | T6 "수직 하" | 0x156 | +1 | **미검증 — §2 불일치** |

## 2. 발견된 불일치 — heave 쌍의 기하학

- **sim (FixedTAM)**: heave 2기가 **좌/우 y=±0.16**에 나란히 장착 → 차동으로 **롤(Mx)
  권한**이 있고, sway 기생 롤모멘트(0.09 arm)를 heave 차동으로 상쇄하는 것이 FixedTAM
  틸트 보상의 핵심이다 (실측 0.000° 문서화돼 있음).
- **teleop TAM**: T5 행 Fz=−1, T6 행 Fz=+1 — 같은 heave 명령에 두 스러스터가 **반대
  방향 추력**. README 명칭도 "수직 상/하"(위치인지 방향인지 불명). 이대로면 실기체
  heave 쌍은 롤 권한이 없거나(마주보는 쌍) 기하가 sim과 전혀 다르다.
- 파급: 실기하가 sim과 다르면 **MPC의 allocation matrix를 실측 기하로 교체**해야
  한다. 코드는 준비돼 있다 — `pkrc_plant_fixed_tam.json`의 `allocation_matrix`만
  실측값으로 바꾸면 컨트롤러/시뮬 재검증까지 그대로 돈다 (sim 재검증: 교체한 TAM으로
  e5_ekf 셀 재실행).

## 3. 실물 검증 프로토콜 (벤치, ~30분)

`thruster_test.py`(전류 스텝 기능 내장, ±5 A 클램프)로 스러스터 1기씩:

1. 로봇을 물 밖(또는 얕게 잠긴 상태로 고정)에서 T1부터 하나씩 **+0.5 A** 인가.
2. 기록: 프로펠러 회전 방향 / 유동(기포) 방향 / 로봇이 밀리는 방향.
3. 표를 채운다: "VESC +전류 → 실제 추력 방향(body frame)" 6행.
4. heave 쌍(T5/T6)은 추가로: 두 기의 **장착 위치**(좌/우인지 상/하 대향인지)와
   같은 부호 전류에서 추력 방향이 같은지/반대인지 확정.
5. 결과가 §1 표와 다르면 이 문서와 `thrust_mapper`의 `polarity`/`order` 파라미터,
   필요 시 `pkrc_plant_fixed_tam.json`의 TAM을 수정한다.

## 4. u→전류 변환 (Step 1–2에서 채울 캘리브레이션)

전류 제어(VESC)는 토크 제어이고 프로펠러는 추력·토크 모두 ~rpm²이므로 **추력 ≈ k×전류**
(1차 근사 선형). 변환 체인:

```
u[-1,1] (sim 순서) ──thrust_mapper──▶ A (VESC 순서·극성 적용) ──teleop auto──▶ CAN
         F = u × 40 N              I = F / k_i  (k_i: 스러스터별 N/A)
```

- **1차값 (Step 1)**: T200 데이터시트의 운용 전압 thrust–current 표로 k_i 초기화.
  주의: VESC 전류(모터 상전류) ≠ 데이터시트 입력 전류 — 벤치에서 대조.
- **실측 (Step 2)**: bollard pull (권장) 또는 전류 스텝→DVL 정상속도 역산.
- **한계 정합**: 실측 최대추력이 40 N 미달이면 (teleop max_current 3/5 A 기준 가능성
  높음) `pkrc_plant_fixed_tam.json`의 `max_thrust`를 실측치로 낮춘다 — 컨트롤러가
  낼 수 없는 추력을 계획하지 않도록.
- 캘리브레이션 전 기본값: k_i = max_current 기준 선형 (u=1 → surge/sway 3 A,
  heave 5 A) — **teleop 수동 운전과 같은 스케일**이라 과추력 위험은 없지만,
  u↔N 정합은 깨져 있으므로 폐루프 게인이 사실상 축소된 상태다. 캘리브레이션 전
  깊이 유지 등 저대역 시험만 할 것.

## 5. 통합 구조 (Step 3 구현)

```
wallscan_controller ──/wallscan/u──▶ thrust_mapper ──/wallscan/current_cmd──▶ teleop(auto)
                                     (repo 소유: 순서·극성·k_i)              (CAN 소유 불변)
```

- CAN 버스 소유자는 teleop 하나뿐 (이중 쓰기 금지). auto 모드에서도 teleop의
  기존 안전 경로(±max_current 클램프, 데드존 보상, 램프 0.5 A/tick, ESC 정지)를
  그대로 통과한다.
- teleop auto 모드 규약: `g` 키로 진입(로그 확인 후), **아무 이동키/`x`로 즉시 수동
  복귀**, `/wallscan/current_cmd` 0.5 s 이상 끊기면 자동 수동 복귀 + 정지.
- teleop 수정은 최소 diff로 하고, 패치 파일을 이 리포에 커밋해 추적한다
  (`marinelab/ros/hero_ws_patches/`).
