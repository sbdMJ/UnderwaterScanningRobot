# hero_ws 패치 모음

로봇 운용 워크스페이스(`~/hero_ws`)는 이 리포 밖에서 관리되므로, 여기에 필요한
수정은 **패치 파일**로 추적한다. 로봇(Jetson)에서 적용:

```bash
cd ~/hero_ws
patch -p1 --dry-run < <repo>/marinelab/ros/hero_ws_patches/0001-teleop-wallscan-auto-mode.patch
patch -p1           < <repo>/marinelab/ros/hero_ws_patches/0001-teleop-wallscan-auto-mode.patch
colcon build --packages-select pkrc_control
```

## 0001 — teleop wallscan auto 모드 (+57줄, 2026-08-09)

`keyboard_control_teleop.py`에 auto 모드 추가:

- `/wallscan/current_cmd` (Float64MultiArray, A 단위, VESC 순서 T1..T6) 구독 —
  `pkrc_wallscan_bridge`의 `thrust_mapper`가 발행.
- **`g` 키로 auto 토글, 그 외 아무 키나 누르면 즉시 수동 복귀.** 명령이 0.5 s
  끊겨도 수동 복귀 + 정지.
- auto 전류도 수동 경로와 동일하게 극성·게인·max_current 클램프·데드존·램프를
  통과한다 — CAN 소유자는 여전히 teleop 하나.
- 기반 버전: 2026-08-09 시점 hero_ws 스냅숏 (bak-20260730 이후). 적용 실패 시
  세 블록을 수동 반영: ① __init__ 구독+상태, ② 키 처리 'g' 토글, ③ target 딕트
  뒤 auto override, ④ `_wallscan_current_cb` 메서드.

검증 상태: py_compile 통과. 실행 검증은 로봇에서 — 순서: 물 밖 프로펠러 회전
확인 (`docs/experiments/sim-to-real/thruster_mapping.md` §3 프로토콜) → 수조
depth-hold.
