# Jetson(aarch64) acados 빌드 + 추론 벤치 절차 (Phase D)

> 데스크톱과 버전을 정합시킬 것: **acados v0.5.3 / acados_template 0.5.1 / casadi 3.7.0**
> (컨테이너 빌드와 동일 — CLAUDE.md의 acados 절 참조). Jetson은 self-host 빌드
> (크로스 컴파일 불필요).

## 0. 전제

- Jetson에 이 리포 체크아웃 (`~/UnderwaterScanningRobot` 가정), mj_ws/hero_ws와 무관.
- python3 ≥ 3.8 + pip, cmake ≥ 3.16, build-essential.
- torch aarch64 (CPU면 충분 — `scan_state_machine`의 1-env 텐서 연산뿐):
  `pip3 install torch --index-url https://download.pytorch.org/whl/cpu` 또는
  JetPack 동봉 wheel. numpy는 casadi와 호환되는 1.x 유지 권장.
- `pip3 install gymnasium` — `tasks.pkrc_wallscan.__init__`의 gym.register가
  당긴다 (등록 문자열만이라 가볍고 isaaclab과 무관). bag 리플레이까지 하려면
  `pip3 install rosbags`.

## 1. acados 소스 빌드

```bash
git clone https://github.com/acados/acados.git ~/acados
cd ~/acados && git checkout v0.5.3 && git submodule update --recursive --init
mkdir -p build && cd build
# BLASFEO 타깃: Orin(Cortex-A78AE)/Xavier(Carmel) 모두 우선 GENERIC으로 시작.
# (ARMV8A_ARM_CORTEX_A57은 Nano/TX 계열용. GENERIC 대비 이득은 벤치로 확인 후 결정 —
#  솔버가 20 ms 예산의 1/3이면 최적화 타깃 튜닝은 불필요.)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DACADOS_WITH_HPIPM=ON -DACADOS_WITH_QPOASES=ON \
      -DBLASFEO_TARGET=GENERIC \
      -DACADOS_INSTALL_DIR=$HOME/acados ..
make -j$(nproc) install
```

## 2. 파이썬 인터페이스

```bash
pip3 install "casadi==3.7.0"          # aarch64 manylinux wheel 존재
pip3 install --no-deps --no-build-isolation -e ~/acados/interfaces/acados_template
pip3 install Deprecated wrapt         # acados_template 런타임 의존
```

setuptools-scm이 "dubious ownership"으로 죽으면 (데스크톱에서 겪은 함정):
`git config --global --add safe.directory ~/acados`.

## 3. t_renderer (코드젠 템플릿 렌더러) — aarch64 함정

acados 코드젠은 `~/acados/bin/t_renderer` 실행 파일이 필요하다. 첫 솔버 빌드 때
자동 다운로드를 시도하지만 **x86_64 바이너리를 받아와 aarch64에서 `Exec format
error`로 죽는 사례가 흔하다**. 해법 둘 중 하나:

```bash
# (a) 릴리스에 aarch64 자산이 있으면 그것을:
#     https://github.com/acados/tera_renderer/releases 에서
#     t_renderer-v*-linux-arm64 다운로드 → ~/acados/bin/t_renderer 로 두고 chmod +x
# (b) 없으면 러스트로 직접 빌드 (수 분):
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
git clone https://github.com/acados/tera_renderer.git ~/tera_renderer
cd ~/tera_renderer && cargo build --release
cp target/release/t_renderer ~/acados/bin/
```

## 4. 환경 변수 (셸 프로파일에 고정)

```bash
export ACADOS_SOURCE_DIR=$HOME/acados
export LD_LIBRARY_PATH=$HOME/acados/lib:$LD_LIBRARY_PATH
```

## 5. 검증 = 추론 벤치 (E4c의 Jetson 절반)

```bash
cd ~/UnderwaterScanningRobot
python3 marinelab/scripts/experiments/bench_inference.py --steps 1000 --label jetson
```

- isaaclab 불필요 (스크립트가 marinelab 패키지 __init__을 우회하는 shim 내장).
- 첫 실행은 OCP 코드젠+컴파일로 수십 초 걸림 (1회성; `isaaclab/logs/…`가 아닌
  `--export-root`로 위치 변경 가능). 이후 실행은 곧바로 돈다.
- plant는 기본으로 실측 확정본 `marinelab/config/pkrc_plant_hw2026.json`을 쓴다.
- 산출: `experimental_results/e4_inference/bench_jetson.json` → 데스크톱 결과와
  함께 E4(c) 표. **판정 기준: total p99 < 20 ms (50 Hz 제어 주기).**

데스크톱 기준치 (2026-08-15, i7/컨테이너, horizon 30, RTI 8회, 동일 프로토콜):

| method | total mean/p99 [ms] | solve mean [ms] | SSI 오버헤드 mean [ms] | 예산 초과 |
|---|--:|--:|--:|--:|
| nominal | 6.36 / 7.78 | 6.09 | — | 0% |
| ssi | 6.49 / 8.06 | 6.11 | 0.38 (RFF 100개 + RK4 예측) | 0% |

Jetson이 데스크톱 대비 ~3× 느려도 p99 ≈ 24 ms로 경계선 — 초과 시 완화 순서:
`--rti-iters 8→4` (RTI 반복이 solve 시간의 지배항) → `--horizon 30→20` →
BLASFEO 타깃 최적화. 완화를 쓰면 **같은 설정으로 e5 sim 셀을 재검증**해
성능 열화를 확인할 것 (rti_iters/horizon은 config `옵션`으로 이미 노출).

## 6. 남은 Jetson 절차 (이 문서 범위 밖)

- 브리지/컨트롤러 노드 colcon build: `marinelab/ros/pkrc_wallscan_bridge/README.md`
- teleop auto 패치 적용 + mj_ws: `marinelab/ros/mj_ws_README.md`
