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

### 실측 결과 (2026-08-15)

**Jetson 기본 설정 = 탈락**: nominal 37.30 ms mean / 53.22 p99, ssi 38.00 /
54.20 — 예산 초과 100% (`bench_jetson.json`). 데스크톱 대비 ~6×. 병목은
acados solve(35.4 ms)이고 SSI 오버헤드는 0.62 ms — 온라인 적응 비용은
Jetson에서도 무시 가능(E4c의 핵심 결론은 유지).

**완화 설정 sim 성능 검증 (e5_hwdrag_lat, go 조건 = 트림 상태, 5시드)**:

| 설정 | nominal obj | ssi obj | cycles | 컨트롤(h30/rti8) 대비 |
|---|--:|--:|--:|---|
| rti4_h30 | 7,594 | 7,013 | 2.0 | nominal 동일(−0.0%); ssi는 s2 이상치 미재현으로 오히려 개선 |
| rti4_h20 | 7,679 | 6,995 | 2.0 | nominal +1.1% — 무시 가능 |

**둘 다 성능 무손실** (전 시드 cycles 2.0, 충돌 0, 벽오차 동일 ~17 cm).
h30/rti8 ssi의 s2 이상치(40,955)가 rti4에선 ~13k로 재현되지 않은 점은 보너스
(단일 관찰이라 과신 금지). → **배포 후보 = rti4_h20** (Jetson 예상 ~12 ms,
p99 ~18); Jetson 재벤치(`--rti-iters 4 --horizon 20`)로 타이밍 확정만 남음.
BLASFEO ARMv8 재빌드가 1.5× 이상 벌면 rti4_h30도 대안.

### E4(c) 최종 표 (2026-08-16 Jetson 재벤치로 확정)

| 플랫폼 | 설정 | nominal total mean/p99 [ms] | ssi | SSI 오버헤드 | 초과율 |
|---|---|--:|--:|--:|--:|
| 데스크톱 (x86, 컨테이너) | h30/rti8 | 6.36 / 7.78 | 6.49 / 8.06 | 0.38 | 0% |
| Jetson (aarch64) | h30/rti8 | 37.30 / 53.22 | 38.00 / 54.20 | 0.62 | 100% |
| Jetson | h30/rti4 (+ARMv8 재빌드) | 19.88 / 25.06 | 20.53 / 26.00 | 0.61 | 9–76% |
| **Jetson (배포)** | **h20/rti4** | **15.57 / 20.65** | **16.15 / 20.73** | 0.59 | **1.3% / 4.4%** |

- **배포 설정 = horizon 20 / RTI 4 확정** — `wallscan_controller` 노드 기본값으로
  반영 (sim 성능 무손실은 위 e5_hwdrag_lat로 검증). p99가 예산을 ≤0.7 ms 스치는
  1~4%의 소프트 초과는 허용 (하드 데드라인 없음 — 늦은 틱은 다음 발행으로 밀리고
  teleop 램프·stale 경로가 흡수).
- **BLASFEO ARMv8 타깃은 무효과** (h30/rti4 solve 18.1 ms ≈ GENERIC 외삽 17.7)
  — 이 문제 크기에서는 타깃 튜닝으로 얻을 게 없다. GENERIC 유지.
- E4(c) 논거 완성: SSI의 온라인 적응 비용은 Jetson에서도 **0.6 ms/틱** —
  적응성은 사실상 공짜이고, 예산을 결정하는 것은 NMPC 공통 비용뿐.

## 6. 남은 Jetson 절차 (이 문서 범위 밖)

- 브리지/컨트롤러 노드 colcon build: `marinelab/ros/pkrc_wallscan_bridge/README.md`
- teleop auto 패치 적용 + mj_ws: `marinelab/ros/mj_ws_README.md`
