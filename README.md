# UnderwaterScanningRobot

PKRC UUV(22.8 kg, T200×6)가 원통 수조(R=6 m, H=10 m) 벽면을 Ping1D 단일빔 소나로 스캔하는
강화학습 프로젝트. Isaac Lab(Isaac Sim) 기반 학습 환경 `marinelab`과 학습된 정책 체크포인트를 포함한다.

## 구성

| 경로 | 내용 |
|:---|:---|
| `marinelab/` | 학습 환경 소스 전체 — wallscan 태스크(13항 보상·스캔 상태기계·센서 모델·DR), PKRC 자산(USD/OBJ), 테스트 84개 |
| `isaaclab/` | Isaac Lab 소스 전체(upstream 47aa161 기준, 커스텀 수정 없음) — 버전 재현용 |
| `checkpoints/` | 최종 정책 `rb_train_model_7998.pt` + 학습 설정(agent/env yaml) |
| `results/` | 평가 궤적 플롯(공칭/스트레스 DR) |

학습 곡선: https://wandb.ai/yju1121-postech/pkrc_wallscan

## 설치 (clone만으로는 실행 불가 — Isaac Sim 별도 설치 필요)

```bash
# 1. clone + LFS (메시/USD 수신 — LFS 없으면 포인터 파일만 받아짐)
git lfs install
git clone https://github.com/yuuu1121/UnderwaterScanningRobot.git
cd UnderwaterScanningRobot

# 2. Isaac Sim 5.1 별도 설치 (NVIDIA 공식, ~10GB) 후 심링크
ln -s <isaac-sim-설치경로> isaaclab/_isaac_sim

# 3. Isaac Lab 의존성 설치
cd isaaclab && ./isaaclab.sh --install && cd ..

# 4. marinelab 설치 (Isaac 번들 파이썬 사용 — conda/venv 만들지 말 것)
<isaac-sim-설치경로>/python.sh -m pip install -e ./marinelab
```

⚠️ setuptools 81+ 환경에서 Isaac Lab 코어 설치 시 `flatdict` 빌드가 무음 실패할 수 있다
(`pkg_resources` 제거 때문). 실패 시 `pip install "setuptools<81"` 후 재시도.

## 검증·학습·평가

```bash
cd isaaclab
# 테스트 84개
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./isaaclab.sh -p -m pytest ../marinelab/tests/ -q

# 학습 (Stage3 5000 iter → Train 3000 iter resume 체인; from scratch 재현 레시피)
./isaaclab.sh -p ../marinelab/scripts/train.py --task Isaac-PKRC-WallScan-Stage3-Direct-v0 \
  --num_envs 4096 --headless --max_iterations 5000 --run_name stage3
./isaaclab.sh -p ../marinelab/scripts/train.py --task Isaac-PKRC-WallScan-Train-Direct-v0 \
  --num_envs 4096 --headless --max_iterations 3000 --run_name train \
  --resume --load_run <stage3-로그폴더명>
```

⚠️ resume은 반드시 CLI `--resume --load_run <폴더명>`으로 지정한다. hydra 오버라이드
`agent.resume=True`는 CLI 기본값에 덮여 **무효**다. 로그의
`Loading model checkpoint from:` 라인으로 실제 로딩을 확인할 것.

`scripts/train.py`에는 탐색 σ를 [0.1, 1.5]로 제한하는 클램프가 들어 있다(σ 소멸/폭주 방지) —
rsl_rl 표준 train.py로 대체하지 말 것.

## 최종 성능 (공칭 조건, 8환경 × 180 s)

| 지표 | 값 |
|:---|:---|
| 기울기 (수직 이동 / 옆걸음) | 0.90° / 2.20° |
| 스캔 속도 (heave / sway) | 0.199 / 0.123 m/s (목표 0.20 / 0.12) |
| 게걸음 감사 (yaw−θ) | 1.50° |
| 호길이 추정 오차 (ŝ) | 평균 1.33 cm |
| 스캔 사이클 달성 | 2.0 |

스트레스 DR(초기 자세 ±45°, 유체계수 ±50%, 센서 마운트 ±8 cm)에서는 기울기 평균 14.3°로
열화되지만 속도 유지·게걸음 없음·임무 완주는 유지된다(`results/trajectory_stress_dr.png`).

### 위 표의 재현 상태 (2026-07-31 확인)

이 표를 만든 런의 궤적은 저장돼 있지 않다. `results/trajectory_{nominal,stress_dr}_repro.npz`는
같은 체크포인트·같은 태스크로 나중에 다시 돌린 **별개의 런**이고, 일부 지표만 일치한다:

| 지표 | 위 표 | `*_repro.npz` 재계산 |
|:---|---:|---:|
| 기울기 수직 / 옆걸음 | 0.90° / **2.20°** | 0.900° / **1.715°** |
| 스캔 속도 heave / sway | 0.199 / 0.123 | 0.197 / 0.121 |
| **게걸음(crab)** | **1.50°** | **0.627°** |
| ŝ 오차 | 1.33 cm | 1.185 cm |
| 사이클 | 2.0 | 2.000 |

기울기(수직)와 사이클은 맞고 crab이 2.4배, 옆걸음 기울기가 1.28배 다르다. `play.py`에 시드
옵션이 없어 스폰과 센서 노이즈 스트림이 런마다 달라지는 것이 가장 유력한 원인이지만, 원본 런의
궤적이 없어 특정할 수 없다. **위 표의 값을 재현값으로 바꾸지 않았다** — 서로 다른 런의 숫자를
같은 표에 섞는 것이고, 재현값이 더 좋아 보이는 방향이라 성능을 과장하게 된다.

`eval_metrics`의 레그 속도 정의는 2026-07-31에 수정됐다(잘린 레그의 왕복을 전진으로 계상하던
문제). 다만 **위 표의 값은 그 수정에 영향받지 않는다** — `*_repro.npz`를 새 정의로 재채점해도
`0.627->0.627`, `0.197->0.197`로 동일하다. 8환경 × 완결 에피소드라 레그가 24/32개 확보돼
잘린 레그의 비중이 작았기 때문이다. 결함이 크게 드러난 것은 1환경 × 에피소드 0을 채점한 경우다
(레그 3/4, 그중 1개가 잘려 heave 속도가 0.196 대신 0.233으로 보고됐다).

표를 재현 가능하게 만들려면 `play.py`에 시드 옵션을 추가하고 다중 시드로 재수립해야 한다.
