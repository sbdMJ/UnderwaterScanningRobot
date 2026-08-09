# Diff-WMPC 업데이트 이식 → 시뮬레이션 실험 완료 TODO (인수인계 문서)

> 대상: 다른 머신에서 최적화 중인 Diff-WMPC 모델이 확정된 뒤, 그것을 이 저장소에
> 이식하고 남은 실험(E1/E2/E3 diff, E2b, E4a, F5)을 완료하는 세션.
> 전제 지식: `CLAUDE.md`(모니터링/스팟체크 규칙), `docs/experiment_work_directives.md`
> (red-flag 규칙), `marinelab/scripts/experiments/README.md`(러너 사용법).

## 아키텍처 요약 — 무엇을 고치고 무엇을 건드리지 않는가

```
[다른 머신에서 옴]                       [이 저장소, 수정 금지]
marinelab/algorithms/diff_wmpc.py  ←→  marinelab/control/diff_wmpc_ctrl.py (어댑터)
checkpoints/diff_wmpc/policy_final.pt        └─ fixed_nmpc.py → mpc_controller (acados)
```

- **이식 대상**: `marinelab/algorithms/diff_wmpc.py`(WeightPolicy + DiffWMPCLearner)와
  체크포인트. 학습 스크립트 `marinelab/scripts/train_diff_wmpc_wallscan.py`도 타 머신
  쪽이 변했으면 함께.
- **수정 금지**: `control/diff_wmpc_ctrl.py`는 inference-only 어댑터로 설계상 불변.
  어댑터는 `WeightPolicy(NE+2, NE, nu, werr_init=…, wu_init=…)`를 **기본 인자**
  (history_len=4, hidden=128, bounds 0.1–5000 / 5e-3–5.0, log_scale=True)로 생성한 뒤
  `state["policy"] if "policy" in state else state`를 `load_state_dict`한다.
- **함정 (가장 중요)**: bounds는 buffer라 체크포인트에 실려 오지만(안전),
  **hidden / history_len / 피처 구성(NE+2)은 buffer가 아니다.** 타 머신에서 이들을
  바꿨다면 — shape 불일치로 load가 죽거나(다행), 피처 의미만 바뀌어 **조용히 다른
  가중치를 내는** 사고가 난다. 이식된 `diff_wmpc.py`의 **ctor 기본값이 학습 시 값과
  일치**하도록 맞추는 것까지가 이식이다 (어댑터를 고치는 게 아니라).

## Phase 0 — 수령물 확인 (타 머신에서 받아야 할 것)

- [ ] `algorithms/diff_wmpc.py` 최종본 (또는 현 커밋 대비 patch) + 변경 요약
- [ ] `checkpoints/diff_wmpc/policy_final.pt` (형식: `{"policy": sd, "opt": …}` —
      DiffWMPCLearner.state_dict; policy-only sd여도 어댑터가 처리함)
- [ ] 학습 하이퍼파라미터: **hidden, history_len, 피처 정의(차원), bounds, log_scale**
      — ctor 기본값 대조용
- [ ] 학습 비용 기록 (W&B run 링크, wall-clock, env steps) — E4(b) 표의 diff 행
- [ ] 학습에 쓴 태스크/DR 설정 (E2 zero-shot 해석에 필요)

## Phase 1 — 코드 이식

- [ ] `marinelab/algorithms/diff_wmpc.py` 교체/병합. **API 계약 유지 확인**:
      `WeightPolicy` 클래스명, `reset_history()`, state_dict 레이아웃, 그리고
      `control/diff_wmpc_ctrl.py`·`_sim_loop.build_controller`가 쓰는 시그니처.
- [ ] 순수성 유지: `algorithms/`는 torch-only (isaaclab/pxr import 금지 — 네이티브
      테스트가 이를 보증).
- [ ] ctor 기본값 = 학습값 대조 (Phase 0 목록). 불일치 시 이식본에서 기본값을 맞출 것.
- [ ] 새 체크포인트를 `checkpoints/diff_wmpc/policy_final.pt`로 배치
      (이전 것은 `policy_final_pre<날짜>.pt`로 보존).

## Phase 2 — 오프라인 검증 (sim 불필요, 수 분)

- [ ] 네이티브 테스트: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest marinelab/tests -q`
      (기준 111개 + control/ 4파일 — 전부 통과해야 함)
- [ ] 체크포인트 로드 스모크 (CPU):
  ```python
  import torch, numpy as np
  from marinelab.algorithms.diff_wmpc import WeightPolicy
  NE, NU = 12, 6
  p = WeightPolicy(NE + 2, NE, NU)          # 어댑터와 동일한 기본 인자
  st = torch.load('checkpoints/diff_wmpc/policy_final.pt', map_location='cpu')
  p.load_state_dict(st['policy'] if 'policy' in st else st)   # 여기서 죽으면 Phase 1 재점검
  p.eval(); p.reset_history()
  w = p(torch.zeros(1, NE + 2))             # forward 시그니처는 이식본에 맞게
  print(w)                                   # bounds 안의 유한값이어야 함
  ```

## Phase 3 — 컨테이너 스모크 (1셀)

환경 전제: `underwater-scan:5.1` 이미지 + `~/docker/acados` 마운트 (없으면
`CLAUDE.md`의 acados 재빌드 절차부터).

```bash
./docker/run.sh './isaaclab.sh -p ../marinelab/scripts/experiments/run_experiment.py \
    diff --cond nominal --seed 0 --config ../marinelab/scripts/experiments/configs/e1_nominal.yaml'
```

- [ ] `[SCORE]`가 유한값, `metrics_diff_nominal_s0.json` 생성, `collided=False`
- [ ] npz의 `aux_*` 채널(가중치 시계열)이 bounds 안에서 **시변**하는지 확인 —
      상수면 정책이 죽은 것 (feat 불일치 의심)
- [ ] `[COST]`의 solve_ms가 ~8–10 ms급 + fail 0% (정적 NMPC와 동일 솔버이므로)

## Phase 4 — 게이트 (red-flag 사전 점검)

부모 논문 주장 C1(가변 > 정적)이 성립해야 하므로, **현재 확정된 기대치**와 대조:

| 실험 | diff의 기대 위치 | 비교 기준치 (커밋 dabe7ad) |
|---|---|---|
| E1 s0 1셀 | bo(17.4)·ssi(17.7) 동급 이하 | nominal s0 = 236 |
| E1 5-seed 평균 | ≤ bo 679 / ssi 697 | nominal 969 |
| E2 조건 평균 | ≤ ssi 3,388–3,464 | nominal 4,234–4,291 |
| E3 step/sine 평균 | ≤ ssi 1,173 / 956 | nominal 2,107 / 1,566 |

- [ ] 스모크 1셀이 위 표에서 크게 벗어나면 **본 실험 진입 금지** —
      `experiment_work_directives.md` §2의 5단계 원인 분석부터 (특히 1번: 피처/ctor
      불일치, 3번: aux 채널로 적응 작동 계측).

## Phase 5 — 본 실험 (E1 diff 5셀 + E2 diff 9셀 + E3 diff 6셀)

**규칙 (CLAUDE.md)**: 셀당 별도 프로세스(멀티셀 세션은 과거 무한루프 이력 — 수정됐지만
표준은 per-cell), 30분+ 작업이므로 **cron 모니터링 루프 등록**, 진행 판정은 stdout이
아니라 **metrics 파일 수** (stdout은 리다이렉트 시 블록 버퍼링).

워커 스크립트 (세션 스크래치패드에 저장 후 사용; 컨테이너별 사설 `isaaclab/logs`
마운트로 acados codegen 충돌 방지):

```bash
#!/usr/bin/env bash
# e2_worker.sh <container-name> '<command chain>'
set -euo pipefail
NAME="$1"; shift
REPO=/home/bon-ubuntu/Documents/mjkim/UnderwaterScanningRobot   # 호스트가 바뀌면 수정
CTR=/workspace/UnderwaterScanningRobot
W=~/docker/e2w/$NAME
mkdir -p "$W"/{logs,kit,ov,pip,glcache,computecache,ovlogs,data,documents}
docker run --rm -i --name "$NAME" --user 0:0 \
  --runtime=nvidia --gpus all --network=host --shm-size=8g \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES -e TERM=xterm \
  -v ~/docker/acados:/opt/acados:rw -e ACADOS_SOURCE_DIR=/opt/acados \
  -e LD_LIBRARY_PATH=/opt/acados/lib -e PYTHONPATH=/opt/acados/pysite \
  -v "$REPO":"$CTR":rw -v "$W/logs":"$CTR/isaaclab/logs":rw \
  -v "$W/kit":/isaac-sim/kit/cache:rw -v "$W/ov":/root/.cache/ov:rw \
  -v "$W/pip":/root/.cache/pip:rw -v "$W/glcache":/root/.cache/nvidia/GLCache:rw \
  -v "$W/computecache":/root/.nv/ComputeCache:rw \
  -v "$W/ovlogs":/root/.nvidia-omniverse/logs:rw \
  -v "$W/data":/root/.local/share/ov/data:rw -v "$W/documents":/root/Documents:rw \
  -w "$CTR/isaaclab" --entrypoint bash underwater-scan:5.1 -lc "$*"
```

실행 (워커 3개 병렬, GPU 12 GB 기준 튜닝 없이 3개면 여유):

```bash
R='./isaaclab.sh -p ../marinelab/scripts/experiments/run_experiment.py'
E1='--config ../marinelab/scripts/experiments/configs/e1_nominal.yaml'
E2='--config ../marinelab/scripts/experiments/configs/e2_dr_sweep.yaml'
E3='--config ../marinelab/scripts/experiments/configs/e3_current.yaml'
# W1: E1 diff (s0..s4 체인)   W2: E2 diff (dr25/50/75 x s0..s2)   W3: E3 diff (step/sine x s0..s2)
./e2_worker.sh dw_e1 "$R diff --cond nominal --seed 0 $E1 && … --seed 4 $E1"
./e2_worker.sh dw_e2 "$R diff --cond dr25 --seed 0 $E2 && … dr75 --seed 2 $E2"
./e2_worker.sh dw_e3 "$R diff --cond step --seed 0 $E3 && … sine --seed 2 $E3"
```

- [ ] cron 등록 (예: 30분 간격): metrics 수 증가 확인, 정체+CPU 스핀 시 py-spy
      사이드카(`docker run --pid=container:<n> --cap-add SYS_PTRACE -v <pyspy>:/ps …`)
      진단 → kill → 남은 셀만 재실행. 전체 완료 시 집계·커밋 후 cron 자가 삭제.
- [ ] 예상 소요: 20셀 × ~5분 / 3워커 ≈ 40분
- [ ] 완료 후 소유권 정리: 컨테이너가 root로 쓰므로
      `docker run --rm --user 0:0 -v <repo>:/repo --entrypoint bash underwater-scan:5.1 -c 'chown -R 1000:1000 /repo/experimental_results'`

## Phase 6 — E2(b) 온라인 fine-tuning + F5

계획 §7(competitor_framework_plan.md) — **아직 미구현**:

- [ ] `marinelab/scripts/experiments/finetune_diff_wmpc.py` 신설: DiffWMPCLearner
      state_dict 로드 → learning ON으로 단기 계속 (optimizer state 필요 시 learner
      서브클래스로). 기존 train 스크립트의 eval-only `--ckpt` 의미는 불변.
- [ ] E2 각 조건에서 fine-tune 후 재평가 → `experimental_results/e2b/`
      (exp: e2b 설정 yaml 신설, e2와 동일 조건/시드)
- [ ] F5 생성: `plot_figures.py f5 experimental_results/e2 experimental_results/e2b --names zero-shot fine-tuned`

## Phase 7 — E4(a) ablation

계획 §7의 토글 3종 — 설정 yaml 미존재, 신설 필요 (`e4_ablation.yaml`, exp: e4_abl):

- [ ] preview on/off: 러너에서 frozen setpoint 생성 (지원 여부 확인 후 옵션 추가)
- [ ] `werr_ub` 500 vs 5000: WeightPolicy ctor 인자 — **inference에서 bounds만 바꾸면
      체크포인트와 불일치**하므로, 학습 변형이 타 머신에서 오는지 먼저 확인. 안 오면
      이 축은 "학습 필요"로 논문에서 제외/부록화 결정.
- [ ] saturation-skip on/off: 학습 쪽 토글 — 위와 동일 판단.

## Phase 8 — 집계·그림·문서 갱신

- [ ] `aggregate.py` e1/e2/e3 (+e2b) 재실행 — diff 행 추가된 Table 1/2/3
- [ ] figure 전체 재생성: F1(f1), F2, **F3 — `--logy` 필수** (PPO 스케일), F4, F5,
      F6(`--tuning experimental_results/tuning`), F7, **F8은 컨테이너에서** (호스트
      matplotlib Axes3D 불가), F9, F10
- [ ] `docs/validation_report.md`에 diff 행/패턴 추가 (C1: diff > bo/ssi 성립 여부),
      §3 매트릭스 갱신
- [ ] `docs/README.md` 체크리스트 갱신, 이 문서의 체크박스 갱신

## Phase 9 — 커밋/푸시

논리 단위: ① `feat(diff): port the externally-optimized WeightPolicy …`
② `results(e1,e2,e3): diff cells …` ③ `results(e2b)+figures: F5 …` ④ docs.
푸시 전 `git lfs pull` 상태와 실행권한(chmod +x) 확인. 커밋 서명 규약은 기존 로그 참조.

## 알려진 함정 (이번 세션에서 실측된 것)

1. 멀티셀 세션 무한루프 — 수정 커밋 `36cc3a8`에 원인·수정 설명. per-cell이 표준.
2. stdout 블록 버퍼링 — 진행 판정은 반드시 산출물 파일로.
3. acados codegen 경로 공유 충돌 — 워커별 사설 `isaaclab/logs` 마운트.
4. 결과 파일 root 소유 — aggregate 전 chown.
5. 솔버 포화 시 감도 0 — 학습이 포화 스텝을 스킵하는 이유; ablation 해석 시 유의.
6. GPU 예산: 컨테이너당 ~2.2–2.8 GB, 12 GB 호스트에서 동시 4개가 안전 상한.
