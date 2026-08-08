# SSI-MPC 튜닝 여정 기록 (논문 실험 섹션용)

> `docs/bo_tuning_history.md`와 동일한 목적·형식: SSI-MPC(④, UM-iRaL/SSI-MPC 포팅)에
> 투입된 튜닝 노력 전체를 기록해 "베이스라인을 덜 튜닝했다"는 리뷰 공격을 봉쇄한다.
> 실행 규칙은 `docs/experiment_work_directives.md`, 파이프라인은 §6 공용 `tune.py`.

## Attempt 1 — 단일 시드 + 오염된 가중치 승계 (무효 판정, 2026-08-08)

- 프로토콜: TPE 103 trials, 탐색 공간 {lr, kernel_std}, 4,500스텝 탐색 × **env seed 0
  단일** + 상위 k 재채점 9,000스텝. 총 477k env steps / 6,479 s.
  채택: trial 22 — lr 4.179e-4, kernel_std 4.807 (full 재채점 objective 6.74).
- 산출물: `experimental_results/tuning/ssi_mpc/` (study.db / trials.csv /
  best_params.json / budget.json).
- **이중 결함으로 무효**:
  1. BO attempt-2의 **과적합 비용 가중치를 승계**한 상태에서 튜닝됨 (§6 출발선 통일
     규칙의 부작용). E1에서 SSI 시드별 점수가 BO-static과 사실상 동일(11.5 / 62,682 /
     14,120 / 21,573 / 72.8)했던 것이 증거 — 성능이 승계 가중치에 완전히 지배되어,
     이 위에서 고른 lr/kernel_std는 신뢰할 수 없다.
  2. BO attempt 1-2와 동일한 **단일 시드(0) 목적함수** — 같은 과적합 위험.

## Attempt 2 — 계획 (BO 게이트 판정 후 실행)

- **선행 조건**: 건강한 비용 가중치 확정 — BO attempt-3 게이트 통과 시 그 가중치,
  실패 시 Fixed-W DEFAULT_WERR (근거: `experiment_work_directives.md` §1, §3).
- **프로토콜**: BO attempt-3와 동일한 다중 시드 개선 적용 — trial당 튜닝 시드
  3개(100/101/102, 평가 시드와 분리) 평균, 재채점 top-k × 3시드 × 9,000스텝.
  `tune.py`는 이미 `seeds:` 리스트를 지원(2026-08-08 수정)하므로
  `tune_ssi_mpc.yaml`에 `seeds: [100, 101, 102]` + `inherit_weights` 갱신만 필요.
- **예산**: attempt 1과 동일 trial 수(~100) 유지 — "동일 예산" 주장 보존. 누적 노력은
  attempt별 budget.json 합산으로 E4(b)에 보고.
- 완료 후 이 문서에 기록할 것: 채택 파라미터, 재채점 objective(시드별), E1/E2 재실행
  결과와 nominal 대비 판정 (red-flag 규칙 §2 기준).
- **중간 스팟체크 (62/100 trial, 2026-08-08)**: 당시 최고 후보 trial 38
  (lr 0.1009, kernel_std 1.397 — attempt 1 채택값 lr 4.2e-4와 240배 차이, 오염 기반
  선택의 무의미함을 방증; 튜닝시드 평균 286.1로 BO-static 292.1보다 우수).
  held-out dr50 × seeds 0–2 (`configs/e2_spotcheck_ssi.yaml` → `e2_interim/`):
  1,365 / 3,402 / 5,324 → 평균 **3,364** — 전 시드에서 BO attempt-3(평균 3,479)
  이상, nominal(4,246) 대비 +21%, 충돌 0. **적응형 ≥ 정적 ≥ 무튜닝 서열 복원 확인.**

### Attempt 2 결과 (2026-08-08): 채택

- 탐색 100 trials 완료, 상위 3개(trial 87/38/75) full 재채점 289.3 / 291.8 / 289.6 —
  **채택: trial 87 (lr 0.14733, kernel_std 0.18398, full 3-시드 평균 289.3)**.
  BO-static attempt-3(292.1)보다 우수 — 적응의 기여가 튜닝 시드에서도 확인.
- e1_nominal / e2_dr_sweep / e3_current yaml의 `ssi_lr`/`ssi_kernel_std` 갱신 완료.
- **E1/E2 재실행 최종 판정 (2026-08-08)**:

  | | E1 평균 (seeds 0–4) | E2 dr25 | E2 dr50 | E2 dr75 |
  |:--|--:|--:|--:|--:|
  | nominal | 969 | 4,234 | 4,246 | 4,291 |
  | BO (attempt 3) | **679** | 3,476 | 3,479 | 3,448 |
  | SSI (attempt 2) | 697 | **3,400** | **3,464** | **3,388** |

  공칭(E1)에서는 SSI ≈ BO (적응할 모델 불일치가 없으니 당연) · 섭동(E2)에서는
  **SSI가 전 조건 1위** — 적응의 기여가 섭동 강도에서만 나타나는 정확히 기대된
  패턴. red-flag 상태(무적응 > 적응형) 완전 해소. 충돌 0, 전 셀 2.0 사이클.

## 튜닝 노력 누계 (E4(b) 소스)

| attempt | trials | env steps | wall-clock | 상태 |
|:--|--:|--:|--:|:--|
| 1 (단일 시드, 오염 가중치) | 103 | 477,000 | 6,479 s | 무효 — 기록용 |
| 2 (다중 시드, 재기반) | 103 | 1,431,000 | 22,093 s | **채택 (trial 87)** |
| **누계** | 206 | **1,908,000** | 28,572 s | 원논문 460k의 ~4.1배 |
