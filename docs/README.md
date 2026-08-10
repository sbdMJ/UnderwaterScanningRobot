# docs/ 안내 — 읽는 순서와 전체 진행 현황

> 처음 이 프로젝트를 받은 사람은 이 문서부터. 마지막 갱신: 2026-08-09 (커밋 기준
> `672f375` 이후). 실험 프레임워크 산출물은 전부 `experimental_results/`에 있고,
> 실행 방법은 `marinelab/scripts/experiments/README.md`가 정본이다.

## 1. 읽는 순서

| 순서 | 문서 | 내용 |
|---|---|---|
| ① | `docs/README.md` (이 문서) | 전체 현황·체크리스트 |
| ② | `docs/experiments/experiments_plan.md` | ICRA 실험 설계 최종안 (E1–E5, 주장↔실험 매핑) |
| ③ | `docs/experiments/competitor_framework_plan.md` | 프레임워크 구현 설계 (§6 튜닝 프로토콜, §7 env 확장) |
| ④ | `docs/experiments/tuning_history/experiment_work_directives.md` | **실행 지침** — 분기 규칙, red-flag 규칙, 오염 사례 |
| ⑤ | `docs/experiments/tuning_history/bo_tuning_history.md` / `docs/experiments/tuning_history/ssi_tuning_history.md` | 튜닝 여정 (논문 실험 섹션 소스) |
| ⑥ | `docs/experiments/validation_report.md` | 결과 타당성 검증 (문헌 패턴 P1–P6 대조) |
| ⑦ | `docs/advanced_experiments_todo.md` | **다음 작업** — 후속 실험 계획 (조류 fine-tune, 속도 축, EKF 게이트) |
| ⑧ | `docs/research/domain_comparison.md` / `docs/research/diffwmpc_domain_adaptations.md` | 부모 논문 대비 도메인·방법 대조 + 향후 concern (C-1~C-9) |
| 참조 | `marinelab/scripts/experiments/README.md`, `docker/README.md`, `CLAUDE.md` | 러너 사용법 / 호스트 런타임 / 세션 규칙 |

## 2. 완료된 작업 (experiments_plan.md · work_directives 대응 체크리스트)

### 인프라·프레임워크 (P0–P2 로드맵)
- [x] P0 controller 계층 (`control/`: types·base·estimator + ①③⑤ 어댑터) + 네이티브 테스트
- [x] P1 러너·집계 (`run_experiment.py`, `aggregate.py`, E1 회귀 검증)
- [x] P2 공용 튜닝 파이프라인 (`tune.py`) + SSI-MPC GPL 격리 이식
- [x] 이 호스트(bon-ubuntu) 재현 환경: `underwater-scan:5.1` 이미지, acados v0.5.3 마운트, git-lfs
- [x] 러너 멀티셀 무한루프 수정 (`36cc3a8`), 다중 시드 튜닝 (`5374556`), F3 log-scale
- [x] 실행 규칙 확립: per-cell 프로세스, cron 모니터링 의무, 중간 스팟체크 의무 (CLAUDE.md)

### 튜닝 (§6)
- [x] BO-static: attempt 1–2 (단일 시드, 과적합 — 무효 판정·아카이브)
- [x] BO-static: **attempt 3 채택** (3-시드 평균, trial 87, E2 게이트 전 조건 통과 +18~20%)
- [x] SSI-MPC: attempt 1 (오염 가중치 승계 — 무효 판정·아카이브)
- [x] SSI-MPC: **attempt 2 채택** (재기반+다중 시드, trial 87) — e1/e2/e3 yaml 반영
- [x] 튜닝 비용 기록 (`budget.json` × 4, 방법당 누적 ~1.9M 샘플 = 원논문 ~4.1배)

### 실험 (방법 × 실험 매트릭스)

| | nominal | BO | SSI | PPO | diff |
|---|---|---|---|---|---|
| E1 (5 seeds) | [x] | [x] | [x] | [x] | [x] V8a |
| E2 (3 DR × 3 seeds) | [x] | [x] | [x] | [x] (s1 충돌 포함) | [x] V8a (전 조건 1위) |
| E2(b) fine-tune | — | — | — | — | [x] 3-arm 완료: zs≈ft≈from-scratch — zero-shot이 이미 environment-specific 수준 |
| E3 (2 조류 × 3 seeds) | [x] | (설계상 제외) | [x] | (설계상 제외) | [x] V8a |
| E4(a) ablation | — | — | — | — | [x] Table 2 (preview +41%, 나머지 무차별) |
| E4(b) 튜닝 비용 | [x] | [x] | [x] | W&B 기록 | [ ] 학습비용 수령 |
| E4 부록 민감도 (11×5) | — | — | [x] | — | — |
| 속도 축 e2s (2조건×3시드) | [x] 2.2× 악화 | [x] 평탄 | [x] 평탄 | (제외: 보상 하드코딩) | [x] 전 속도 선두 |
| 조류 3-arm e3b (4조건) | — | — | — | — | [x] zs≈ft, sc 열세 |
| E4(c) 추론 벤치 (Jetson) | [ ] | [ ] | [ ] | [ ] | [ ] |
| E5 하드웨어 | [ ] | [ ] | [ ] | [ ] | [ ] |

### 그림·표
- [x] Table 1/2/3 소스 (`tables/table.{csv,tex}` — e1/e2/e3/e4_ssi_sens)
- [x] F1, F2, F3(log), F4, F6, F7, F8, F9, F10
- [x] F5 (zero-shot vs fine-tuned)
- [x] 결과 타당성 검증 보고서 (P1–P6 전부 충족)

## 3. 남은 작업 (우선순위 순)

1. **diff-WMPC 이식 + E1/E2/E3 diff (20셀)** — 타 머신 모델 확정 대기.
   diff 이식 절차(port_todo)는 캠페인 완료로 폐기 — 후속 계획은 `docs/advanced_experiments_todo.md`.
2. **E2(b) 온라인 fine-tune** (`finetune_diff_wmpc.py` 신설) → **F5**
3. **E4(a) ablation** (`e4_ablation.yaml` 신설; 학습 변형 필요 축은 타 머신과 협의)
4. E4(c) Jetson 추론 벤치 (`bench_inference.py`, isaaclab 무의존 — Jetson에서 실행)
5. (선택) E4 민감도를 E3 조류 조건에서 반복 — validation_report §2-E4의 권고
6. (별도 축) 상태소스 `state: ekf` 실험 — 현재 전 실험이 GT 상태(Phase 2a) 기준
7. E5 하드웨어 이식 — `competitor_framework_plan.md` §8 체크리스트

## 4. 판정 기준치 요약 (다음 세션이 바로 쓸 수 있게)

objective ↓, 커밋 `dabe7ad` 기준:

| | E1 평균 | E2 dr25/50/75 평균 | E3 step/sine 평균 |
|---|--:|--:|--:|
| nominal | 969 | 4,234 / 4,246 / 4,291 | 2,107 / 1,566 |
| BO | 679 | 3,476 / 3,479 / 3,448 | — |
| SSI | 697 | 3,400 / 3,464 / 3,388 | 1,173 / 956 |
| PPO (8-env 합산, 별도 스케일) | 297,303 | s1 전 조건 충돌 | — |

diff는 부모 논문 주장 C1에 따라 **BO/SSI 이하(우위)**가 나와야 하며, nominal보다
나쁘면 red-flag 규칙(§work_directives 2) 발동 — 재실행 반복 금지, 원인 분석 우선.
