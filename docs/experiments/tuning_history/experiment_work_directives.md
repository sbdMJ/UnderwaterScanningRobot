# 실험 진행 작업 지침 (2026-08-08 확정)

> `docs/experiments/experiments_plan.md`(설계)·`docs/experiments/tuning_history/bo_tuning_history.md`(BO 여정)의 하위 실행 지침.
> 실험을 수행하는 세션은 이 문서의 분기 규칙과 red-flag 규칙을 따른다.

## 1. 현재 파이프라인 분기 (사전 등록)

```
BO attempt 3 (3-튜닝시드 평균, 진행 중)
 ├─ 게이트 통과 (E2에서 BO 평균 ≤ nominal 평균, 조건별)
 │   → BO 채택. SSI-MPC를 attempt-3 가중치로 재기반(re-base) + §6 예산으로
 │     ssi_lr/kernel_std 재튜닝 → E1 ssi 재실행 → E2 ssi 실행
 └─ 게이트 실패
     → BO 포기 (결과는 논문에 그대로 보고). SSI-WMPC 실험을 우선 진행:
       SSI의 E2가 nominal보다 여전히 나쁜지 확인이 최우선 질문.
       단, 아래 §3의 오염 규칙 때문에 attempt-2 가중치 승계 상태로 돌리는 것은
       무의미 — nominal(Fixed-W DEFAULT_WERR) 승계로 전환한 뒤 ssi 하이퍼파라미터를
       재튜닝하고 나서 E2 ssi를 실행한다.
```

## 2. Red-flag 규칙: "nominal이 적응형을 일관되게 이기면 설명이 안 된다"

아무 적응도 하지 않는 Fixed-W nominal이 환경 적응력을 갖춘 방법(BO-WMPC, SSI-WMPC,
Diff-WMPC)보다 **일관되게** 좋게 나오는 결과는 방법론적으로 설명이 되지 않는 상태다.
이 패턴이 보이면:

- **무의미한 재실행/재튜닝 반복 금지.** 연산을 더 태우기 전에 원인 분석을 완수한다.
- 원인 분석 체크리스트 (순서대로):
  1. **가중치/설정 오염**: 적응형 방법이 승계·로드하는 파라미터의 출처가 건강한가?
     (§3의 사례 — SSI가 과적합 BO 가중치를 승계해 함께 침몰)
  2. **튜닝 프로토콜 결함**: 목적함수가 단일 시나리오인가? 튜닝 시드와 평가 시드가
     겹치거나, 반대로 프로토콜 길이/task가 다른가? (`bo_tuning_history.md` attempt 1-2)
  3. **적응이 실제로 작동하는지 계측**: aux 채널(가중치 시계열, SysID 잔차)을 npz에서
     확인 — 적응량이 0에 가까우면 "적응형이 진 것"이 아니라 "적응이 꺼져 있던 것".
  4. **스코어링 정합성**: objective가 방법 간 비교 가능한 방식으로 집계되는가?
     (예: PPO는 8-env 합산이라 1-env MPC와 절대값 비교 불가 — 방법 내 조건 간만 유효)
  5. **솔버 상태**: fail_frac/saturated_frac (metrics의 controller_cost) — 포화·실패가
     높으면 가중치가 아니라 OCP 정식화/제약이 원인일 수 있다.
- 분석 결과는 `bo_tuning_history.md`(BO) 또는 본 문서(그 외)에 증거와 함께 기록한다.

## 3. 확정된 오염 사례 (2026-08-08) — SSI 재실행 전 필수 숙지

E1에서 SSI가 nominal에 진 것은 **적응의 실패가 아니었다**:

| seed | 0 | 1 | 2 | 3 | 4 |
|:--|--:|--:|--:|--:|--:|
| BO (attempt 2) | 11.5 | 62,682 | 14,121 | 21,588 | 72.8 |
| SSI (BO 가중치 승계) | 11.5 | 62,682 | 14,120 | 21,573 | 72.8 |
| nominal | 236 | 1,246 | 866 | 2,185 | 310 |

SSI의 시드별 점수가 BO-static과 사실상 동일 = 성능이 **승계된 과적합 비용 가중치에
완전히 지배**되었고, 온라인 SysID 적응은 이 격차를 만회하지 못했다(적응은 모델을
고치지, 잘못된 비용 가중치를 고치지 않는다). 따라서:

- **SSI E2를 attempt-2 가중치로 실행하는 것은 금지** — 결과가 사전에 무의미함이
  확정되어 있다 (무의미한 연산 반복 금지 규칙).
- SSI 재평가는 반드시 건강한 가중치(게이트 통과 시 attempt-3, 실패 시 DEFAULT_WERR)
  로 재기반 + `tune_ssi_mpc.yaml` 재튜닝(다중 시드 프로토콜, §6 동일 예산) 후 진행.
- SSI 튜닝의 모든 attempt는 `docs/experiments/tuning_history/ssi_tuning_history.md`에 기록한다
  (BO의 `bo_tuning_history.md`와 동일 형식 — attempt별 프로토콜·예산·판정·무효 사유).
- E1 ssi 결과 5셀도 같은 이유로 무효 — 재기반 후 재실행 대상.

## 4. 비교 기준 (요약; 상세는 bo_tuning_history.md §0)

스팟체크·결과 분석은 부모 논문(Diff-WMPC, RA-L 2026, Jahncke et al.)의 MOBO-WMPC
패턴 대비로 판정한다: 정적 튜닝 방법은 (1) 공칭에서 Fixed-W 이상, (2) 섭동에서 완만
열화, (3) 적응형에게만 열세. Fixed-W nominal이 "구조적 한계 vs 튜닝 결함"의 경계선.

## 5. 기타 실행 규칙 (이 세션에서 확립)

- diff(Diff-WMPC)는 다른 머신에서 최적화 진행 중 — **모델 확정 통보 전 실행 금지**
  (2026-08-08 지시). E1 다섯 번째 방법·E2 diff는 그 이후.
- 30분 이상 작업은 자가치유 cron 등록, 튜닝은 중간 스팟체크 의무 (CLAUDE.md 참조).
- 멀티셀 실행은 셀당 별도 프로세스가 기본 (Isaac 타임라인 정지 버그 회피 —
  `run_experiment.py`의 수정 이력 참조).
