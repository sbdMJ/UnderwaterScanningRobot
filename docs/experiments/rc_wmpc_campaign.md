# RC-WMPC 캠페인 기록 (2026-08-24 ~ 08-25)

브랜치 `feature/rc-wmpc` (441e41e → 7aaae9f). 제안서: claude.ai artifact c98a58f8.
이 문서가 집필의 단일 소스다 — 모든 수치는 `experimental_results/{e1_rc, e3_rc,
rc_authprobe, rc_authprobe_v1, rc_authsweep, rc_auth40_noctx}/metrics/`에서 재계산 가능.

## 한 줄 결론

> **입증된 것은 잔차-다양 joint training이다(H2): 학습된 가중치 정책의 OOD 붕괴를 막고,
> 권한 희소 영역에서 실무적 조합(naive stacking) 대비 우위를 만든다. ctx 조건화(H1)는
> 방향성 있는 시사 증거(n=10, p=0.065)에 그친다 — 사전 등록 기준으로 미확정.**

## 방법 및 체크포인트 명명

| 이름 | 정책 학습 | ctx | 배포 시 SSI |
|---|---|---|---|
| diff | 공칭 GT, 20k (`checkpoints/diff_wmpc/policy_final.pt`) | — | — |
| naive1 (`rc_naive`) | 위와 동일 ckpt | — | ✓ (ablation A1) |
| rc V0 | +dr_fluid 0.75, current 0.15, SSI-in-loop, 20k (`rc_wmpc/`) | ✓ | ✓ |
| rc V1 (`rc`) | V0 + authority mix U(3,12)N + degenerate skip, 30k (`rc_wmpc_v1/`) | ✓ | ✓ |
| naive2 | V1 레시피 − SSI-in-loop(−ctx), 30k (`dw_v1_noctx/`) | — | ✓ (인과 분리용) |

## 결과 연대기 (판정 게이트 순)

### 1. E1 공칭 (5 seeds, GT) — 예측 P1 확인
rc 12.15±3.13 ≈ naive1 12.13 ≈ diff 12.12 cm. 결합은 잔차 없는 영역에서 해가 없다.

### 2. E3 조류 (GT, 40 N) — H3 게이트 발동
rc ≈ naive1 ≈ diff (전체 및 사전 등록 과도기 창 [90,105)s: 10.13/10.14/10.18 cm).
**단 메커니즘은 발화**: 스텝 순간 rc 가중치 soften→re-stiffen (radial −9.5%, z −22.4%
→ 회복; naive1은 −0.5/−2.7%). 이득이 없는 이유 = 평평한 가중치 지형(서로 다른 절대
가중치로 동일 성능) + 0.15 m/s는 추력 여유 내부.

### 3. rc_authprobe (EKF + 실측 센서 + 3.68 N 실측 추력) — 갭 발견
- 유효성: 포화 71–100% (E3 ~1%).
- 정적 트림 조건(lowthrust/trim): rc ≈ naive1 (±1 cm). 부수 발견: diff 계열 전체가
  기록된 e5 대조군을 압도 (2.4–14.8 vs nominal 24.8±25.7, ssi 161±327 cm).
- **lowthrust_current: rc V0 − naive1 = −23.1±25.2 cm, 5/5승 (부호검정 p=0.031).**

### 4. V1 학습 + Phase A — 재현 확인, in-regime 재학습 무효과
- V1 학습 건강: 30k, sat-skip 0.16% (V0 7.4%), 업데이트 2915, ctx 게이트 0.94 PASS.
- **rc V1 − naive1 = −24.1, 4/5 (재현: 독립 학습 2회).** rc V1 − rc V0 = −1.1 (3/5, 무효과).

### 5. 권한 스윕 (figure: `rc_authsweep/figures/fig_gap_vs_authority.*`)
| 권한 [N] | rc | naive1 | 쌍갭 | 승 | cycles rc/naive1 |
|---|---|---|---|---|---|
| 3.68 | 165.3 | 189.4 | −24.1 | 4/5 | 0 / 0 |
| 8 | 61.6 | 79.3 | −17.7 | 2/3 | **1.00 / 0.67** |
| 16 | 18.9 | 19.0 | −0.1 | 2/3 | 2.0 / 2.0 |
| 40 (EKF) | 37.9 | 47.9 | −10.0 | 3/5 (동률 2) | — |

### 6. 40 N 재출현의 정체 — naive1의 OOD 가중치 붕괴 (궤적 증거)
naive1 s1, t=110+: w_radial 4896→13→10, w_z→0 고착, wall_err 1.50 m 표류
(2026-07-31 문서화된 붕괴 병리의 OOD 재발). rc는 동일 조건에서 안정(4996/2600/4500).
갭은 붕괴 seed에서만 나고 정상 seed는 소수점까지 동률.

### 7. 인과 분리 (naive2) — 분포가 인과
40 N: naive2 무붕괴 5/5 (min w_radial 132), **최강 13.1 cm** — rc(37.9)·naive1(47.9)보다
좋음. rc의 s2 초기획득 실패(139.5 vs naive2 18.3)에서 ctx cold-start 부작용 후보(n=1).
→ "perception-aware ctx 이득" 가설 기각. 40 N 붕괴 방지 = **학습 분포 인과**.

### 8. 헤드라인 deconfound (3.68 N, n=10) — 사전 등록 최종 판정
rc V1 − naive2: 개별 {+4.8, −20.9, −60.9, +13.5, −19.4, −3.8, −2.0, +4.8, −18.9, −18.6},
**평균 −12.2±21.1, 7/10승, Wilcoxon 편측 p=0.065, 부호검정 p=0.17 → 미확정(시사적).**
패턴: 어려운 seed에서 크게 이기고(−19~−61) 쉬운/획득 seed에서 소폭 짐(+5/+13).
n=10에서 정지 — 결과를 본 뒤의 seed 추가(optional stopping)는 금지.

### 9. E2 rc 열 (zero-shot DR 스윕, 2026-08-25) — 본 비교표 완결

`e2_rc.yaml` (e2와 동일 프로토콜: Eval task, dr_fluid ±25/50/75%, GT, 3 seeds).
18/18 셀 완주, 전 방법 전 조건 2사이클 완주(붕괴 없음).

| wall_dist_err [cm] | nominal | bo | ssi | diff | naive1 | rc |
|---|---|---|---|---|---|---|
| dr25 | 13.97 | 16.40 | 16.44 | 15.70 | 15.92 | **14.55** |
| dr50 | 13.63 | 16.26 | 17.78 | 15.42 | 15.52 | **14.21** |
| dr75 | 13.32 | 16.15 | 16.77 | 15.16 | 15.16 | **13.93** |

- **score.objective는 rc가 전 강도에서 전체 1위** (2699/2628/2588 vs nominal
  4234/4246/4291 — nominal은 wall_dist만 좋고 tilt 14.2° vs diff 계열 11.9°).
- 쌍갭 rc − naive1 = −1.36/−1.30/−1.23 cm (3 강도 모두 음수, seed2가 −3.4~−3.8로
  견인, seed0/1은 ±0.5 내 — "어려운 seed에서 이긴다" 패턴의 GT 재현).
- crab_deg도 rc가 일관 우위 (dr75: 1.67° vs naive1 2.43°).
- dr75에서 saturated_frac ~0.30 — DR이 포화를 올릴수록 rc 갭 방향 유지(권한 스윕
  서사와 정합). 열화 기울기는 전 방법 평평(25→75% 스윕에 대해 zero-shot 강건, C3).
- figure: `e2_rc/figures/fig_f3_with_rc.*` (+objective 버전), 조인 뷰는 symlink 병합.

### 10. E4 rc 열 (비용, 2026-08-25)

- **(b) 학습 비용**: `tuning/rc_wmpc_training/budget.json` — V0 20k 스텝 **~611 s**
  (ckpt mtime 도출, 앱 기동 ~2–3분 제외 명기; V1 30k ~985 s, naive2 30k ~981 s).
  diff 60k ~3000 s의 1/5. `collect_budgets`/f6가 자동 수집 (`rc_wmpc_training`→`rc`).
- **(c) 추론 비용** (e1 nominal, 데스크톱, 5 seeds): rc solve **8.34±1.05 ms**
  (naive1 8.40, ssi 8.03, diff 9.16, nominal 8.82) — ctx forward + RFF 업데이트는
  acados solve에 묻힘, p95 13.5 ms로 50 Hz 예산(20 ms) 내. Jetson 실측은 diff 계열
  기존 bench(`e4_inference/`)에서 외삽 가능하나 rc 자체 실측은 미실시(잔여 항목).
- figure: `e1_rc/figures/fig_f6_with_rc.*` (offline/inference 이중 패널).
- E4(a) ablation의 rc 대응물은 별도 실행 불필요 — naive1(A1)·naive2가 이 캠페인의
  ablation 축이며 §5–8에 기록됨.

인프라 노트: `aggregate.collect()`가 파일명 대신 json 내부의 method/cond/seed를
우선하도록 수정 (`rc_naive` 같은 밑줄 방법명이 regex에서 오파싱되던 문제).
METHOD_{ORDER,LABELS,COLORS,MARKERS}에 rc("RC-WMPC (ours)", #A2142F, P)·
rc_naive("SSI+Diff-WMPC (naive)", #4DBEEE, X) 등록.

## 방법론적 부산물

- **완화 배리어 sensitivity: 구조적 실패** (`mpc_controller.sens_relax`, 봉인).
  interior 충실도 1e-7이나 deep-saturation에서 cos ≤ 0.05; 넓은 σ는 interior 오염
  (relFD 0.65+). 원인: 포화 최적점의 w→u₀ 맵이 계단 (w_z ×0.8에 u₀ 113 N 점프, 국소
  미분 ~0). 프로브: `isaaclab/logs/_probe_relax_sens.py`.
- **degenerate-skip** (`DiffWMPCLearner saturation_skip="degenerate"`): 부분 포화
  스텝의 sensitivity는 비포화 부분공간에서 정확 — skip 7.4%→0.16%로 신호 복구.
  단 in-regime 재학습의 평가 이득은 없음(V1−V0 −1.1). 주장은 "학습 가능화"까지만.
- **런타임 권한 바운드** (`WallScanMPC.set_input_bounds`) + trainer `--authority_mix`.
- **ctx 인프라** (`policy_features(ctx=)`, `rc_context`, `WeightPolicy(ctx_dim=)`,
  `RCWMPCController`): 체크포인트 왕복 호환, ctx-less ckpt는 자동으로 naive stacking
  (=A1)으로 강등. 정적 ctx 민감도 프로브(`_probe_rc_ctx.py`)는 폐루프 이득을 보장하지
  않음이 확인됨 (0.94~2.1 PASS여도 성능 무차이 가능).

## 실무 함정 (재발 방지)

- 병렬 컨테이너가 공유 codegen 디렉토리를 동시 재생성 → 솔버 로드 시 JSONDecodeError.
  스트림 시차 시작(sleep 120)으로 회피; 근본 해결은 method별 export dir.
- config의 `seeds:` 목록 밖의 `--seed`는 "no cells match"로 조용히 스킵 — 확장 시 목록 먼저.
- 컨테이너(root)가 만든 결과 디렉토리는 호스트에서 쓰기 불가 — `chown -R 1000:1000`.

## 논문 골격 (H2-주기여 재구성)

- **헤드라인**: 잔차-다양 joint training이 학습된 MPC 가중치 정책의 OOD 붕괴를 막고,
  권한 희소 영역에서 실무적 조합 대비 우위를 만든다.
- **기여**: ① 붕괴 현상의 실증 + 인과 분리 (naive1/naive2/rc 3-way, 40 N EKF)
  ② 권한 스윕 — 적응이 값을 하는 영역의 지도 (대표 figure)
  ③ degenerate-skip — 포화 영역 학습 가능화 (+완화 배리어의 구조적 실패라는 부정 결과)
  ④ ctx 조건화 — 시사적 이득(p=0.065)과 cold-start 부작용, future work로 정직 배치.
- **하드웨어(E5)**: 기존 계획 유지 — 3.68 N이 실측치이므로 sim 스윕의 저권한 끝단이
  곧 실기 조건. rc vs naive2 비교를 실기에서 1회 재현하면 완결.
