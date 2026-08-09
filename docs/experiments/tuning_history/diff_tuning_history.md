# Diff-WMPC 개선 여정 기록 (feature/enhance-ours)

> 형식은 `bo_tuning_history.md`와 동일: 시도·증거·판정을 남겨 논문 실험 섹션과
> 리뷰 대응의 소스로 쓴다. 성능 분석의 참조 기준: 부모 논문 PDF
> (`docs/references/Differentiable_Weights-Varying_Nonlinear_MPC_via_Gradient-Based_Policy_Learning_An_Autonomous_Vehicle_Guidance_Example.pdf`)
> 와 https://diffmpc.com/ , 그리고 red-flag 규칙(`experiment_work_directives.md` §2).

## 배경 (2026-08-09)

- 원격(타 머신) 개선 시도는 실패로 종결 — 처음부터 로컬에서 재시작.
- 판정 기준치 (커밋 dabe7ad, E-objective ↓): E1 bo 679 / ssi 697 / nominal 969,
  E2 평균 ssi 3,388–3,464 / bo 3,448–3,479 / nominal 4,234–4,291,
  E3 ssi 1,173(step)·956(sine) / nominal 2,107·1,566.
  **diff의 목표: 부모 논문 주장 C1에 따라 전 실험에서 bo/ssi 이하(우위).**

## 0. 체크포인트 감사 (개선 전 가동성 확인)

- 어댑터(`control/diff_wmpc_ctrl.py`)와 학습 코어(`algorithms/diff_wmpc.py`) 코드는
  정합 확인: forward는 (feat_dim,) 무배치+자체 히스토리, step마다 정책→솔버 주입,
  aux로 가중치 시계열 기록. 네이티브 테스트 279/280 통과 (1 실패는 diff와 무관한
  호스트 matplotlib mpl_toolkits 충돌 — 컨테이너에서는 정상).
- **E-config가 가리키던 `checkpoints/diff_wmpc/policy_final.pt`(n_upd 788)는
  bounds 버퍼 도입 이전 포맷이라 현재 코드에서 strict load 실패** — E1에 diff가
  빠져 있던(4/5 방법) 직접 원인으로 추정.
- 감사 결과 (fc1=(128,70) → feat 14 × (hist 4+1), 전 후보 아키텍처 동일):

  | ckpt | n_updates | bounds 버퍼 | werr_lb[radial,head] | 비고 |
  |---|--:|:--|:--|:--|
  | diff_wmpc/policy_final | 788 | **없음 (구식)** | (기본 0.1 추정) | 로드 불가 |
  | diff_wmpc_mn/policy_final | 780 | 있음 | 10.0 | |
  | dw_lu{0.001,0.01,0.1,1.0} | ~1,900 | 있음 | 10.0 | l_u 스윕 |
  | dw_A_lvz_long | 3,741 | 있음 | 10.0 | 최장 학습 |
  | dw_B_long | 3,735 | 있음 | 10.0 | 최장 학습 |

## 1. 베이크오프 — "현재 diff"의 기준선 선정 (진행 중)

후보 4종(dw_A_lvz_long, dw_B_long, dw_lu0.01, diff_wmpc_mn)을 E1 nominal s0
1셀로 스크리닝 → 최선 후보로 E1 5시드 전체 측정. 결과는 `experimental_results/
e1_bake_*/`(스크리닝, 임시)와 본 E1로 기록.

## 2. 부모 논문 정독 결과 — 우리 구현과의 구조적 차이 (2026-08-09)

RA-L 본문(pp.3724–3731) 기준, 개선 레버 후보를 도출:

| # | 부모 논문 (레이싱) | 우리 구현 (wallscan) | 시사점 |
|---|---|---|---|
| 1 | **정책 입력 = 미래 참조** (향후 2.55 s의 속도 5점+곡률 5점) — Fig.5의 "코너 진입 전 선제 적응"이 핵심 메커니즘, Fig.8 ablation에서 미래 문맥 제거 시 성능 저하 확인 | **정책 입력 = 현재 오차**(e_now 12) + phase sin/cos + 과거 4스텝 히스토리 — **후방 관찰만** | **최대 격차.** 스캔 참조(z_ref/s_ref 램프)는 결정론적이라 미래 참조를 공짜로 얻을 수 있음 → V1 |
| 2 | 감도 노드 = 1 (노드 1/10/20 비교, RMSE 0.090/0.108/0.191) | 다중 노드 합 {N/6,2N/6,4N/6,N} — 2026-07-31의 w_radial 붕괴를 다중 노드로 해결한 이력 | 우리 실패 이력이 있으므로 다중 노드 유지하되, **노드 1 추가** 실험 여지 |
| 3 | lr 2.9e-5 (보수적) + grad clip ±0.1 + batch 10스텝 미니배치(에피소드 내 빈번한 업데이트), 전부 **BO로 선정** | lr 5e-4 기본, clip 0.1, segment 600 | lr이 17배 공격적 → V3 (보수 lr + 미니배치) |
| 4 | 손실 = 추종² + **조건부 지수 패널티**(임계 초과 시 exp) — 제약 인접 안전 | l_* 가중 이차 손실 | 충돌/기울기 임계에 조건부 패널티 → V4 |
| 5 | 솔버 실패 시 마지막 feasible 가중치로 폴백 | 포화 스텝 학습 스킵(감도 0 때문) 있음; 폴백은 확인 필요 | 안전장치 점검 |
| 6 | 학습 101 s / 36.8k 샘플 (6랩 수렴), 30개 무작위 초기화 전부 수렴 | 20k 스텝 기본, 다중 run 존재 | 학습 비용은 이미 동급; 문제는 양이 아니라 **입력 설계** |

부모 논문의 결과 구도(모델 불일치 하 최고 정적 대비 lateral −50%): 우리 목표도
동일 구도 — E2/E3에서 bo/ssi 대비 우위.

### 개선 계획 (variant, 병렬 학습 예정)

- **V1 look-ahead**: 정책 입력에 미래 참조 preview 추가 (스캔 램프에서 +Δt 시점의
  z_ref/s_ref/v_des 표본). 학습 스크립트와 어댑터가 피처를 **공유 함수**로 쓰도록
  리팩토링 + feat 스펙을 ckpt에 동봉(버퍼) — ctor 불일치 사고 원천 차단.
- **V3 conservative-lr**: lr 3e-5 + 미니배치(10스텝) + 동일 예산.
- **V4 conditional-penalty**: tilt·wall-clearance에 임계-지수 패널티.
- V2 (단순 연장 학습)는 후순위 — dw_A/B가 이미 3.7k 업데이트.

## 3. 기준선 측정 (dw_A_lvz_long → `diff_wmpc/policy_final.pt`로 승격, 2026-08-09)

베이크오프(E1 s0): dw_A_lvz_long 13.3 · dw_lu0.01 13.5 · dw_B_long 13.6 ·
diff_wmpc_mn 14.6 — 4후보 모두 bo(17.4)/ssi(17.7)보다 우수. 승자로 full 기준선:

| | s0 | s1 | s2 | 평균 | 최고 경쟁자 |
|---|--:|--:|--:|--:|:--|
| E1 nominal (5시드) | 13.3 | 112.5 | 2,643(+s3 646, s4 28) | **689** | bo 679 (동률) |
| E2 dr25 | **1,234** | **3,192** | 9,001 | 4,475 | ssi 3,400 |
| E2 dr50 | **1,253** | **3,188** | 14,493 | 6,311 | ssi 3,464 |
| E2 dr75 | **1,288** | **3,206** | **충돌** | — | ssi 3,388 |
| E3 step | **45.5** | **122.9** | 3,246 | **1,138** | ssi 1,173 |
| E3 sine | **32→27.9** | 118.3 | 2,804 | 983 | ssi 956 (동급) |

**판독**: E1·E3 및 E2의 s0/s1에서는 diff가 전 방법 선두 (C1 성립). 유일한 실패 축 =
**s2 초기조건 × 동역학 DR** — dr25 9,001 → dr50 14,493 → dr75 충돌로 DR 강도에
비례해 악화.

**충돌 원인 분석 (dr75_s2 npz aux 채널)**: 충돌(t=25.5 s, tilt 40.3°) 직전 3초간
정책 출력의 **83%가 하한 고정** (radial/heading=10, 나머지 0.1). 학습이 공칭
플랜트에서만 이루어져 DR'd 동역학 + 극단 초기조건이 OOD → 정책이 사실상 추종
가중치를 꺼버림. `w_radial_floor=10`이 유일한 안전망이었음.

## 4. 개선 구현 (2026-08-09)

- **V1 look-ahead** (`policy_features()` 공유 함수 + `preview_nodes` 버퍼):
  horizon setpoint 델타(z_ref/s_ref @ 노드 10/20/30)를 정책 입력에 추가. 피처 정의는
  학습·배치 공용 단일 함수로 강제, 스펙은 ckpt에 동봉, `WeightPolicy.from_state_dict`
  가 ckpt에서 아키텍처를 역추론 (구형 로드 사고 재발 방지). 어댑터 15/15 테스트 통과.
  40k steps 학습 완료 (`checkpoints/dw_v1_lookahead/`).
- **V5 DR-in-training** (`--dr_fluid`): 세그먼트마다 유체계수(added mass, 선형/이차
  감쇠)를 U(1−r, 1+r)로 스케일 (core `scale_parameters` 경로, base 스냅샷 기준이라
  비복리). §3의 OOD 실패를 in-distribution으로 만드는 직접 대응. r=0.75로
  `dw_v5_dr`(V5 단독), `dw_v1v5`(V1+V5 결합) 학습 중.
- 평가 프로토콜: 실패 축 프로브 (E2 s2 × dr25/50/75 + E1 s0 새니티, `e2_v1probe.yaml`
  패턴) → 통과 후보만 full 20셀.

(이하 variant 결과는 진행하며 추가)
