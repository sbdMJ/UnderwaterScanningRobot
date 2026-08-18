# 도메인 대조: 부모 논문 (Diff-WMPC, 자율주행 레이싱) vs 본 연구 (UUV 수조 벽면 스캔)

작성 2026-08-10. 근거: 부모 논문 전문 (Jahncke et al., "Differentiable Weights-Varying
Nonlinear MPC via Gradient-Based Policy Learning: An Autonomous Vehicle Guidance
Example", RA-L vol.11 no.3, 2026, pp.3724–3731 — `docs/references/`의 PDF), diffmpc.com,
그리고 본 저장소의 실측 설정. 방법 자체의 변경점은
`docs/research/diffwmpc_domain_adaptations.md` 참조.

## 1. 플랫폼·태스크 대조표

| 축 | 부모 논문 (레이싱) | 본 연구 (UUV wallscan) |
|---|---|---|
| 플랜트 | Dallara AV-24 레이스카 (Indy Autonomous Challenge), 비선형 single-track + Pacejka MF 타이어, 하중이동·다운포스·항력 | PKRC UUV 22.8 kg, 6×T200 스러스터, Fossen 6-DOF (부가질량, 선형/이차 감쇠, 부력/복원) |
| 태스크 | 사전계산된 시간최적 궤적(레이싱라인) 추종 — Monza, Laguna Seca | 원통 수조(R=6 m, H=10 m) 벽면 lawnmower 스캔 — 스캔 상태머신이 참조를 온라인 생성 (DESCEND→SWAY→ASCEND→SWAY) |
| 속도 영역 | 피크 ≈ 79 m/s, **핸들링 한계** (타이어 포화, combined slip) | 스캔 속도 ≈ 0.2 m/s, **준정적 저속** (감쇠·부력 지배) |
| 성능 지배 요인 | **플랜트** (타이어 한계가 곧 성능; 모델 정확도가 결정적) | **상황** (phase·초기자세·외란; 플랜트 계수 변화는 폐루프가 흡수 — E2 평탄성, E2b′ 동률의 물리적 배경) |
| 환경 축 (전이) | 트랙 교체 (Monza→Laguna Seca) + 시뮬레이터 충실도 (Sim1→Sim2: MF96, per-wheel 하중, combined slip) | 트랙 개념 없음. 유체계수 DR ±25/50/75% (E2), 결정론적 플랜트 시프트 ×0.5/×1.5 (E2b′), 초기자세 ±45° 스트레스, 조류 step/sine 0.15 m/s (E3) |
| 외란 | 명시적 외란 축 없음 (공력은 모델 내) | **지속 해류** — 부모 논문에 없는 축. SSI-MPC 논문의 ground-effect에 대응 |
| 상태/제어 차원 | x = [x,y,ψ,v_long,v_lat,ψ̇,δ_f,a] (8), u = [저크 j, 조향률 ω_f] (2) | x13 (위치·쿼터니언·선속·각속), u = 정규화 6-스러스터 |
| 안전 경계 | 트랙 경계 + 결합 가속 제약 (OCP 제약으로 명시) | 벽 충돌은 **OCP 제약이 아니라 비용+종료 판정** (analytic geometry); 수면/바닥 kill 경계 |
| 상태 소스 | 시뮬 GT (하드웨어 배포는 future work) | 시뮬 GT (Phase 2a) + 실기체 EKF 축은 feature/sim-to-real에서 별도 진행 (marker-fix 기반 s-보정) |

## 2. MPC·학습 설정 대조표

| 축 | 부모 논문 | 본 연구 |
|---|---|---|
| Horizon | N=34, 2.55 s (dt 0.075 s) | N=30, 1.5 s (dt_mpc 0.05 s) |
| 솔버 | ACADOS SQP_RTI + HPIPM, KKT tol 1e-6, exact Hessian | 동일 스택. 명목 솔버 GAUSS_NEWTON + **감도 전용 EXACT-Hessian 솔버 분리** (rti_iters 8) |
| 감도 노드 | **노드 1** (1/10/20 비교 실측: 0.090/0.108/0.191 m RMSE) | **다중 노드 합 {N/6, N/3, 2N/3, N}** — 단일 원거리 노드에서 w_radial 붕괴 실측(2026-07-31) 후 시간스케일 재균형 목적 |
| 학습 가중치 수 | 6 (q_lat, q_ψ, q_vlong, q_alat, r_j, r_ω) | 18 (w_err 12 + w_u 6), log-scale sigmoid 매핑 + **per-entry 하한** (radial/heading floor 10) |
| 정책 입력 | **미래 참조** — 향후 2.55 s의 속도 5점 + 곡률 5점 (선제 적응이 핵심 메커니즘, Fig.5/8) | **현재 오차(12) + phase sin/cos + 과거 4스텝 히스토리** — look-ahead(V1)는 통제 실험에서 기각: 스캔 램프가 단순해 phase가 이미 정보를 담음 |
| 정책 구조 | FC 2×128, softplus 출력(양수 보장) | FC 2×128 tanh + bounded sigmoid 출력 (동급) |
| 손실 | lateral² + velocity² + **조건부 지수 패널티**(저크·조향률 임계 초과 시 exp) | 다항 추종(radial/heading/depth/자세/속도) + **정규화 제어항** u/max_thrust (뉴턴 단위면 "learn-to-do-nothing" 병리 — 포팅 불변량) |
| 옵티마이저 | Adam lr 2.9e-5, batch 10 timestep, clip ±0.1, 실패 시 마지막 feasible 가중치 폴백 | Adam lr 5e-4, batch 10, clip 0.1, **포화 스텝 스킵** (감도가 경계에서 정확히 0 — 실측 불변량) |
| 학습 데이터 구성 | 에피소드(랩) 연속 주행 — 6랩(≈101 s, 36.8k 샘플)에 수렴 | **600-스텝 랜덤 세그먼트** (반경·방위·심도·자세·phase 추첨) — 에피소드(180 s)가 길어 커버리지 우선. 최종 레시피 60k 스텝 + **자세 0.7 rad 20% 혼합** |
| 학습 중 플랜트 | Sim1 고정 (모델 불일치 학습은 별도 실험: MPC 내부 Sim1, 구동 Sim2) | 공칭 고정 — **동역학 DR 혼입(V5)은 실측 악화로 기각** |
| Solve 시간 | ≈2.2–2.4 ms/step (Ryzen 7950X) | ≈8–9 ms/step (N=30, 13-상태; 데스크톱; Jetson 실측은 E4c 대기) |

## 3. 평가 프로토콜·결과 구도 대조

| 축 | 부모 논문 | 본 연구 |
|---|---|---|
| 베이스라인 | HE-MPC(수동), MOBO-MPC(460k 샘플/1,071 s), RL-WMPC(1M 샘플) | Fixed-W(수동), BO-static(다중시드 재튜닝, 1.43M 샘플), SSI-MPC(공식 포팅+재튜닝), PPO(DORAEMON DR — 비대칭 공개) |
| 핵심 결과 | 모델 불일치 하 최고 정적 대비 lateral **−50%**, 속도오차 −37.5%; 학습 101 s (RL 대비 38.5×) | E2 전 조건 SSI 대비 **−15~18%**, E1 동률 1위, E3 step 1위; 자세-OOD 충돌 해소가 캠페인 본체 |
| Fine-tune 구도 | Monza→Laguna Seca **27 s fine-tune으로 from-scratch 수준 도달** (적응이 필요했고, 빨랐다) | 고정 플랜트 3종에서 **zero-shot ≈ fine-tune ≈ from-scratch** (적응이 애초에 불필요 — 상한이 zero-shot 수준) |
| 강건성 곡선 | 트랙/충실도 전이에서 열화를 fine-tune으로 회복 | 유체계수 스윕에 평탄 (부호 있는 추첨의 시드 상쇄 + 자세 스트레스 지배 — validation_report §4.1) |
| 민감도 | 초기 가중치 30개 무작위 → 전부 수렴 | (대응 실험 없음 — BO/SSI 쪽은 예산·다중시드로 방어) |

## 4. 도메인 차이의 함의 (요약)

1. **플랜트-지배 vs 상황-지배**: 레이싱에서 가중치 적응의 가치는 "코너 진입 전
   플랜트 한계를 예견"하는 데 있고, wallscan에서는 "phase·자세 상황에 맞는 축
   우선순위 전환"에 있다 (F7: sway leg에서 diff만 tilt 0°대; E4a: preview 제거 시
   +41%). 같은 방법, 다른 가치 축.
2. **전이 축의 부재**: 우리 도메인엔 "다른 트랙"이 없어 fine-tune 서사가 플랜트
   시프트로 번역되는데, 저속 폐루프가 이를 흡수해 zero-shot으로 충분하다 — 부모
   논문의 역상이며, 논문에서는 도메인 차이로 명시해야 공격받지 않는다.
3. **외란 축의 추가**: 조류(E3)는 부모 논문에 없는 축 — 적응형 방법(SSI, diff)의
   가치가 가장 크게 드러나는 곳이므로 우리 논문의 차별화 포인트.
4. **감도 노드·세그먼트 학습·자세 커버리지**는 도메인 특이 요구로 원 방법에서
   변경된 부분 — 상세는 adaptations 문서.
