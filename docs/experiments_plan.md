# ICRA 실험 설계 (최종안)

**논문 방향**: Diff-WMPC(RA-L 2026, Jahncke et al.)를 6-DOF UUV 수조 벽면 스캔으로 확장
**제약**: 시뮬레이션 위주, 하드웨어 최소화, 기존 논문(SSI-MPC/AC-MPC/Diff-WMPC) 실험 규모 수준
**기존 자산 재활용**: 고정 가중치 NMPC baseline(`run_wallscan_mpc.py`), PPO 정책(`rb_train_model`), stress DR·`OceanCurrent`·`eval_metrics` 인프라, 8env×180s 평가 프로토콜

---

## 논문이 입증해야 할 주장 (실험 ↔ 주장 매핑)

| # | 주장 | 입증 실험 |
|---|---|---|
| C1 | 가변 가중치가 고정 가중치 NMPC보다 wallscan에서 우수 (부모 논문 축의 핵심) | E1, E4 |
| C2 | 가중치 적응이 모델 적응(SSI-MPC)·순수 RL(PPO) 대비 경쟁력 있음 | E1, E2, E3 |
| C3 | 불확실성 하에서 zero-shot 전이 + 온라인 적응으로 강건 (확장 기여) | E2 |
| C4 | 실기체 이식 가능 (sim-to-real + 추론 예산) | E4(c), E5 |

---

## E1. 공칭 조건 본 비교 [MUST]

- **방법 (5개)**: ① Fixed-W NMPC (보유) ② BO-tuned static NMPC (Optuna ~100 trial로 최적 고정 가중치 — 부모 논문 MOBO-MPC 대응; "가변이 정말 필요한가"에 대한 방어) ③ PPO (보유) ④ SSI-MPC (이식: 동일 acados 스택, 학습 불필요·튜닝만) ⑤ 제안 Diff-WMPC
- **프로토콜**: 8 env × 180 s × 5 seed, GT 상태 (Phase 2a 기준으로 통일)
- **지표**: 기울기(수직/sway), 스캔속도(heave/sway), 게걸음(yaw−θ), ŝ 오차, 사이클 달성 + 추종 RMSE + 제어 스무스니스
- **통계**: mean±SD + per-trial 점 오버레이 (min/max 막대 금지 — SSI-MPC가 심사에서 비판받은 지점)
- **산출물**: Table 1 (본 비교표)

## E2. 강건성: zero-shot 전이 + 온라인 적응 [MUST]

부모 논문의 Monza→Laguna Seca 실험의 수중판. 확장 기여(C3)의 본체.

- **조건**: 공칭에서 학습한 가중치 정책을 그대로 → (a) stress DR zero-shot (유체계수 ±25/±50/±75% 3단계 스윕, 초기자세 ±45°, 센서 마운트 ±8 cm) → (b) 온라인 fine-tuning 후 (부모 논문의 "quick online fine-tuning" 대응)
- **방법**: 제안, SSI-MPC, PPO, Fixed-W NMPC (4개)
- **산출물**: Fig. 성능 vs 섭동 강도 곡선 (AC-MPC의 질량/관성 스윕 형식) + zero-shot/fine-tuned 막대 비교
- **추가(저비용, 권장)**: EKF-in-loop 조건 1개 (`wall_frame_ekf`) — GT vs EKF 상태로 성능 차이를 시뮬에서 미리 보여 하드웨어 결과의 해석 근거 마련

## E3. 조류 외란 적응 [SHOULD]

- **조건**: `OceanCurrent` 스텝 변화(급전환 — SSI-MPC 지면효과 실험의 수중판) + 사인 변화 1개
- **방법**: 제안, SSI-MPC, Fixed-W NMPC (3개)
- **주의**: 외란 모델 학습은 SSI-MPC의 홈그라운드. 여기서 밀리면 정직하게 보고하고 "가중치 적응과 모델 적응은 상보적"으로 서술 (결합은 future work) — 오히려 분석의 깊이로 점수를 얻는 축

## E4. Ablation + 비용 [MUST (a, c) / SHOULD (b)]

- **(a) 구성요소 ablation** (2–3개만): 가중치 상한 500 vs 5000 (롤 가중치 ~2000 실측 근거의 정당화), saturation-skip on/off (수중 특화 기여 입증), reference preview on/off
- **(b) 학습 비용 표**: 학습 시간/샘플 수 — 제안 vs PPO (vs RL-WMPC 여력 시). 부모 논문의 "2분 학습" 셀링포인트 계승
- **(c) 추론 비용 표**: acados solve time + policy forward, 데스크톱 + **Jetson 실측 권장**. AC-MPC(13.5–210 ms)와의 문헌 대비 — AC-MPC를 비교군에서 제외하는 근거를 겸함
- **산출물**: Table 2 (ablation), Table 3 (비용)

## E5. 하드웨어 [MUST, 최소 규모]

SSI-MPC 실기체 규모(3 시나리오 × 5회)가 상한 준거. 그 이하로:

- **시나리오 1개**: 실수조 벽면 스캔, 공칭 조건, zero-shot (시뮬 학습 가중치 정책 그대로), EKF 상태
- **방법 3개**: 제안 vs Fixed-W NMPC vs PPO — 각 **3–5회** = 총 9–15회 주행
- **여력 시 +α**: 변형 조건 1개(페이로드 부착 등 질량/부력 변화) × 제안만 3회
- **주장 한정**: "시뮬 순위가 실기체에서 유지된다" — 이 이상 주장하지 않음
- **산출물**: Table 4 + 궤적 Fig + (비디오 첨부)

---

## 제외 및 근거 (논문/리뷰 대응용)

| 항목 | 처리 | 근거 |
|---|---|---|
| AC-MPC 비교 | 제외, related work 인용 | 미분가능 MPC 학습 30× 비용 + 추론 13.5–210 ms (T-RO 원문 Table III) + 공개 코드 메모리 누수(CA-AC-MPC가 보고, acmpc_public issue #1). E4(c)에서 문헌 수치로 간접 비교 |
| Diff-WMPC 원 세팅 재현 | 불가 명시 | 원 코드 골격만 공개(특허 계류) — 자체 구현임을 논문에 명시, "based on open-source" 표현 금지 |
| RL-WMPC | 여력 시만 | 블랙박스 RL 가중치 조정 — 학습 비용 큼, E4(b)의 비용 비교로 부분 대체 |
| DOB-MPC | 여력 시만 | E1에 저비용 추가 가능하나 5개 방법이면 이미 충분 |
| 벽 근접 유체효과 모델링 | 제외, limitation 명시 | 새 모델링 작업 — 이번 범위 밖 |
| 코드 공개 | 기여분만 | wallscan NMPC 정식화·EKF·태스크는 공개, Diff-WMPC 학습 코어는 특허 계류 고려해 비공개/on-request. 원저자(TUM AVS) 접촉 권장 |

## 규모 sanity check (기존 논문 대비)

- SSI-MPC: 시뮬 2 태스크 + 민감도, 하드웨어 3 시나리오 × 5회 → **본 설계: 시뮬 4 실험, 하드웨어 1 시나리오 × 3방법 × 3–5회** — 동급
- Diff-WMPC: 시뮬 only 2 트랙 + 초기화 강건성 → 본 설계가 하드웨어 포함으로 우위
- 학습 필요 항목: 제안 방법(+ablation 변형)뿐 — 대조군 중 학습 필요한 것 없음(PPO 보유, SSI-MPC 온라인, BO는 튜닝)

## 실행 순서 (의존성)

1. E1 (Fixed-W·PPO는 즉시, BO 튜닝 병렬, SSI-MPC 이식 착수) — 본 비교표 확보가 최우선
2. E4(a) — E1 인프라 그대로 변형 실행
3. E2 → E3 — DR/조류 스윕 (E1 상위 방법 확정 후)
4. E4(b,c) — 언제든 병렬
5. E5 — 시뮬 결과 고정 후 마지막 (하드웨어 슬롯 예약은 미리)
