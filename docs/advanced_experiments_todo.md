# 후속 실험 계획 (advanced experiments TODO)

작성 2026-08-10. 배경: 시뮬 실험 계획(E1–E4, E2b′, F1–F10)은 완결
(`docs/README.md` §2). 이 문서는 그 결과와 concern(C-1~C-9,
`docs/research/diffwmpc_domain_adaptations.md` §4)에서 도출된 **다음 실험 후보**의
실행 계획이다. 실행 규칙은 CLAUDE.md(모니터링 cron·스팟체크 의무)와
`docs/experiments/tuning_history/experiment_work_directives.md`(red-flag)를 따른다.

## A. 조류 변화 환경 온라인 fine-tune (우선순위 1) — ★

**질문**: 적응이 실제로 값을 하는 축(지속·방향성 외란)에서는 원논문의 "quick online
fine-tuning" 서사가 복원되는가? (E2b′의 플랜트 시프트에서는 zero-shot이 상한이었음)

**설계 근거**: 조류는 월드-고정 방향 + 로봇은 벽 순회 → 벽 좌표계 외란이 θ에 따라
변조 = 가중치 정책의 히스토리 피처가 설계상 겨냥한 "미모델 외란 추론" 상황.
E3에서 적응 축임이 확인됨 (ssi가 nominal +39~44%; diff zero-shot 1,138/983).

- [ ] 학습기에 조류 주입: `train_diff_wmpc_wallscan.py`에 `--current speed,heading[,profile]`
      — 러너 `CurrentDriver`와 동일 메커니즘 (core `OceanCurrent` 주입; 세그먼트당
      상수 or step/sine). MPC 내부 모델은 공칭 유지 (외란 = 시험 대상).
- [ ] 조건: 기존 E3 step/sine (0.15 m/s) + **강한 조류 0.25 m/s** 신설 (headroom 확보).
      조건별 ft = V8a에서 10k 계속학습 (`--resume_ckpt`, E2b′와 동일 프로토콜).
- [ ] 3-arm: zero-shot(V8a) / fine-tuned(10k) / from-scratch(60k) × 조건 × seeds 0–2
      → `e3b_zs/ft/sc` (조건 명명·디렉토리 분리 규약 준수).
- [ ] 판정: ft가 zs 대비 유의 개선이면 "적응 필요 축에서는 빠른 적응" 서사 복원;
      동률이면 zero-shot 충분성 주장 강화. 어느 쪽이든 diff_tuning_history에 기록.
- 예상 비용: 구현 ~1 h + 학습(ft 3×10 min, sc 3×50 min) + 평가 ~27셀 ≈ **반나절–1일**.
- 충돌 검토: trainer·config·신규 결과 디렉토리만 — sim-to-real과 무충돌.

## B. 속도 상향 실험 (우선순위 2, A와 병렬 가능)

**질문**: 스캔 속도를 올리면 (이차 항력 지배 시작) "상황-지배 → 플랜트-지배" 경계가
어디서 나타나는가? — E2 평탄성(C-3)의 단조 축 대안 + "zero-shot 충분" 주장의
적용 한계 자기 측정(C-7).

- 물리 여유: 정상상태 항력 기준 ~1 m/s급까지 (0.2 m/s는 추력 예산의 수 %).
- [ ] 1차 (zero-shot 프로브): 속도 오버라이드 조건 (`ScanCfg.ref_step/ref_step_s`
      스케일, env_variants 방식 additive) — 0.3 / 0.4 m/s × {nominal, bo, ssi, diff}
      × seeds 0–2 = 24셀. **셀당 wall-clock 동일** (같은 9,000스텝). ≈ 2 h.
  - 주의: reach band 통과 타이밍 변화 — CLAUDE.md의 평형점 vs reach-band 함정
    계열. 첫 셀에서 `Scan/end_phase`·cycles로 상태머신 정상 동작 확인 후 확대.
- [ ] 2차 (속도별 재최적화, 1차 결과가 흥미로울 때만): BO/SSI 재튜닝(각 ~6 h, 병렬)
      + diff 재학습(~1 h) ≈ +1일.
- [ ] PPO는 보상에 속도 타깃·캡(0.2 m/s) 하드코딩 → 재학습 필수(+1–2일, 커리큘럼
      리스크) — 1·2차에서 제외하고 필요 시 별도 결정.

## C. diff의 EKF-조건 게이트 (우선순위 3 — sim-to-real 병합 후)

**질문**: 가중치 정책 입력이 추정 상태로 오염될 때 diff가 성능을 유지하는가?
(concern C-2 — E5 투입 전 필수 게이트)

- **순서 제약**: 측정 센서 모델·EKF s-보정·vis7 조건은 전부 feature/sim-to-real에
  있음. 우리 브랜치에서 지금 돌리면 그쪽이 수정한 5–7× 공통모드 열화를 재발견하는
  중복 작업 — **병합(또는 그 브랜치 위) 후 실행**.
- [ ] 병합 시 회귀 게이트 (C-9): 그쪽 sfix s4 = 96.7 재현 + 우리 E1 diff s0 = 13.3
      재현을 양방향 확인.
- [ ] diff zero-shot × measured_aruco_sfix_vis7 조건 (e5_ekf 프로토콜, 5셀 ≈ 30 min).
- [ ] 격차 발견 시에만: EKF-조건 fine-tune ("지각 스택에 대한 적응") — 학습기에
      SimSensorStream 통합 필요 (반나절+). 격차 없으면 E5 go에 diff 추가.

## D. 백로그 (여력 시)

- [ ] E4 민감도를 E3 조류 조건에서 반복 (validation_report §2-E4 권고 — 적응이
      실제로 일할 때의 lr/M 민감도).
- [ ] 보수 lr(2.9e-5) 대조 학습 + 초기화 강건성(무작위 30개 수렴) — 원논문 대응
      부록 실험 (C-8).
- [ ] 초기자세 스윕을 E2의 단조 축으로 신설 (figure_trend_audit §2.1의 대안 축).
- [ ] F6 offline 비용 공정성: PPO(W&B)·diff(60k≈50 min) 학습 비용을
      `collect_budgets` 호환 json으로 주입 (figure_trend_audit §3-b).
- [ ] E4(c) Jetson 추론 벤치 (`bench_inference.py` 신설, sim-to-real Phase D와 조율).

## 진행 규칙 리마인드

- 각 항목 착수 시 30분+ 작업이면 자가치유 cron 등록, 튜닝·학습이면 중간 스팟체크.
- diff가 어느 실험에서든 bo/ssi에 뒤지면 즉각 원인 분석(aux 계측) — 재실행 반복 금지.
- 결과·여정은 `docs/experiments/tuning_history/diff_tuning_history.md`에 이어서 기록,
  조건 의미가 바뀌면 새 조건명 (덮어쓰기 금지).
- 동시 컨테이너 상한 4 (12 GB GPU 실측 한계 — 2026-08-09 OOM 사례).
