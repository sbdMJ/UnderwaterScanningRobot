# Figure 트렌드 감사 (예측 vs 실제) — 2026-08-09

> 방법: `experimental_results/*/figures/`의 전 그림을 렌더링 이미지로 직접 검수.
> 각 그림에 대해 (설계 의도·문헌에서 오는) 기대 트렌드를 먼저 적고 실제와 대조.
> 불일치는 원인 분석 → 해결책 → (sim-to-real 브랜치 충돌 검토 후) 즉시 수정 여부 순.
> sim-to-real(feature/sim-to-real)은 `figures.py`/`plot_figures.py`를 건드리지 않으므로
> 본 감사의 모든 수정은 충돌 없음 (§4 참조).

## 1. 감사 표

| Fig | 내용 | 기대 트렌드 | 실제 | 판정 | 조치 |
|---|---|---|---|---|---|
| F1 (e1) | 본 비교 4패널 | objective: diff≈bo≈ssi < nominal ≪ PPO(8-env 합산); wall_dist는 PPO 최소; tilt는 MPC 우위; cycles 전부 2.0 | 기대와 일치 — 단 objective 패널이 선형축에서 PPO에 눌려 판독 불가**였음** | ✅ (수정 후) | objective 패널 log 축 (`logy_metrics`) — 수정·재생성 완료 |
| F2 (e1) | 대표 궤적 s–z | 5개 방법 전부 lawnmower 추종, PPO만 초기 spin-search 일탈 | 일치 — 단 **diff 패널 누락**이었음 (채택 이전 렌더) | ✅ (수정 후) | 재생성으로 5패널 완성 |
| F3 (e2) | 강건성 곡선 | (소박한 기대) 레벨↑ → objective↑ | **전 방법 평탄** | ⚠️ 불일치 — 원인 규명됨 | §2.1 참조. 그림 자체는 정직 (per-trial 점이 산포 증가를 보임). 수치 근거는 validation_report §4.1 |
| F4 (e3) | step 조류 응답 | t=90 역전에서 교란 반응 + 적응형(ssi/diff)이 빠른 회복, nominal은 지속 오차 | 기존 2패널(wall-dist, tilt)에서는 **거의 안 보였음** | ⚠️ → ✅ (수정 후) | §2.2 참조. 호길이 오차 패널 추가 — nominal 0.1–0.17 m 지속 오차 vs ssi/diff 0.01–0.02 m (10×)가 이제 가시화 |
| F5 (e2 vs e2b) | zero-shot vs fine-tuned | ft ≤ zs | 대략 동률 (분포-수준 ft — 프로토콜 결함으로 판명) | ⚠️ 재실험 중 | E2(b)′ 3-arm 완료 시 e2c 기반으로 재생성 예정. PPO inf는 막대 생략(수정 완료), log 축 적용 |
| F6 (e1) | 비용 비교 | offline: 튜닝 방법만 ~6ks; inference: MPC ~8ms ≫ PPO 0.04ms | 일치 — 단 PPO inference 막대가 선형축에서 **비가시**였음 + diff 열 누락 | ✅ (수정 후) | ms 패널 >50× 자동 log + 재생성. **잔여 주의**: PPO·diff의 offline(학습) 비용이 budget.json에 없어 0으로 보임 — W&B/학습 로그 기반 수치 주입 필요 (§3-b) |
| F7 (e1) | 상태 패널 | diff가 sway leg에서 tilt 최평탄 (가변 가중치의 가치), 램프 추종 동급 | 일치 — diff tilt가 유일하게 sway 중 0°대 유지. **누락이었던 diff 추가됨** | ✅ (수정 후) | 미관 이슈: t=180 에피소드 종료 리셋 프레임이 스파이크로 찍힘 (전 방법 공통, §3-c) |
| F8 (e1) | 태스크 다이어그램 | 정적 도식 | 정상 | ✅ | 호스트 matplotlib Axes3D 충돌로 컨테이너 렌더 필요 (기록됨) |
| F9 (e3) | SSI 예측 오차 | 과도구간(초기·sway 전환·t=90)마다 스파이크 후 적응으로 소멸 | 일치 — 초기 0.4 → 10 s 내 0 수렴, 전환마다 0.2급 스파이크 후 즉시 소멸 | ✅ | — |
| F10 (e4) | lr·M 민감도 | 두 패널 모두 평탄 (원논문 Fig.4 메시지) | m 패널 정상, **lr 패널에 1점만** 표시되는 버그 | ❌ → ✅ (수정 후) | §2.3 참조. family 분리 로직 수정·재생성 — lr 6점 스윕 정상 표시, 평탄 확인 |

## 2. 불일치 원인 분석

### 2.1 F3 평탄성 (실험 성질 — 버그 아님)

`docs/experiments/validation_report.md` §4.1에 상세 기록. 요지: ① DR 적용 경로는
정상, ② 유체계수 추첨은 부호가 있어 시드별로 단조 방향이 반대(s0↑/s1 평탄/s2↓ —
nominal·diff 동일)라 3-시드 평균이 상쇄, ③ 상수인 초기자세 스트레스(시드 축,
1.9–3×)가 스윕 축(±1.3%)을 지배. **해결책**: 논문에서는 "±75%까지 흡수 + 방법 간
간격 유지"로 서술; 단조 축이 필요하면 최악-시드 곡선 또는 초기자세 스윕 조건 신설.
(초기자세 스윕은 `env_variants`에 조건 추가만으로 가능 — sim-to-real과 무충돌.
단, 조건 추가는 실험 볼륨 증가라 진행 여부는 논문 필요성 판단 후.)

### 2.2 F4 교란 반응 비가시 (그림 설계 결함 — 수정 완료)

지속 조류는 반경 방향(wall-dist)이 아니라 **접선 방향 = 호길이 추종**을 주로
때린다 — E3 objective 격차(ssi/diff가 nominal의 절반)의 실제 출처. 기존 패널
(wall-dist, tilt)은 이 축을 안 보여줬다. 호길이 오차 패널을 추가해 해결: nominal의
지속 0.1–0.17 m 오차와 t=90 반응, 적응형의 10× 우위가 가시화됨.

### 2.3 F10 lr 패널 버그 (플로팅 버그 — 수정 완료)

lr-패밀리 셀은 `ssi_n_rf` 키가 없고(업스트림 기본값 사용) M-패밀리만 설정한다.
기존 로직은 "최빈 n_rf"로 lr 패널을 필터링해 M-패밀리의 한 값(n_rf=25, 고정 lr)만
남겼다. 수정: **키 부재 자체가 패밀리 마커** — `ssi_n_rf is None` → lr 패널,
설정됨 → M 패널.

## 3. 잔여 항목 (기록만, 수정 보류)

- (a) **F5 최종본**: E2(b)′ 3-arm(e2c_zs/ft/sc) 완료 후 교체 — 진행 중.
- (b) **F6 offline 비용의 공정성**: PPO(rb_train, W&B 기록)와 diff(V8a 60k 학습,
  ~50분)의 학습 비용이 budget.json 체계 밖이라 0으로 표시. E4(b) 표와 함께
  학습 비용을 `collect_budgets` 호환 json으로 내보내는 작업 필요.
- (c) **t=180 리셋 프레임 스파이크** (F4/F7): 러너가 마지막 스텝에 리셋 후 상태를
  기록 — 플롯에서 마지막 1스텝 트림이 안전하지만, npz 규약 변경은 rescore 호환성
  검토 후 (전 방법 공통이라 비교 왜곡은 없음).

## 4. sim-to-real 충돌 검토

본 감사의 수정 파일: `marinelab/marinelab/experiments/figures.py`,
`marinelab/scripts/experiments/plot_figures.py`, `experimental_results/*/figures/*`.
feature/sim-to-real 브랜치의 변경 파일(sensors/estimator/wall_frame_ekf/_sim_loop의
SimSensorStream + ros/ + hw_*)과 **교집합 없음** — 병합 충돌·의미 충돌 모두 없음.
그쪽 실험(e5_ekf: ssi/nominal, EKF 상태 축)은 objective 스케일이 우리 그림에
들어오지 않으므로 그림 변경이 그쪽 결과 해석에 영향 주지 않음. F3 관련 조건 신설
(초기자세 스윕)을 하게 되면 `env_variants.py`를 확장하게 되는데, 그쪽도 이 파일을
읽기만 하므로(신규 조건 추가는 additive) 충돌 없음.
