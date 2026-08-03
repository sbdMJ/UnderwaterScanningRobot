# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Per-env waypoint state machine for the vertical zig-zag wall scan.

Phases cycle DESCEND(0) -> SWAY(1) -> ASCEND(2) -> SWAY(3) -> DESCEND ...
Entering a SWAY phase bumps s_ref by sway_step (direction held constant, so
the scan keeps progressing around the tank rather than oscillating back and
forth -- a lawnmower/boustrophedon pattern in (depth, s)) and latches z_hold
to the depth at that instant, so z_ref holds steady through the sway instead
of trivially tracking live z (which would give the depth-hold reward no
signal to resist vertical drift while swaying)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

DESCEND, SWAY_A, ASCEND, SWAY_B = 0, 1, 2, 3


@dataclass
class ScanCfg:
    z_top: float
    z_bottom: float
    reach_eps: float
    reach_hold: int
    sway_step: float = 1.0
    # Skip SWAY phases entirely (DESCEND<->ASCEND only). For sway-disabled curriculum
    # stages: with sway rewards off, nothing holds s, so the ~1 m of free tangential
    # drift during a 34 s descend beats any |s| auto-clear band (g_stage2: end_phase
    # pinned at 1.0 for 4000 iters even with sway_step=0).
    skip_sway: bool = False
    # Reference RAMP (07-26 real-scan pacing): metres the emitted z_ref/s_ref may move
    # per control step toward the phase target. 0 = instant refs (old behaviour). A
    # speed-cap PENALTY lost the reward-economics fight (m_train7: dwell-at-target
    # tracking + earlier waypoint bonuses out-paid the fine, speeds stayed 0.5-0.65 m/s)
    # — a moving reference aligns the economics instead: overtaking the ramp LOSES
    # tracking reward, so the scan speed pins to ref_step/dt by construction.
    ref_step: float = 0.0
    # Separate SWAY ramp speed (07-28 tilt mitigation): sidestep thrust heels the hull
    # via the TAM My coupling roughly in proportion to speed, so the sway leg can run
    # slower than the heave legs. 0 = use ref_step for both axes.
    ref_step_s: float = 0.0


class ScanState:
    """Per-env waypoint state, N envs."""

    def __init__(self, n: int, device=None, dtype: torch.dtype = torch.float32):
        self.phase = torch.zeros(n, dtype=torch.long, device=device)
        self.s_ref = torch.zeros(n, dtype=dtype, device=device)
        self.sway_dir = torch.ones(n, dtype=dtype, device=device)
        self.z_hold = torch.zeros(n, dtype=dtype, device=device)  # depth latched on SWAY entry
        self._hold = torch.zeros(n, dtype=torch.long, device=device)  # reach-hold counter
        # Ramped references (used only when cfg.ref_step > 0); the env re-anchors these
        # at reset and at search end so the ramp starts from where the robot actually is.
        self.z_ramp = torch.zeros(n, dtype=dtype, device=device)
        self.s_ramp = torch.zeros(n, dtype=dtype, device=device)


def step(state: ScanState, z: torch.Tensor, s: torch.Tensor, cfg: ScanCfg, z_latch: torch.Tensor | None = None):
    """Advance the state machine one step. Returns (z_ref, s_ref, phase_sc[N,2], advanced).

    `z`/`s` drive phase TIMING (reach detection) -- these are what the vehicle perceives, so the
    caller passes the (possibly DR-biased) sensor readings. `z_latch` is the value snapshotted into
    z_hold on SWAY entry, which becomes z_ref for the depth-hold reward; pass GROUND-TRUTH depth
    here so a biased depth sensor never puts a constant offset into the GT-compared depth reward.
    Defaults to `z` when omitted (unbiased/test callers)."""
    if z_latch is None:
        z_latch = z
    is_sway = (state.phase == SWAY_A) | (state.phase == SWAY_B)
    is_ascend = state.phase == ASCEND
    z_target = torch.where(is_ascend, torch.full_like(z, cfg.z_top), torch.full_like(z, cfg.z_bottom))

    reach = torch.where(is_sway, (s - state.s_ref).abs() < cfg.reach_eps, (z - z_target).abs() < cfg.reach_eps)
    if cfg.ref_step > 0.0:
        # Ramp gate (07-26 pacing fix): the vehicle can DASH to the final target ahead of
        # the ramp and clear reach early — heave stayed at 0.54 m/s while sway (short
        # hops, nothing to gain) complied. Require the RAMP to have arrived too, so
        # outrunning the reference buys nothing.
        ramp_there = torch.where(
            is_sway,
            (state.s_ramp - state.s_ref).abs() < cfg.reach_eps,
            (state.z_ramp - z_target).abs() < cfg.reach_eps,
        )
        reach = reach & ramp_there
    state._hold = torch.where(reach, state._hold + 1, torch.zeros_like(state._hold))

    advanced = state._hold >= cfg.reach_hold
    phase_inc = 2 if cfg.skip_sway else 1
    state.phase = torch.where(advanced, (state.phase + phase_inc) % 4, state.phase)
    state._hold = torch.where(advanced, torch.zeros_like(state._hold), state._hold)

    entering_sway = advanced & ((state.phase == SWAY_A) | (state.phase == SWAY_B))
    # s_ref bumps from the CURRENT s, not the previous s_ref (07-25 trace: ~1 m of lateral
    # drift during the descend landed s inside the old absolute-ladder band, so the sway
    # phase cleared in ~1 s with no deliberate sidestep and the ascent overlapped it —
    # the "diagonal move" seen in the viewer. Relative bump = always a real 1 m step.
    state.s_ref = torch.where(entering_sway, s + state.sway_dir * cfg.sway_step, state.s_ref)
    state.z_hold = torch.where(entering_sway, z_latch, state.z_hold)

    is_sway_out = (state.phase == SWAY_A) | (state.phase == SWAY_B)
    is_ascend_out = state.phase == ASCEND
    z_ref = torch.where(is_sway_out, state.z_hold, torch.where(is_ascend_out, torch.full_like(z, cfg.z_top), torch.full_like(z, cfg.z_bottom)))

    phase_f = state.phase.to(z.dtype)
    phase_sc = torch.stack([torch.sin(2 * math.pi * phase_f / 4), torch.cos(2 * math.pi * phase_f / 4)], dim=-1)

    if cfg.ref_step > 0.0:
        # Slew the emitted refs toward the phase targets at ref_step per call. Reach
        # detection above stays on the FINAL targets, so phases still advance only
        # when the vehicle truly arrives at the endpoint band.
        step_s = cfg.ref_step_s if cfg.ref_step_s > 0.0 else cfg.ref_step
        state.z_ramp = state.z_ramp + (z_ref - state.z_ramp).clamp(-cfg.ref_step, cfg.ref_step)
        state.s_ramp = state.s_ramp + (state.s_ref - state.s_ramp).clamp(-step_s, step_s)
        return state.z_ramp, state.s_ramp, phase_sc, advanced

    return z_ref, state.s_ref, phase_sc, advanced


def search_step(
    swept: torch.Tensor,
    best_dist: torch.Tensor,
    best_yaw: torch.Tensor,
    sonar: torch.Tensor,
    heading: torch.Tensor,
    omega: float,
    dt: float,
):
    """One step of the initial spin-search: while the env sweeps the yaw reference a full
    turn, track the bearing of the sonar minimum (= nearest wall). All tensors [N].

    Returns (swept, best_dist, best_yaw, active): `active` goes False once swept >= 2*pi;
    the caller then locks its yaw reference to `best_yaw` and starts the scan cycle.
    """
    better = sonar < best_dist
    best_dist = torch.where(better, sonar, best_dist)
    best_yaw = torch.where(better, heading, best_yaw)
    swept = swept + omega * dt
    active = swept < 2.0 * math.pi
    return swept, best_dist, best_yaw, active


def demo():
    n = 3
    st = ScanState(n=n)
    cfg = ScanCfg(z_top=9.0, z_bottom=1.0, reach_eps=0.1, reach_hold=2)

    # not yet at bottom -> no advance, hold counter builds but doesn't cross reach_hold
    z, s = torch.full((n,), 5.0), torch.zeros(n)
    _, _, _, adv = step(st, z, s, cfg)
    assert not adv.any() and (st.phase == DESCEND).all()

    # reach bottom for reach_hold consecutive steps -> advances to SWAY, s_ref bumped
    z = torch.full((n,), 1.02)
    step(st, z, s, cfg)  # 1st consecutive reach
    _, s_ref, phase_sc, adv = step(st, z, s, cfg)  # 2nd -> advance
    assert adv.all() and (st.phase == SWAY_A).all()
    assert torch.isclose(s_ref, torch.full((n,), 1.0), atol=1e-4).all()
    assert torch.allclose(phase_sc, torch.tensor([[1.0, 0.0]]).repeat(n, 1), atol=1e-6)

    # sway reached -> advance to ASCEND, s_ref unchanged
    step(st, z, s_ref, cfg)
    _, s_ref2, _, adv = step(st, z, s_ref, cfg)
    assert adv.all() and (st.phase == ASCEND).all() and torch.equal(s_ref2, s_ref)

    print("scan_state_machine.py demo OK")


if __name__ == "__main__":
    demo()
