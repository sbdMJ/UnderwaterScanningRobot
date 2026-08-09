# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""WallScanControlLoop: the §8 closed-loop guarantee, natively.

``competitor_framework_plan.md`` §8 item 1: the full hardware tick — sensor sample ->
wall-frame EKF -> scan state machine / reference preview -> controller — must run with no
simulator. These tests drive exactly that chain with a scripted vehicle and a recording
dummy controller.
"""
import math

import numpy as np
import pytest

from marinelab.control.estimator import SensorSample, WallFrameStateEstimator
from marinelab.control.hw_bridge import TankCalib, TopicSampleAssembler
from marinelab.control.scan_loop import WallScanControlLoop
from marinelab.control.types import ControlOutput, VehicleState
from marinelab.tasks.pkrc_wallscan.mpc_reference import WallScanMPCCfg
from marinelab.tasks.pkrc_wallscan.scan_state_machine import ScanCfg


class RecordingController:
    """WallScanController-shaped stub: returns zero thrust, remembers what it was shown."""

    def __init__(self):
        self.refs = []
        self.states = []

    def reset(self, state) -> None:
        self.refs.clear()
        self.states.clear()

    def step(self, state, ref) -> ControlOutput:
        self.states.append(state)
        self.refs.append(ref)
        return ControlOutput(u_cmd=np.zeros(6))


def _cfgs():
    # Stage3 scan parameters (wallscan_env_cfg) and the runner's MPC reference cfg.
    scan = ScanCfg(z_top=8.5, z_bottom=1.0, sway_step=1.0, reach_eps=0.6, reach_hold=10,
                   ref_step=0.004, ref_step_s=0.002)
    mpc = WallScanMPCCfg(tank_radius=6.0, d_ref=1.5, z_top=8.5, z_bottom=1.0, sway_step=1.0,
                         ref_step=0.004, ref_step_s=0.002, step_dt=0.02, dt_mpc=0.05)
    return scan, mpc


def _veh(z: float, r: float = 4.5, theta: float = 0.0) -> VehicleState:
    return VehicleState(
        pos_w=np.array([r * math.cos(theta), r * math.sin(theta), z]),
        quat_wb=np.array([1.0, 0.0, 0.0, 0.0]),
        lin_vel_b=np.zeros(3), ang_vel_b=np.zeros(3))


def test_one_full_hardware_tick_runs_without_any_simulator():
    """§8 item 1 verbatim: assembler -> EKF -> scan loop -> controller, one tick, no sim."""
    asm = TopicSampleAssembler(TankCalib())
    est = WallFrameStateEstimator()
    ctl = RecordingController()
    scan, mpc = _cfgs()
    loop = WallScanControlLoop(ctl, scan, mpc, horizon=30)

    # sensor messages arrive (the rclpy callbacks' job)
    asm.feed_dvl(0.05, 0.0, 0.0, 0.0)
    asm.feed_imu(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    asm.feed_depth(2.0, 0.0)            # 2 m below surface -> z = 8 m
    asm.feed_wall_range(1.5, 0.0)
    asm.feed_ukfm(4.5, 0.0, 0.0, 0.0)   # on the +x axis, nose on the radial

    sample = asm.assemble(0.02)
    assert sample is not None and len(sample.ukfm) == 3
    r0, phi0, theta0 = sample.ukfm
    est.reset(r0=r0, phi0=phi0, theta0=theta0)
    veh = est.step(sample, dt=0.02)

    loop.reset(veh)
    out = loop.step(veh, s_hat=est.s_hat, theta_hat=est.theta_hat)

    assert out.u_cmd.shape == (6,)
    ref = ctl.refs[-1]
    assert ref.z_ref.shape == (31,), "horizon-length preview"
    assert np.isfinite(ref.z_ref).all() and np.isfinite(ref.s_ref).all()
    assert ref.phase == 0 and loop.phase == 0
    assert ref.theta_anchor == pytest.approx(est.theta_hat)
    assert ref.s_anchor == pytest.approx(est.s_hat)


def test_phase_machine_advances_descend_to_sway_on_a_scripted_dive():
    """Drive the loop with a vehicle that tracks the ramped z_ref; DESCEND must hand over
    to SWAY at the bottom with a bumped s_ref, mirroring _sim_loop behaviour."""
    ctl = RecordingController()
    scan, mpc = _cfgs()
    loop = WallScanControlLoop(ctl, scan, mpc)
    loop.reset(_veh(z=1.05))  # start just above the bottom target so DESCEND is short

    z, s = 1.05, 0.0
    phases = [loop.phase]
    for _ in range(200):
        loop.step(_veh(z=z), s_hat=s, theta_hat=s / 6.0)
        z_ref, s_ref = loop.refs
        z += np.clip(z_ref - z, -0.02, 0.02)  # perfectly obedient plant
        s += np.clip(s_ref - s, -0.01, 0.01)
        phases.append(loop.phase)

    assert phases[0] == 0 and 1 in phases, f"never reached SWAY: phases={set(phases)}"
    # entering SWAY bumps the state machine's TARGET by sway_step; the preview's s_ref is
    # the RAMP toward it (ref_step_s per tick), so it starts near 0 and climbs.
    assert float(loop.sm.s_ref[0]) == pytest.approx(scan.sway_step, abs=0.05)
    first_sway = phases.index(1)
    later = min(first_sway + 50, len(ctl.refs) - 1)
    assert ctl.refs[later].s_ref[0] > ctl.refs[first_sway].s_ref[0], \
        "the ramped s_ref must climb toward the sway target"


def test_reset_reanchors_at_the_current_depth_and_zeroes_progress():
    ctl = RecordingController()
    scan, mpc = _cfgs()
    loop = WallScanControlLoop(ctl, scan, mpc)
    loop.reset(_veh(z=7.7))
    assert loop.phase == 0 and loop.cycles == 0
    assert float(loop.sm.z_ramp[0]) == pytest.approx(7.7)
    loop.step(_veh(z=7.7), s_hat=0.0, theta_hat=0.0)
    z_ref, _ = loop.refs
    assert abs(z_ref - 7.7) < 0.1, "ramped reference starts from the reset depth"
