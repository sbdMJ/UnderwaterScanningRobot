# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""SSI-MPC port (GPL-isolated): learner convergence, episode semantics, d_world injection."""

import numpy as np
import pytest

from marinelab.control.types import ScanReference, VehicleState
from marinelab.tasks.pkrc_wallscan.mpc_reference import NE
from marinelab.third_party.ssi_mpc_gpl.rff_learner import RFFOnlineLearner
from marinelab.third_party.ssi_mpc_gpl.ssi_controller import SSIMPCController


def test_learner_converges_to_constant_residual():
    """Zero nominal dynamics + constant unmodelled acceleration: OGD must identify it."""
    dt, c = 0.02, 0.8
    learner = RFFOnlineLearner(state_dim=2, u_dim=1, target_mask=[1], input_mask=[0, 1, 2],
                               n_rf=50, lr=0.5, kernel_std=1.0, seed=0)
    predict_nominal = lambda x, u, dt: np.asarray(x, float)  # noqa: E731
    rng = np.random.default_rng(1)
    x = np.zeros(2)
    for _ in range(2000):
        u = rng.uniform(-1, 1, size=1)
        learner.record_control(x, u)
        x = x + dt * np.array([0.0, c])  # true plant: residual accel c on dim 1
        x[0] = rng.uniform(-1, 1)  # excitation on the feature inputs
        learner.update(x, dt, predict_nominal)
    # OGD is deliberately slow (no-regret step, not RLS): ~6% error after 2000 steps
    assert float(learner.residual_now(x)[0]) == pytest.approx(c, rel=0.1)


def test_learner_episode_reset_keeps_alpha():
    learner = RFFOnlineLearner(state_dim=2, u_dim=1, target_mask=[1], input_mask=[0, 1, 2],
                               n_rf=8, lr=0.5, seed=0)
    learner.record_control(np.zeros(2), np.zeros(1))
    learner.update(np.array([0.0, 0.1]), 0.02, lambda x, u, dt: x)
    alpha_before = learner.alpha.copy()
    assert np.abs(alpha_before).max() > 0
    learner.reset_episode()
    # transition dropped: the next update scores nothing...
    out = learner.update(np.array([0.0, 0.2]), 0.02, lambda x, u, dt: x)
    np.testing.assert_allclose(out, alpha_before)  # ...but the learned model persists


class FakeMPC:
    nu = 6
    n_pglobal = NE + 6

    def __init__(self):
        self.d_worlds = []

    def param_matrix(self, ref, theta_anchor, s_anchor, d_world=None):
        self.d_worlds.append(None if d_world is None else np.array(d_world))
        return np.zeros((len(ref["z_ref"]), 10))

    def solve(self, x0, P, weights, *, want_sensitivity=True, init_state_traj=False):
        u0 = np.full(6, 4.0)
        return {"u0": u0, "u0_cmd": u0 / 40.0, "status": 0}


def _controller(fake):
    return SSIMPCController(mpc=fake, step_dt=0.02, mass=22.8, max_thrust=40.0,
                            predict_fn=lambda x, u, dt: np.asarray(x, float),
                            ssi_n_rf=16, ssi_lr=0.3, ssi_seed=0)


def _state(vz=0.0):
    x = np.zeros(13)
    x[0], x[2], x[3] = 4.5, 5.0, 1.0  # r=4.5, z=5, identity quat
    x[9] = vz
    return VehicleState.from_x13(x)


def test_controller_injects_learned_residual_as_d_world():
    fake = FakeMPC()
    ctl = _controller(fake)
    ctl.reset(_state())
    ref = ScanReference.frozen(30, z_ref=5.0, s_ref=0.0)

    out = ctl.step(_state(), ref)
    assert "ssi_residual_b" in out.aux and out.aux["ssi_alpha_norm"] == 0.0
    np.testing.assert_allclose(fake.d_worlds[0], 0.0)  # nothing learned yet

    # measured heave velocity keeps growing vs the frozen nominal prediction
    for k in range(1, 60):
        ctl.step(_state(vz=0.02 * k), ref)
    assert ctl.learner.alpha.any()
    d = fake.d_worlds[-1]
    assert d is not None and np.abs(d).max() > 0
    # identity attitude: d_world = mass * residual_b, dominant on the z (heave) axis
    r_b = ctl.learner.residual_now(_state(vz=0.02 * 59).to_x13())
    np.testing.assert_allclose(d, 22.8 * r_b, rtol=1e-9)
    assert np.argmax(np.abs(d)) == 2


def test_controller_reconfigure_and_identity():
    fake = FakeMPC()
    ctl = _controller(fake)
    assert ctl.name == "ssi"
    ref = ScanReference.frozen(30, z_ref=5.0, s_ref=0.0)
    ctl.reset(_state())
    for k in range(10):
        ctl.step(_state(vz=0.1 * k), ref)
    assert ctl.learner.alpha.any()
    ctl.reconfigure(lr=0.05, kernel_std=2.0)
    assert not ctl.learner.alpha.any()  # fresh learner
    assert ctl.learner.lr == pytest.approx(0.05)
