# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""RC-WMPC: ctx feature contract, checkpoint round-trip, and the conditioned controller.

Fake-solver tests only (no acados/casadi) — same pattern as test_ssi_mpc / test_nmpc_adapters.
"""

import numpy as np
import pytest
import torch

from marinelab.algorithms.diff_wmpc import RC_CTX_DIM, WeightPolicy, policy_features, rc_context
from marinelab.control.types import ScanReference, VehicleState
from marinelab.tasks.pkrc_wallscan.mpc_reference import NE, WallScanMPCCfg
from marinelab.third_party.ssi_mpc_gpl.rc_wmpc_controller import RCWMPCController


# ---------------------------------------------------------------- feature contract

def test_policy_features_ctx_appends_after_base_layout():
    e = torch.arange(NE, dtype=torch.float32)
    base = policy_features(e, 1.0)
    ctx = torch.full((RC_CTX_DIM,), 0.25)
    feat = policy_features(e, 1.0, ctx=ctx)
    assert base.numel() == NE + 2
    assert feat.numel() == NE + 2 + RC_CTX_DIM
    # ctx=None must reproduce the pre-RC layout bit-for-bit, and ctx must only append
    torch.testing.assert_close(feat[: NE + 2], base)
    torch.testing.assert_close(feat[NE + 2:], ctx)


def test_rc_context_bounds_sign_and_first_tick_nan():
    ctx = rc_context(np.array([100.0, -100.0, 0.0]), pred_err=1e9, alpha_norm=1e9)
    assert ctx.shape == (RC_CTX_DIM,)
    assert (ctx.abs() <= 1.0).all()  # tanh-bounded: pathological learner cannot blow the input
    assert ctx[0] > 0.99 and ctx[1] < -0.99 and ctx[2] == 0.0  # sign/direction preserved
    # the learner's first tick reports pred_err = nan; that must contribute 0, not NaN
    ctx0 = rc_context(np.zeros(3), pred_err=float("nan"), alpha_norm=0.0)
    assert torch.isfinite(ctx0).all() and ctx0[3] == 0.0
    with pytest.raises(ValueError):
        rc_context(np.zeros(2), pred_err=0.0, alpha_norm=0.0)


def test_weight_policy_ctx_dim_roundtrip_and_legacy():
    torch.manual_seed(0)
    p = WeightPolicy(NE + 2 + RC_CTX_DIM, NE, 6, history_len=2, ctx_dim=RC_CTX_DIM)
    rebuilt = WeightPolicy.from_state_dict({"policy": p.state_dict()}, NE, 6)
    assert int(rebuilt.ctx_dim) == RC_CTX_DIM
    assert rebuilt.feat_dim == NE + 2 + RC_CTX_DIM and rebuilt.history_len == 2
    # a pre-RC checkpoint (no ctx_dim buffer) rebuilds a ctx-less policy
    legacy = WeightPolicy(NE + 2, NE, 6)
    rebuilt_legacy = WeightPolicy.from_state_dict({"policy": legacy.state_dict()}, NE, 6)
    assert int(rebuilt_legacy.ctx_dim) == 0 and rebuilt_legacy.feat_dim == NE + 2


# ---------------------------------------------------------------- controller

class FakeMPC:
    nu = 6
    n_pglobal = NE + 6

    def __init__(self):
        self.d_worlds = []
        self.weights = []

    def param_matrix(self, ref, theta_anchor, s_anchor, d_world=None):
        self.d_worlds.append(None if d_world is None else np.array(d_world))
        return np.zeros((len(ref["z_ref"]), 10))

    def solve(self, x0, P, weights, *, want_sensitivity=True, init_state_traj=False):
        self.weights.append(np.array(weights))
        u0 = np.full(6, 4.0)
        return {"u0": u0, "u0_cmd": u0 / 40.0, "status": 0}


def _policy(ctx_dim):
    torch.manual_seed(0)
    p = WeightPolicy(NE + 2 + ctx_dim, NE, 6, history_len=1, ctx_dim=ctx_dim)
    # the head is deliberately near-constant at init (gain 0.01); amplify it so the test
    # measures the ctx PATHWAY, not the init scale
    torch.nn.init.orthogonal_(p.head.weight, gain=1.0)
    return p


def _controller(fake, policy):
    return RCWMPCController(mpc=fake, mpc_cfg=WallScanMPCCfg(), policy=policy,
                            step_dt=0.02, mass=22.8, max_thrust=40.0,
                            predict_fn=lambda x, u, dt: np.asarray(x, float),
                            ssi_n_rf=8, ssi_lr=0.3, ssi_seed=0)


def _state():
    x = np.zeros(13)
    x[0], x[2], x[3] = 4.5, 5.0, 1.0
    return VehicleState.from_x13(x)


def _ref():
    return ScanReference.frozen(30, z_ref=5.0, s_ref=0.0)


def test_rc_controller_ctx_reaches_the_policy():
    """Identical state/ref sequence, different learner state => different weights."""
    pol = _policy(RC_CTX_DIM)
    a = _controller(FakeMPC(), pol)
    a.reset(_state())
    out = a.step(_state(), _ref())
    assert a.name == "rc"
    assert "rc_pred_err" in out.aux and "ssi_residual_b" in out.aux

    b = _controller(FakeMPC(), pol)
    b.reset(_state())
    b.learner.alpha[:] = 50.0  # saturates the ctx block (d_world + alpha_norm channels)
    b.step(_state(), _ref())
    w_a, w_b = a._mpc.weights[0], b._mpc.weights[0]
    assert np.abs(w_a - w_b).max() > 1e-6  # the conditioning edge is live
    # and the model channel still works: the residual reached the OCP as d_world
    assert np.abs(b._mpc.d_worlds[0]).max() > 0


def test_rc_controller_weights_change_with_learner_convergence():
    """Within one run, the SAME vehicle situation gets different weights as alpha evolves —
    the behavior a fixed-weight SSI cannot express."""
    fake = FakeMPC()
    ctl = _controller(fake, _policy(RC_CTX_DIM))
    ctl.reset(_state())
    ctl.step(_state(), _ref())
    w_first = fake.weights[0].copy()
    for k in range(1, 40):  # growing unmodelled heave velocity -> learner adapts
        x = np.zeros(13)
        x[0], x[2], x[3], x[9] = 4.5, 5.0, 1.0, 0.05 * k
        ctl.step(VehicleState.from_x13(x), _ref())
    ctl._policy.reset_history()
    fake.weights.clear()
    ctl.step(_state(), _ref())  # same situation as step 1, different adaptation state
    assert np.abs(fake.weights[0] - w_first).max() > 1e-6


def test_rc_controller_naive_fallback_is_ablation_a1():
    """A ctx-less (pre-RC) checkpoint degrades to naive stacking and says so in the name."""
    fake = FakeMPC()
    ctl = _controller(fake, _policy(0))
    assert ctl.name == "rc_naive"
    ctl.reset(_state())
    out = ctl.step(_state(), _ref())
    assert "rc_pred_err" not in out.aux  # no ctx telemetry: nothing is conditioned
    assert fake.weights[0].shape == (NE + 6,)
    assert fake.d_worlds[0] is not None  # the SSI half still injects


def test_rc_controller_rejects_foreign_ctx_width():
    torch.manual_seed(0)
    odd = WeightPolicy(NE + 2 + 3, NE, 6, history_len=1, ctx_dim=3)
    ctl = _controller(FakeMPC(), odd)
    ctl.reset(_state())
    with pytest.raises(ValueError):
        ctl.step(_state(), _ref())


def test_rc_controller_reset_clears_policy_history():
    ctl = _controller(FakeMPC(), _policy(RC_CTX_DIM))
    ctl.reset(_state())
    ctl.step(_state(), _ref())
    assert len(ctl._policy._history) > 0
    ctl.reset(_state())
    assert len(ctl._policy._history) == 0
