# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Adapter tests with a fake solver: no acados/casadi needed, torch only.

The fake mimics ``WallScanMPC``'s call surface (``nu``, ``n_pglobal``, ``param_matrix``,
``solve``) and records what it was given, so these tests pin the adapter contract: state
layout, reference plumbing, warm-start flags, weight sourcing, and stats accounting.
"""

import json

import numpy as np
import pytest
import torch

from marinelab.control.diff_wmpc_ctrl import DiffWMPCController
from marinelab.control.fixed_nmpc import FixedWeightNMPC
from marinelab.control.types import ScanReference, VehicleState
from marinelab.tasks.pkrc_wallscan.mpc_controller import DEFAULT_WERR, DEFAULT_WU
from marinelab.tasks.pkrc_wallscan.mpc_reference import NE, WallScanMPCCfg


class FakeMPC:
    def __init__(self, nu=6, status=0, u0=None):
        self.nu = nu
        self.n_pglobal = NE + nu
        self.status = status
        self.u0 = np.full(nu, 10.0) if u0 is None else np.asarray(u0, float)
        self.calls = []

    def param_matrix(self, ref, theta_anchor, s_anchor, d_world=None):
        n_stages = len(ref["z_ref"]) - 1
        P = np.zeros((n_stages + 1, 10))
        P[:, 1] = ref["z_ref"]
        P[:, 5], P[:, 6] = theta_anchor, s_anchor
        return P

    def solve(self, x0, P, weights, *, want_sensitivity=True, init_state_traj=False):
        self.calls.append({"x0": np.array(x0), "P": P, "weights": np.array(weights),
                           "want_sensitivity": want_sensitivity, "init_state_traj": init_state_traj})
        return {"u0": self.u0, "u0_cmd": np.clip(self.u0 / 40.0, -1, 1), "status": self.status}


def _state():
    return VehicleState.from_x13(np.r_[4.5, 0.0, 5.0, 1.0, 0, 0, 0, np.zeros(6)])


def _ref():
    return ScanReference.frozen(30, z_ref=5.0, s_ref=0.0, theta_anchor=0.2, s_anchor=1.0)


def test_fixed_nmpc_contract():
    fake = FakeMPC()
    ctl = FixedWeightNMPC(mpc=fake)
    ctl.reset(_state())
    out1 = ctl.step(_state(), _ref())
    out2 = ctl.step(_state(), _ref())

    # default weights are DEFAULT_WERR ++ DEFAULT_WU per thruster
    np.testing.assert_allclose(fake.calls[0]["weights"][:NE], DEFAULT_WERR)
    np.testing.assert_allclose(fake.calls[0]["weights"][NE:], DEFAULT_WU)
    # x0 is the 13-D layout, references and anchors reach param_matrix
    np.testing.assert_allclose(fake.calls[0]["x0"], _state().to_x13())
    assert fake.calls[0]["P"][0, 5] == pytest.approx(0.2)
    # cold start only on the first solve after reset, no gradients requested
    assert fake.calls[0]["init_state_traj"] is True
    assert fake.calls[1]["init_state_traj"] is False
    assert fake.calls[0]["want_sensitivity"] is False
    ctl.reset(_state())
    ctl.step(_state(), _ref())
    assert fake.calls[2]["init_state_traj"] is True
    # outputs: normalized command, newton command, stats accounting
    np.testing.assert_allclose(out1.u_cmd, 0.25)
    np.testing.assert_allclose(out1.u_newton, 10.0)
    assert out2.status == 0
    assert ctl.stats.n_steps == 3 and ctl.stats.n_fail == 0 and ctl.stats.n_saturated == 0


def test_fixed_nmpc_counts_failures_and_saturation():
    fake = FakeMPC(status=2, u0=np.full(6, 40.0))  # saturated command, failed solve
    ctl = FixedWeightNMPC(mpc=fake)
    ctl.step(_state(), _ref())
    s = ctl.stats.summary()
    assert s["fail_frac"] == 1.0 and s["saturated_frac"] == 1.0


def test_weights_from_tuning_json(tmp_path):
    werr = list(range(1, NE + 1))
    path = tmp_path / "best_params.json"
    path.write_text(json.dumps({"werr": werr, "wu": [0.5]}))
    ctl = FixedWeightNMPC(mpc=FakeMPC(), params_json=str(path))
    np.testing.assert_allclose(ctl.weights[:NE], werr)
    np.testing.assert_allclose(ctl.weights[NE:], 0.5)  # scalar wu broadcast to nu
    assert ctl.name == "bo"


def test_weight_length_mismatch_rejected():
    with pytest.raises(ValueError):
        FixedWeightNMPC(mpc=FakeMPC(), werr=[1.0, 2.0])


def test_diff_wmpc_policy_drives_weights(tmp_path):
    from marinelab.algorithms.diff_wmpc import WeightPolicy

    fake = FakeMPC()
    torch.manual_seed(0)
    policy = WeightPolicy(NE + 2, NE, fake.nu)
    ckpt = tmp_path / "policy.pt"
    torch.save({"policy": policy.state_dict()}, ckpt)

    ctl = DiffWMPCController(mpc=fake, mpc_cfg=WallScanMPCCfg(), ckpt_path=str(ckpt))
    ctl.reset(_state())
    out = ctl.step(_state(), _ref())

    w = fake.calls[0]["weights"]
    assert w.shape == (NE + fake.nu,)
    # WeightPolicy outputs are bounded by construction
    assert (w[:NE] >= 0.1 - 1e-6).all() and (w[:NE] <= 5000.0 + 1e-6).all()
    assert (w[NE:] >= 5e-3 - 1e-6).all() and (w[NE:] <= 5.0 + 1e-6).all()
    np.testing.assert_allclose(out.aux["werr"], w[:NE])
    assert ctl.name == "diff"

    # reset clears the policy's feature history
    ctl.step(_state(), _ref())
    assert len(ctl._policy._history) > 0
    ctl.reset(_state())
    assert len(ctl._policy._history) == 0
