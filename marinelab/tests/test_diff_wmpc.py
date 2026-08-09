# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Diff-WMPC learner (algorithms/diff_wmpc.py).

The acados half is not needed here: the learner only consumes the sensitivity matrices, so
hand-made Jacobians exercise every path natively. What these tests actually protect is the
gradient CHAIN — a transposed sensitivity or a dropped minus sign would still train, just
toward the wrong weights, and no runtime error would ever say so.
"""

import math

import numpy as np
import pytest
import torch

from marinelab.algorithms.diff_wmpc import (
    DiffWMPCLearner,
    WallScanLossCfg,
    WeightPolicy,
    wallscan_loss,
)

NE, NU, FEAT = 12, 6, 14
NPG = NE + NU
NX = 13


def _policy(**kw):
    torch.manual_seed(0)
    return WeightPolicy(FEAT, NE, NU, history_len=2, hidden=16, **kw)


NODES = (5, 10, 20)


def _mpc_out(status=0, umax=0.3, seed=0, nodes=NODES):
    """Multi-node solver output, as WallScanMPC.solve now returns."""
    rng = np.random.default_rng(seed)
    return {
        "status": status,
        "u0": rng.normal(size=NU) * 5.0,
        "u0_cmd": np.full(NU, umax),
        "x_nodes": {k: rng.normal(size=NX) for k in nodes},
        "sens_x_nodes": {k: rng.normal(size=(NX, NPG)) for k in nodes},
        "sens_u": rng.normal(size=(NU, NPG)),
        "sens_node": max(nodes),
    }


def _single_node_out(status=0, umax=0.3, seed=0, node=20):
    """Legacy shape: only sens_x/x_node. The learner must still accept it."""
    rng = np.random.default_rng(seed)
    return {
        "status": status,
        "u0": rng.normal(size=NU) * 5.0,
        "u0_cmd": np.full(NU, umax),
        "x_node": rng.normal(size=NX),
        "sens_x": rng.normal(size=(NX, NPG)),
        "sens_u": rng.normal(size=(NU, NPG)),
        "sens_node": node,
    }


def _error_fn(x, node=0):
    """Stand-in for mpc_reference.wallscan_errors. The node argument shifts the reference, as
    a real per-node reference would, so a caller that ignores it is detectable."""
    off = 0.01 * node
    return torch.stack([x[i % NX] * (1.0 + 0.1 * i) - off for i in range(NE)])


# ---------------------------------------------------------------------------
# WeightPolicy
# ---------------------------------------------------------------------------


def test_policy_starts_at_the_requested_weights():
    """Training must start from the hand-tuned controller, not a random one."""
    werr = np.array([40, 40, 40, 10, 5, 5, 20, 20, 2000, 2000, 0.5, 0.5], float)
    wu = np.full(NU, 0.01)
    p = _policy(werr_init=werr, wu_init=wu)
    w = p(torch.zeros(FEAT))
    assert torch.allclose(w[:NE], torch.as_tensor(werr, dtype=torch.float32), rtol=0.02)
    assert torch.allclose(w[NE:], torch.as_tensor(wu, dtype=torch.float32), rtol=0.02)


def test_roll_weight_of_2000_is_inside_the_bounds():
    """The measured sway-tilt fix needs w_roll ~2000; a 500 ceiling would exclude it."""
    p = _policy()
    assert float(p.werr_ub.min()) >= 2000.0
    assert float(p.werr_lb.max()) <= 0.5, "tuned angular-rate weights are 0.5; a 1.0 floor clamps them"
    w = _policy(werr_init=np.full(NE, 2000.0))(torch.zeros(FEAT))
    assert float(w[:NE].min()) > 1500.0


def test_policy_output_always_respects_bounds():
    p = _policy()
    for scale in (0.0, 1.0, 1e3, -1e3):
        w = p(torch.full((FEAT,), scale))
        assert (w[:NE] >= p.werr_lb - 1e-6).all() and (w[:NE] <= p.werr_ub + 1e-6).all()
        assert (w[NE:] >= p.wu_lb - 1e-9).all() and (w[NE:] <= p.wu_ub + 1e-9).all()


def test_policy_is_differentiable_wrt_parameters():
    p = _policy()
    w = p(torch.randn(FEAT))
    g = torch.autograd.grad(w.sum(), [q for q in p.parameters()], allow_unused=True)
    assert any(x is not None and torch.isfinite(x).all() and x.abs().sum() > 0 for x in g)


def test_history_is_seeded_then_shifts():
    p = _policy()
    p.reset_history()
    p(torch.zeros(FEAT))
    assert len(p._history) == p.history_len
    p(torch.ones(FEAT))
    assert torch.allclose(p._history[0], torch.ones(FEAT))


def test_reset_history_forgets_the_previous_segment():
    p = _policy()
    p(torch.ones(FEAT))
    p.reset_history()
    assert p._history == []


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_loss_is_zero_only_at_zero_error_and_zero_control():
    cfg = WallScanLossCfg()
    assert float(wallscan_loss(torch.zeros(NE), torch.zeros(NU), cfg)) == pytest.approx(0.0)
    assert float(wallscan_loss(torch.zeros(NE), torch.full((NU,), 10.0), cfg)) > 0.0


def test_control_term_uses_the_normalized_command():
    """Raw newtons would dwarf the tracking terms -> the 'learn to do nothing' pathology."""
    cfg = WallScanLossCfg(l_u=1.0, max_thrust=40.0)
    at_full = float(wallscan_loss(torch.zeros(NE), torch.full((NU,), 40.0), cfg))
    assert at_full == pytest.approx(1.0, abs=1e-5), "|u|=max must cost exactly l_u, not 1600*l_u"


def test_heading_and_rollpitch_are_weighted_as_the_stated_priorities():
    cfg = WallScanLossCfg()
    e = torch.zeros(NE)
    e[6] = 0.1
    head = float(wallscan_loss(e, torch.zeros(NU), cfg))
    e = torch.zeros(NE)
    e[1] = 0.1
    depth = float(wallscan_loss(e, torch.zeros(NU), cfg))
    assert head > depth, "heading is the problem this port exists to fix"


def test_loss_gradients_flow_to_state_and_control():
    x = torch.randn(NX, requires_grad=True)
    u = torch.randn(NU, requires_grad=True)
    loss = wallscan_loss(_error_fn(x), u, WallScanLossCfg())
    gx, gu = torch.autograd.grad(loss, (x, u))
    assert torch.isfinite(gx).all() and torch.isfinite(gu).all()
    assert gx.abs().sum() > 0 and gu.abs().sum() > 0


# ---------------------------------------------------------------------------
# Learner: the gradient chain
# ---------------------------------------------------------------------------


def test_multi_node_gradient_equals_autograd_on_the_first_order_surrogate():
    """Exact identity for the SUMMED chain — the core of the multi-node fix.

    The accumulated parameter gradient must equal d/dtheta of

        L_hat = sum_k loss(err(x_k + Sx_k @ dw, k), u0 + Su @ dw)     [u0 term on node min(k)]

    at dw = 0. This is what catches a per-node sign error, a mismatched node<->sensitivity
    pairing, or the control term being double-counted across nodes.
    """
    torch.manual_seed(2)
    p = _policy()
    out = _mpc_out(seed=5)
    feat = torch.randn(FEAT)
    cfg = WallScanLossCfg()

    p.reset_history()
    with torch.no_grad():
        w0 = p(feat).clone()

    lr = DiffWMPCLearner(p, n_pglobal=NPG, batch_size=99, grad_clip=1e9)
    p.reset_history()
    assert lr.learn_step(lr.compute_weights(feat), out, _error_fn, cfg) is not None
    g_chain = {n: q.grad.detach().clone() for n, q in p.named_parameters() if q.grad is not None}
    assert g_chain

    p.zero_grad(set_to_none=True)
    p.reset_history()
    dw = p(feat) - w0
    Su = torch.as_tensor(out["sens_u"], dtype=torch.float32)
    u0 = torch.as_tensor(out["u0"], dtype=torch.float32)
    first = min(out["sens_x_nodes"])
    surrogate = 0.0
    for k in sorted(out["sens_x_nodes"]):
        Sx = torch.as_tensor(out["sens_x_nodes"][k], dtype=torch.float32)
        x_k = torch.as_tensor(out["x_nodes"][k], dtype=torch.float32)
        u_k = u0 + Su @ dw if k == first else u0
        surrogate = surrogate + wallscan_loss(_error_fn(x_k + Sx @ dw, k), u_k, cfg)
    surrogate.backward()

    for name, g in g_chain.items():
        direct = dict(p.named_parameters())[name].grad
        assert torch.allclose(g, direct, atol=1e-5, rtol=1e-3), name


def test_multi_node_loss_is_the_sum_of_the_per_node_losses():
    p = _policy()
    lr = DiffWMPCLearner(p, n_pglobal=NPG, batch_size=99, grad_clip=1e9)
    out = _mpc_out(seed=7)
    cfg = WallScanLossCfg()
    total = lr.learn_step(lr.compute_weights(torch.zeros(FEAT)), out, _error_fn, cfg)
    u0 = torch.as_tensor(out["u0"], dtype=torch.float32)
    first = min(out["sens_x_nodes"])
    expect = 0.0
    for k in sorted(out["x_nodes"]):
        x_k = torch.as_tensor(out["x_nodes"][k], dtype=torch.float32)
        expect += float(wallscan_loss(_error_fn(x_k, k), u0 if k == first else u0 * 0 + u0, cfg))
    assert total == pytest.approx(expect, rel=1e-5)


def test_control_effort_is_counted_once_not_once_per_node():
    """Otherwise l_u would be silently multiplied by the number of nodes."""
    p = _policy()
    cfg = WallScanLossCfg(l_u=1.0, max_thrust=1.0)
    zero_err = lambda x, node: torch.zeros(NE)  # noqa: E731 - isolate the control term
    out = _mpc_out(seed=9)
    out["u0"] = np.ones(NU)
    lr = DiffWMPCLearner(p, n_pglobal=NPG, batch_size=99, grad_clip=1e9)
    total = lr.learn_step(lr.compute_weights(torch.zeros(FEAT)), out, zero_err, cfg)
    assert total == pytest.approx(1.0, abs=1e-5), "|u|=max costs l_u once, not len(nodes)*l_u"


def test_node_weights_scale_each_node():
    p = _policy()
    lr = DiffWMPCLearner(p, n_pglobal=NPG, batch_size=99, grad_clip=1e9)
    out = _mpc_out(seed=11)
    cfg = WallScanLossCfg()
    full = lr.learn_step(lr.compute_weights(torch.zeros(FEAT)), out, _error_fn, cfg)
    p.zero_grad(set_to_none=True)
    p.reset_history()
    only_last = lr.learn_step(lr.compute_weights(torch.zeros(FEAT)), out, _error_fn, cfg,
                              node_weights={k: (1.0 if k == max(NODES) else 0.0) for k in NODES})
    assert only_last < full, "zeroing nodes must reduce the summed loss"


def test_error_fn_receives_the_node_index():
    """A caller that reuses the stage-0 reference for every node would reintroduce myopia."""
    seen = []

    def spy(x, node):
        seen.append(node)
        return _error_fn(x, node)

    p = _policy()
    lr = DiffWMPCLearner(p, n_pglobal=NPG, batch_size=99, grad_clip=1e9)
    lr.learn_step(lr.compute_weights(torch.zeros(FEAT)), _mpc_out(), spy, WallScanLossCfg())
    assert sorted(seen) == sorted(NODES)


def test_single_node_output_is_still_accepted():
    """Back-compat with the Phase 3 baseline's single-node solve."""
    p = _policy()
    lr = DiffWMPCLearner(p, n_pglobal=NPG, batch_size=1, grad_clip=1e9)
    loss = lr.learn_step(lr.compute_weights(torch.zeros(FEAT)), _single_node_out(),
                         _error_fn, WallScanLossCfg())
    assert loss is not None and math.isfinite(loss) and lr.n_updates == 1


def test_per_entry_floor_prevents_the_radial_weight_from_collapsing():
    """Direct guard against the measured failure: w_radial fell to 2-3 and tracking died."""
    lb = np.full(NE, 0.1)
    lb[0] = 30.0  # radial
    torch.manual_seed(0)
    p = WeightPolicy(FEAT, NE, NU, history_len=2, hidden=16, werr_lb=lb,
                     werr_init=np.full(NE, 40.0))
    for scale in (0.0, 5.0, -5.0, 50.0):
        w = p(torch.full((FEAT,), scale))
        assert float(w[0]) >= 30.0 - 1e-4, f"radial weight escaped its floor at {scale}"


def test_learn_step_skips_a_failed_solve():
    lr = DiffWMPCLearner(_policy(), n_pglobal=NPG, batch_size=1)
    w = lr.compute_weights(torch.zeros(FEAT))
    assert lr.learn_step(w, _mpc_out(status=4), _error_fn, WallScanLossCfg()) is None
    assert lr.n_skipped_status == 1 and lr.n_updates == 0


def test_learn_step_skips_a_saturated_solve():
    """At a thrust bound the sensitivity is exactly zero — a wrong lesson, not a noisy one."""
    lr = DiffWMPCLearner(_policy(), n_pglobal=NPG, batch_size=1)
    w = lr.compute_weights(torch.zeros(FEAT))
    assert lr.learn_step(w, _mpc_out(umax=0.999), _error_fn, WallScanLossCfg()) is None
    assert lr.n_skipped_sat == 1 and lr.n_updates == 0


def test_learn_step_skips_missing_sensitivities():
    lr = DiffWMPCLearner(_policy(), n_pglobal=NPG, batch_size=1)
    out = _mpc_out()
    out["sens_x_nodes"] = None
    w = lr.compute_weights(torch.zeros(FEAT))
    assert lr.learn_step(w, out, _error_fn, WallScanLossCfg()) is None
    assert lr.n_skipped_status == 1


def test_learn_step_skips_non_finite_gradients():
    lr = DiffWMPCLearner(_policy(), n_pglobal=NPG, batch_size=1)
    out = _mpc_out()
    out["sens_x_nodes"][NODES[0]][0, 0] = np.inf
    w = lr.compute_weights(torch.zeros(FEAT))
    assert lr.learn_step(w, out, _error_fn, WallScanLossCfg()) is None
    assert lr.n_skipped_nan == 1


def test_adam_steps_once_per_batch_not_once_per_sample():
    lr = DiffWMPCLearner(_policy(), n_pglobal=NPG, batch_size=4)
    for i in range(9):
        w = lr.compute_weights(torch.randn(FEAT))
        lr.learn_step(w, _mpc_out(seed=i), _error_fn, WallScanLossCfg())
    assert lr.n_updates == 2, "9 accepted steps at batch 4 -> 2 optimizer steps"


def test_skipped_steps_do_not_count_toward_the_batch():
    lr = DiffWMPCLearner(_policy(), n_pglobal=NPG, batch_size=2)
    for _ in range(5):
        w = lr.compute_weights(torch.zeros(FEAT))
        lr.learn_step(w, _mpc_out(status=4), _error_fn, WallScanLossCfg())
    assert lr.n_updates == 0 and lr.n_skipped == 5


def test_gradient_clipping_bounds_the_pglobal_gradient():
    lr = DiffWMPCLearner(_policy(), n_pglobal=NPG, batch_size=1, grad_clip=1e-8)
    before = [q.detach().clone() for q in lr.policy.parameters()]
    w = lr.compute_weights(torch.randn(FEAT))
    lr.learn_step(w, _mpc_out(), _error_fn, WallScanLossCfg())
    moved = max(float((a - b).abs().max()) for a, b in zip(before, lr.policy.parameters()))
    assert moved < 1e-2, "a tiny clip must produce a tiny parameter step"


def test_state_dict_round_trip():
    p = _policy()
    lr = DiffWMPCLearner(p, n_pglobal=NPG, batch_size=1)
    w = lr.compute_weights(torch.randn(FEAT))
    lr.learn_step(w, _mpc_out(), _error_fn, WallScanLossCfg())
    state = lr.state_dict()

    p2 = _policy(werr_init=np.full(NE, 7.0))
    lr2 = DiffWMPCLearner(p2, n_pglobal=NPG, batch_size=1)
    lr2.load_state_dict(state)
    f = torch.zeros(FEAT)
    p.reset_history()
    p2.reset_history()
    assert torch.allclose(p(f), p2(f), atol=1e-6)
    assert lr2.n_updates == lr.n_updates


def test_weight_bounds_round_trip_in_the_state_dict():
    """The bounds ARE part of the output mapping, so a checkpoint must carry them.

    Loading into a policy built with different bounds would otherwise produce different cost
    weights from the identical network, silently and with no error raised anywhere.
    """
    lb = np.full(NE, 0.1)
    lb[0] = 25.0
    torch.manual_seed(0)
    trained = WeightPolicy(FEAT, NE, NU, history_len=2, hidden=16, werr_lb=lb,
                           werr_init=np.full(NE, 40.0))
    assert "werr_lb" in trained.state_dict(), "bounds must be buffers, not bare attributes"

    torch.manual_seed(0)
    loaded = WeightPolicy(FEAT, NE, NU, history_len=2, hidden=16,  # DEFAULT bounds on purpose
                          werr_init=np.full(NE, 40.0))
    assert float(loaded.werr_lb[0]) != pytest.approx(25.0), "precondition: bounds differ"
    loaded.load_state_dict(trained.state_dict())
    assert float(loaded.werr_lb[0]) == pytest.approx(25.0)

    f = torch.randn(FEAT)
    trained.reset_history()
    loaded.reset_history()
    assert torch.allclose(trained(f), loaded(f), atol=1e-6)


# ---------------------------------------------------------------------------
# Policy input layout (FEAT_MODES / build_features)
#
# This is the contract between the trainer and the runner: they must assemble the SAME vector
# or a checkpoint silently produces different weights than it was trained to produce. The
# `preview` mode is the source paper's design (a look-ahead on the reference, no current
# error); `error_phase` is the legacy reactive one that was measured to collapse to a constant.
# ---------------------------------------------------------------------------
from marinelab.algorithms.diff_wmpc import (  # noqa: E402
    FEAT_MODES,
    N_PREVIEW,
    build_features,
    feature_dim,
    preview_stages,
)


def _ref(n_envs=3, n_stages=30):
    """[n_envs, n_stages+1] reference; each env distinct, so reading the wrong row shows up
    as a wrong value rather than as a silent pass."""
    base = torch.arange(n_stages + 1, dtype=torch.float32)
    return {
        "z_ref": torch.stack([base + 100 * e for e in range(n_envs)]),
        "s_ref": torch.stack([base + 200 * e for e in range(n_envs)]),
        "v_z_des": torch.stack([base + 300 * e for e in range(n_envs)]),
        "v_tan_des": torch.stack([base + 400 * e for e in range(n_envs)]),
    }


def test_feature_dim_matches_built_vector():
    ref, phase, e_now = _ref(), torch.tensor(2.0), torch.arange(NE, dtype=torch.float32)
    for mode in FEAT_MODES:
        got = build_features(mode, e_now=e_now, phase=phase, ref=ref, n_stages=30)
        assert got.shape == (feature_dim(mode, NE),), mode


def test_preview_mode_carries_no_current_error():
    """The paper's policy sees the upcoming reference only. If the error leaked in, changing it
    would change the features -- and the anticipation claim would be untestable."""
    ref = _ref()
    a = build_features("preview", e_now=torch.zeros(NE), phase=torch.tensor(0.0),
                       ref=ref, n_stages=30)
    b = build_features("preview", e_now=torch.full((NE,), 9.0), phase=torch.tensor(3.0),
                       ref=ref, n_stages=30)
    assert torch.equal(a, b)


def test_preview_reads_the_requested_env_row():
    """Per-env by construction: sharing one feature vector across envs running independent
    phases is the bug this argument exists to prevent."""
    ref = _ref()
    f0 = build_features("preview", e_now=torch.zeros(NE), phase=torch.tensor(0.0),
                        ref=ref, n_stages=30, env=0)
    f2 = build_features("preview", e_now=torch.zeros(NE), phase=torch.tensor(0.0),
                        ref=ref, n_stages=30, env=2)
    assert not torch.equal(f0, f2)
    # v_z_des row offset is 300 per env, and the first half of the vector is v_z_des
    assert torch.allclose(f2[:N_PREVIEW] - f0[:N_PREVIEW], torch.full((N_PREVIEW,), 600.0))


def test_preview_stages_are_look_ahead_only():
    ks = preview_stages(30)
    assert len(ks) == N_PREVIEW
    assert ks == sorted(ks) and ks[0] > 0, "stage 0 is where the vehicle already is"
    assert ks[-1] == 30, "the preview must reach the end of the horizon"


def test_preview_changes_when_the_upcoming_reference_changes():
    """The whole mechanism: an upcoming phase switch must move the features BEFORE it arrives."""
    ref = _ref()
    flat = {k: torch.zeros_like(v) for k, v in ref.items()}
    step = {k: v.clone() for k, v in flat.items()}
    step["v_z_des"][:, 20:] = -0.2          # a heave leg starting 1.0 s into the horizon
    a = build_features("preview", e_now=torch.zeros(NE), phase=torch.tensor(0.0),
                       ref=flat, n_stages=30)
    b = build_features("preview", e_now=torch.zeros(NE), phase=torch.tensor(0.0),
                       ref=step, n_stages=30)
    assert not torch.equal(a, b)


def test_both_mode_is_the_concatenation():
    ref, phase, e_now = _ref(), torch.tensor(1.0), torch.arange(NE, dtype=torch.float32)
    ep = build_features("error_phase", e_now=e_now, phase=phase, ref=ref, n_stages=30)
    pv = build_features("preview", e_now=e_now, phase=phase, ref=ref, n_stages=30)
    both = build_features("both", e_now=e_now, phase=phase, ref=ref, n_stages=30)
    assert torch.equal(both, torch.cat([ep, pv]))


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        build_features("lookahead", e_now=torch.zeros(NE), phase=torch.tensor(0.0),
                       ref=_ref(), n_stages=30)
    with pytest.raises(ValueError):
        feature_dim("lookahead", NE)


def test_policy_accepts_each_mode_width():
    ref, phase, e_now = _ref(), torch.tensor(0.0), torch.zeros(NE)
    for mode in FEAT_MODES:
        pol = WeightPolicy(feature_dim(mode, NE), NE, NU)
        w = pol(build_features(mode, e_now=e_now, phase=phase, ref=ref, n_stages=30))
        assert w.shape == (NE + NU,) and torch.isfinite(w).all()


def test_error_phase_mode_is_bit_identical_to_the_legacy_inline_build():
    """`checkpoints/dw_ekf/policy_final.pt` was trained against the inline expression the
    trainer used before build_features existed. If this drifts, that checkpoint's published
    numbers stop being reproducible."""
    ref, e_now = _ref(), torch.arange(NE, dtype=torch.float32)
    for ph in (0.0, 1.0, 2.0, 3.0):
        phase = torch.tensor(ph)
        legacy = torch.cat([e_now.cpu(), torch.stack([
            torch.sin(2 * math.pi * phase / 4), torch.cos(2 * math.pi * phase / 4)]).cpu()])
        got = build_features("error_phase", e_now=e_now, phase=phase, ref=ref, n_stages=30)
        assert torch.equal(got, legacy), ph


def test_sym_sway_makes_mirrored_sway_legs_identical():
    """SWAY_A and SWAY_B are the same motion mirrored, and weights multiply squared errors, so
    they must map to the same features once the symmetry is declared."""
    n = 30
    a = {"v_z_des": torch.zeros(1, n + 1), "v_tan_des": torch.full((1, n + 1), 0.1),
         "z_ref": torch.zeros(1, n + 1), "s_ref": torch.zeros(1, n + 1)}
    b = {**a, "v_tan_des": torch.full((1, n + 1), -0.1)}
    kw = dict(e_now=torch.zeros(NE), phase=torch.tensor(0.0), n_stages=n)
    assert not torch.equal(build_features("preview", ref=a, **kw),
                           build_features("preview", ref=b, **kw)), "signed default must differ"
    assert torch.equal(build_features("preview", ref=a, sym_sway=True, **kw),
                       build_features("preview", ref=b, sym_sway=True, **kw))


def test_plant_block_width_and_placement():
    from marinelab.algorithms.diff_wmpc import N_PLANT
    ref = _ref()
    kw = dict(e_now=torch.zeros(NE), phase=torch.tensor(0.0), ref=ref, n_stages=30)
    base = build_features("both", **kw)
    plant = [1.0, 2.0, 3.0, 4.0, 5.0]
    got = build_features("both", plant=plant, **kw)
    assert got.shape == (feature_dim("both", NE, plant=True),)
    assert torch.equal(got[:base.numel()], base), "plant block must APPEND, not reorder"
    assert torch.equal(got[base.numel():], torch.tensor(plant))
    with pytest.raises(ValueError):
        build_features("both", plant=[1.0, 2.0], **kw)
    assert feature_dim("both", NE, plant=True) - feature_dim("both", NE) == N_PLANT


def test_static_weights_ignore_features_and_stay_in_bounds():
    from marinelab.algorithms.diff_wmpc import StaticWeights
    init = np.arange(1, NE + 1) * 10.0
    m = StaticWeights(FEAT, NE, NU, werr_init=init, wu_init=np.full(NU, 0.01))
    a = m(torch.zeros(FEAT))
    b = m(torch.randn(FEAT) * 100)
    assert torch.equal(a, b), "Diff-MPC is theta_k = theta; the input must not matter"
    assert torch.allclose(a[:NE], torch.tensor(init, dtype=torch.float32), rtol=1e-3)
    assert (a[:NE] >= m.werr_lb).all() and (a[:NE] <= m.werr_ub).all()


def test_static_weights_are_learnable_and_roundtrip():
    from marinelab.algorithms.diff_wmpc import StaticWeights
    m = StaticWeights(FEAT, NE, NU, werr_init=np.full(NE, 40.0), wu_init=np.full(NU, 0.01))
    before = m(torch.zeros(FEAT)).clone()
    m(torch.zeros(FEAT)).sum().backward()
    assert m.raw.grad is not None and torch.isfinite(m.raw.grad).all()
    with torch.no_grad():
        m.raw += 0.5
    assert not torch.equal(m(torch.zeros(FEAT)), before)
    m2 = StaticWeights(FEAT, NE, NU)
    m2.load_state_dict(m.state_dict())
    assert torch.equal(m2(torch.zeros(FEAT)), m(torch.zeros(FEAT)))
