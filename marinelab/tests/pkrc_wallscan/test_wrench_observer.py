# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Guard tests for the residual-wrench observer.

Native (no Isaac Sim, no acados): ``wrench_observer`` is pure numpy by design. The one test that
needs casadi — the cross-check that the numpy model mirror really matches the CasADi model the
solver integrates — skips itself when casadi is absent and runs in the container.

The property being protected is narrow and worth stating: the observer must converge to the
disturbance a plant actually has, and must NOT invent one when the plant is nominal. A biased
observer is worse than none, because the MPC trims against it with full authority.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from marinelab.tasks.pkrc_wallscan.mpc_reference import ND, ND_FORCE, ND_MOMENT, NP_REF
from marinelab.tasks.pkrc_wallscan.wrench_observer import (
    GRAVITY,
    WrenchObserver,
    WrenchObserverCfg,
    plant_wrench,
    quat_to_rot,
)


# PKRC's shipped 6x6 allocation matrix is not needed here: the observer only ever multiplies
# B @ u, so any full-rank B exercises the same code. A diagonal-ish B keeps the expected values
# hand-checkable, and one test below uses the real fixed-TAM matrix instead.
def _plant(**over):
    """Duck-typed PlantParams stand-in, so these tests never import mpc_controller (casadi)."""

    class P:
        mass = 22.8
        rigid_body_inertia = (1.412, 1.406, 0.393)
        coriolis_inertia = (0.25, 0.25, 0.25)
        added_mass = (19.40, 64.65, 64.65, 0.5, 0.5, 0.5)
        linear_damping = (97.79, 119.44, 119.44, 15.0, 15.0, 4.0)
        quadratic_damping = (180.85, 38.51, 38.51, 30.0, 30.0, 8.0)
        buoyancy_force = 228.57
        center_of_buoyancy = (0.0, 0.0, 0.15)
        max_thrust = 40.0
        # Unit wrench basis: u[i] drives generalized axis i directly.
        allocation_matrix = tuple(tuple(1.0 if i == j else 0.0 for j in range(6)) for i in range(6))

    for k, v in over.items():
        setattr(P, k, v)
    return P


IDENT_Q = (1.0, 0.0, 0.0, 0.0)


def _quat_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


# ---------------------------------------------------------------------------
# parameter layout
# ---------------------------------------------------------------------------


def test_nd_is_force_plus_moment():
    """The moment half is what makes a standing tilt correctable; a force at the body origin
    produces no moment, so a force-only ND could never trim one (see mpc_reference's comment)."""
    assert (ND_FORCE, ND_MOMENT, ND) == (3, 3, 6)
    assert NP_REF == 7


# ---------------------------------------------------------------------------
# the model mirror
# ---------------------------------------------------------------------------


def test_static_hover_wrench_is_net_buoyancy_and_trim():
    """At rest, level, no thrust: the only terms left are buoyancy, weight and the CoB moment."""
    prm = _plant()
    f, m = plant_wrench(prm, IDENT_Q, np.zeros(3), np.zeros(3), np.zeros(6))
    net = prm.buoyancy_force - prm.mass * GRAVITY          # +4.89 N, PKRC is slightly buoyant
    assert f == pytest.approx([0.0, 0.0, net], abs=1e-9)
    # r_cob x f_buoy with both along +z is zero -- a centred CoB produces no trim moment.
    assert m == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_cob_offset_makes_a_trim_moment_at_zero_tilt():
    """A lateral CoB offset is the mechanism the moment channel exists to cancel.

    Eval's DR shifts CoB/CoG by +-5 cm; against 228.57 N of buoyancy that is 11.4 N*m, which is
    the number ``WrenchObserverCfg.max_moment`` is sized from.
    """
    prm = _plant(center_of_buoyancy=(0.0, 0.05, 0.15))
    _, m = plant_wrench(prm, IDENT_Q, np.zeros(3), np.zeros(3), np.zeros(6))
    # (0, 0.05, 0.15) x (0, 0, 228.57) = (0.05*228.57, 0, 0)
    assert m[0] == pytest.approx(0.05 * prm.buoyancy_force, rel=1e-12)
    assert m[1:] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert abs(m[0]) < WrenchObserverCfg().max_moment


def test_buoyancy_is_world_vertical_under_roll():
    """Buoyancy stays along world +z whatever the attitude -- the reason the FORCE residual is
    parameterized in the world frame while the moment is not."""
    prm = _plant()
    q = _quat_from_rpy(0.4, 0.0, 0.0)
    f, _ = plant_wrench(prm, q, np.zeros(3), np.zeros(3), np.zeros(6))
    f_world = quat_to_rot(q) @ f
    net = prm.buoyancy_force - prm.mass * GRAVITY
    assert f_world == pytest.approx([0.0, 0.0, net], abs=1e-9)


def test_damping_signs_oppose_motion():
    prm = _plant()
    v = np.array([0.3, 0.0, 0.0])
    f, _ = plant_wrench(prm, IDENT_Q, v, np.zeros(3), np.zeros(6))
    expected = -(prm.linear_damping[0] * 0.3 + prm.quadratic_damping[0] * 0.3 * 0.3)
    assert f[0] == pytest.approx(expected, rel=1e-12)


def test_thrust_enters_through_the_allocation_matrix():
    B = np.zeros((6, 6))
    B[2, 0] = 4.0        # u[0] -> 4 N of body heave
    B[3, 1] = 0.5        # u[1] -> 0.5 N*m of roll
    prm = _plant(allocation_matrix=tuple(tuple(r) for r in B))
    u = np.zeros(6)
    u[0], u[1] = 2.0, 3.0
    f, m = plant_wrench(prm, IDENT_Q, np.zeros(3), np.zeros(3), u)
    net = prm.buoyancy_force - prm.mass * GRAVITY
    assert f[2] == pytest.approx(net + 8.0, rel=1e-12)
    assert m[0] == pytest.approx(1.5, rel=1e-12)


# ---------------------------------------------------------------------------
# the observer
# ---------------------------------------------------------------------------


def _roll_out(prm, obs, d_true_force_world, d_true_moment_body, *, steps=4000, dt=0.02,
              quat=IDENT_Q, u=None, saturated=False):
    """Integrate the TRUE plant (= nominal model + a constant extra wrench) and feed the observer.

    The plant is stepped with explicit Euler at the control rate, which is the same discretization
    the observer inverts, so a converged estimate is a statement about the observer and not about
    integration error.
    """
    u = np.zeros(6) if u is None else np.asarray(u, float)
    R = quat_to_rot(quat)
    v = np.zeros(3)
    w = np.zeros(3)
    I_rb = np.asarray(prm.rigid_body_inertia, float)
    for _ in range(steps):
        f_model, m_model = plant_wrench(prm, quat, v, w, u)
        f_tot = f_model + R.T @ np.asarray(d_true_force_world, float)
        m_tot = m_model + np.asarray(d_true_moment_body, float)
        v = v + dt * (f_tot / prm.mass - np.cross(w, v))
        w = w + dt * (m_tot / I_rb)
        obs.update(quat, v, w, u, dt, saturated=saturated)
    return obs.d.copy()


def test_converges_to_a_constant_force_and_moment():
    """The headline property: a plant carrying an unmodelled constant wrench is identified."""
    prm = _plant()
    obs = WrenchObserver(prm, WrenchObserverCfg())
    d_f = np.array([0.0, 0.0, -30.0])       # 30 N of excess buoyancy, world frame
    d_m = np.array([0.0, 8.0, 0.0])         # 8 N*m of pitch trim, body frame
    d = _roll_out(prm, obs, d_f, d_m)
    assert d[0:3] == pytest.approx(d_f, abs=0.5)
    assert d[3:6] == pytest.approx(d_m, abs=0.5)


def test_nominal_plant_yields_no_disturbance():
    """A nominal plant must not produce a phantom estimate. The MPC trims against ``d`` with full
    authority, so a biased observer is actively worse than no observer."""
    prm = _plant()
    obs = WrenchObserver(prm, WrenchObserverCfg())
    d = _roll_out(prm, obs, np.zeros(3), np.zeros(3))
    assert np.abs(d[0:3]).max() < 0.05
    assert np.abs(d[3:6]).max() < 0.05


def test_world_frame_force_is_recovered_from_a_tilted_vehicle():
    """The force channel is world-frame, so a rolled vehicle must still report a vertical
    disturbance as vertical -- not smeared into body y/z."""
    prm = _plant()
    q = _quat_from_rpy(0.35, -0.2, 1.1)
    obs = WrenchObserver(prm, WrenchObserverCfg())
    d_f = np.array([0.0, 0.0, -25.0])
    d = _roll_out(prm, obs, d_f, np.zeros(3), quat=q)
    assert d[0:3] == pytest.approx(d_f, abs=0.6)


def test_thrust_coefficient_error_is_absorbed_as_disturbance():
    """Eval scales the thrust coefficient by 0.7-1.3. The observer sees the plant's real thrust
    but computes the model's nominal thrust, so the gap lands in ``d`` -- which is what lets the
    MPC compensate an actuator-gain error it has no state for."""
    B = np.zeros((6, 6))
    B[2, 0] = 1.0
    prm_nom = _plant(allocation_matrix=tuple(tuple(r) for r in B))
    u = np.zeros(6)
    u[0] = 20.0
    # The true plant delivers 30% less thrust than the nominal model believes.
    obs = WrenchObserver(prm_nom, WrenchObserverCfg())
    prm_true = _plant(allocation_matrix=tuple(tuple(r) for r in B * 0.7))

    dt = 0.02
    R = quat_to_rot(IDENT_Q)
    v, w = np.zeros(3), np.zeros(3)
    I_rb = np.asarray(prm_true.rigid_body_inertia, float)
    for _ in range(4000):
        f_tot, m_tot = plant_wrench(prm_true, IDENT_Q, v, w, u)
        v = v + dt * (f_tot / prm_true.mass - np.cross(w, v))
        w = w + dt * (m_tot / I_rb)
        obs.update(IDENT_Q, v, w, u, dt)
    del R
    # 30% of 20 N, missing along body/world +z.
    assert obs.d[2] == pytest.approx(-6.0, abs=0.5)


def test_channel_mask_withholds_only_the_export():
    """A masked channel must still be ESTIMATED -- the diagnostics are what tell you whether
    withholding it was the right call -- but must not reach the solver."""
    prm = _plant()
    obs = WrenchObserver(prm, WrenchObserverCfg(
        channel_mask=(False, False, False, True, True, True)))
    d_f = np.array([0.0, 0.0, -30.0])
    d_m = np.array([0.0, 8.0, 0.0])
    exported = _roll_out(prm, obs, d_f, d_m)
    # _roll_out returns obs.d, the raw estimate: the force channel converged as usual.
    assert obs.d_force_world[2] == pytest.approx(-30.0, abs=0.5)
    assert obs.d_moment_body[1] == pytest.approx(8.0, abs=0.5)
    # What the solver receives has the force zeroed.
    exported = obs.exported()
    assert exported[0:3].tolist() == [0.0, 0.0, 0.0]
    assert exported[3:6] == pytest.approx(obs.d[3:6], rel=1e-12)


def test_default_mask_exports_everything():
    """The default must reproduce the all-channel behaviour the 3-seed measurement used, so the
    published dwDRobs numbers keep their meaning."""
    assert WrenchObserverCfg().channel_mask == (True,) * 6
    obs = WrenchObserver(_plant())
    obs.d[:] = np.arange(6, dtype=float)
    assert obs.exported() == pytest.approx(np.arange(6, dtype=float))


def test_bounds_clamp_a_runaway_estimate():
    prm = _plant()
    cfg = WrenchObserverCfg(max_force=10.0, max_moment=2.0)
    obs = WrenchObserver(prm, cfg)
    d = _roll_out(prm, obs, np.array([0.0, 0.0, -200.0]), np.array([50.0, 0.0, 0.0]), steps=2000)
    assert np.abs(d[0:3]).max() <= cfg.max_force + 1e-9
    assert np.abs(d[3:6]).max() <= cfg.max_moment + 1e-9
    assert obs.n_clipped > 0


def test_warmup_suppresses_the_first_steps():
    """The first residual needs a previous velocity sample, and the steps right after a spawn are
    dominated by the release transient rather than by any disturbance."""
    prm = _plant()
    obs = WrenchObserver(prm, WrenchObserverCfg(warmup_steps=5))
    for _ in range(5):
        obs.update(IDENT_Q, np.array([0.0, 0.0, 1.0]), np.zeros(3), np.zeros(6), 0.02)
    assert obs.n_update == 0
    assert np.all(obs.d == 0.0)


def test_reset_forgets_the_estimate():
    """Buoyancy and CoB are re-drawn on every DR reset, so carrying ``d`` across an episode
    boundary would start the new episode trimming for the previous vehicle."""
    prm = _plant()
    obs = WrenchObserver(prm, WrenchObserverCfg())
    _roll_out(prm, obs, np.array([0.0, 0.0, -30.0]), np.zeros(3), steps=1500)
    assert abs(obs.d[2]) > 1.0
    obs.reset()
    assert np.all(obs.d == 0.0)
    assert obs.d_force_world.tolist() == [0.0, 0.0, 0.0]
    assert obs.d_moment_body.tolist() == [0.0, 0.0, 0.0]


def test_freeze_on_saturation_is_opt_in():
    """Default OFF is a modelling claim: with the APPLIED command as input the residual stays
    valid under saturation, and freezing there would blind the observer in the 29%-saturated DR
    regime it was built for."""
    prm = _plant()
    assert WrenchObserverCfg().freeze_on_saturation is False
    obs = WrenchObserver(prm, WrenchObserverCfg(freeze_on_saturation=True))
    d = _roll_out(prm, obs, np.array([0.0, 0.0, -30.0]), np.zeros(3), steps=1500, saturated=True)
    assert np.all(d == 0.0)
    assert obs.n_update == 0


def test_lowpass_gain_is_exact_first_order():
    """``alpha = 1 - exp(-lam*dt)`` rather than ``lam*dt``: the latter is unstable once
    ``lam*dt > 2``, and dt here is set by the env, not by this module."""
    prm = _plant()
    cfg = WrenchObserverCfg(lam_force=1.0, lam_moment=1.0, warmup_steps=0)
    obs = WrenchObserver(prm, cfg)
    dt = 0.02
    # Prime the velocity history, then hold a constant residual for exactly one time constant.
    obs.update(IDENT_Q, np.zeros(3), np.zeros(3), np.zeros(6), dt)
    d = _roll_out(prm, obs, np.array([0.0, 0.0, -10.0]), np.zeros(3), steps=int(1.0 / dt), dt=dt)
    # One time constant of a step response reaches 1 - 1/e = 63.2%.
    assert d[2] == pytest.approx(-10.0 * (1 - math.exp(-1.0)), rel=0.15)


# ---------------------------------------------------------------------------
# the mirror really is a mirror (container only)
# ---------------------------------------------------------------------------


def test_numpy_mirror_matches_the_casadi_model():
    """``plant_wrench`` must equal the wrench inside ``mpc_controller._continuous_dynamics``.

    This is the test that keeps the observer honest: any drift between the two makes the observer
    report a MODELLING difference as an environmental disturbance, and the MPC would then trim
    against the discrepancy. Compared through xdot rather than the wrench directly, because the
    CasADi function only exposes the assembled derivative.
    """
    ca = pytest.importorskip("casadi", reason="casadi lives in the container")
    from marinelab.tasks.pkrc_wallscan.mpc_controller import PlantParams, _continuous_dynamics

    prm = PlantParams(allocation_matrix=tuple(
        tuple(1.0 if i == j else 0.0 for j in range(6)) for i in range(6)))
    B = np.asarray(prm.allocation_matrix, float)

    rng = np.random.default_rng(0)
    for _ in range(8):
        v = rng.normal(0, 0.3, 3)
        w = rng.normal(0, 0.2, 3)
        q = _quat_from_rpy(*rng.normal(0, 0.3, 3))
        u = rng.normal(0, 10.0, 6)

        x = np.concatenate([np.zeros(3), q, v, w])
        xs = ca.SX.sym("x", 13)
        us = ca.SX.sym("u", 6)
        f = ca.Function("f", [xs, us],
                        [_continuous_dynamics(xs, us, prm, B, ca.DM.zeros(3), ca.DM.zeros(3))])
        xdot = np.asarray(f(x, u)).reshape(-1)

        f_np, m_np = plant_wrench(prm, q, v, w, u)
        v_dot = f_np / prm.mass - np.cross(w, v)
        w_dot = m_np / np.asarray(prm.rigid_body_inertia, float)
        assert xdot[7:10] == pytest.approx(v_dot, rel=1e-9, abs=1e-9)
        assert xdot[10:13] == pytest.approx(w_dot, rel=1e-9, abs=1e-9)


def test_casadi_model_applies_the_disturbance_where_the_observer_reports_it():
    """Frame check across the seam: a world-frame force and a body-frame moment must land on the
    axes the observer measured them on, at a non-trivial attitude."""
    ca = pytest.importorskip("casadi", reason="casadi lives in the container")
    from marinelab.tasks.pkrc_wallscan.mpc_controller import PlantParams, _continuous_dynamics

    prm = PlantParams(allocation_matrix=tuple(
        tuple(1.0 if i == j else 0.0 for j in range(6)) for i in range(6)))
    B = np.asarray(prm.allocation_matrix, float)
    q = _quat_from_rpy(0.3, -0.25, 0.9)
    x = np.concatenate([np.zeros(3), q, np.zeros(3), np.zeros(3)])

    xs = ca.SX.sym("x", 13)
    us = ca.SX.sym("u", 6)
    d_f = ca.DM([0.0, 0.0, -20.0])
    d_m = ca.DM([3.0, 0.0, 0.0])
    f0 = ca.Function("f0", [xs, us],
                     [_continuous_dynamics(xs, us, prm, B, ca.DM.zeros(3), ca.DM.zeros(3))])
    f1 = ca.Function("f1", [xs, us], [_continuous_dynamics(xs, us, prm, B, d_f, d_m)])
    u = np.zeros(6)
    delta = np.asarray(f1(x, u)).reshape(-1) - np.asarray(f0(x, u)).reshape(-1)

    # Body-frame acceleration delta, rotated back out, must be the world-frame force / mass.
    a_world = quat_to_rot(q) @ delta[7:10]
    assert a_world == pytest.approx([0.0, 0.0, -20.0 / prm.mass], abs=1e-9)
    # The moment is body-frame already: straight onto the roll axis.
    assert delta[10:13] == pytest.approx([3.0 / prm.rigid_body_inertia[0], 0.0, 0.0], abs=1e-9)
