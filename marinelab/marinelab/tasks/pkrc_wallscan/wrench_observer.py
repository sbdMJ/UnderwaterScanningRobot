# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Residual-wrench observer: the online buoyancy/trim estimate the NMPC needs under DR.

Pure numpy — no isaaclab, no acados — so the model mirror and the observer gains are
unit-testable natively, the same way ``wall_frame_ekf`` and ``mpc_reference`` are.

## Why this exists

``mpc_controller`` carries a NOMINAL plant. Under the ``Eval`` task's domain randomization the
plant is not nominal: ``volume_scale=(0.85, 1.15)`` is +-34 N of buoyancy against 40 N per
thruster, CoB/CoG shift +-5 cm, and thrust coefficient scales +-30%. A pure NMPC has no integral
action, so a constant disturbance leaves a STANDING offset — and the offset measured across
8 envs x 3 seeds (``results/metrics_dwDRds_*.json``, settled window) is large:

| | nominal | stress DR |
|:--|--:|--:|
| tilt (heave leg) | 0.18 deg | **13.83 deg** |
| wall standoff error | 1.10 cm | 5.52 cm |
| saturated steps | 1.3% | 29% |
| QP failures | 0.00% | 0.00% |

QP failures stayed at zero and solve time did not move, so this is not a solver problem. The
mechanism worth spelling out, because it explains why the tilt is so much worse than the
standoff: the nominal model has a RESTORING moment (CoB 0.15 m above the origin), so the
solver's prediction says an observed tilt will right itself for free. It therefore declines to
spend thrust on a tilt that its model claims is transient, and the true equilibrium tilt — set
by the DR'd CoB/CoG offset against the DR'd buoyancy — persists for the whole leg. Tracking
error in ``z`` gets corrected because nothing in the model claims it is self-correcting.

Feeding the residual back as a model parameter is what removes the standing offset: the solver
then predicts the true equilibrium and trims against it, which is standard offset-free NMPC.
Diff-WMPC cannot substitute for this — sensitivity is exactly zero at saturation, so the learner
never sees the regime that is failing (``skipped`` 39% -> 87% when trained under DR, and
``w_z`` moved 1308 -> 1347, i.e. not at all).

## What it actually bought, and the actuation limit it exposed

MEASURED 2026-08-04, stress DR, 8 envs x 3 seeds, paired per seed, signs unanimous
(``dwDRobs`` vs ``dwDRds``): tilt heave **13.83 -> 11.44 deg**, tilt sway 13.27 -> 11.09, but
wall standoff **5.52 -> 9.64 cm**, crab 2.00 -> 3.01 deg, and saturated steps **39% -> 67%**.
QP failures stayed 0.00%, zero collisions. Nominal condition is near-neutral: tilt 0.183 ->
0.239 deg (unanimous but tiny), standoff and saturation unchanged. **Net, this is not an
improvement, which is why ``--dobs`` is off by default.**

The nominal +0.06 deg is itself informative: on a well-modelled plant the observer still reports
|d_f| ~ 2 N and |d_m| ~ 0.3 N*m, which is the unmodelled thruster lag (see the traps below)
being fed back as though it were environmental.

The finding that matters is why the remaining 11.4 deg is there. Under
``PKRCThrusterCfgFixedTAM`` the pitch row of the allocation matrix is **all zeros** — the sway
pair sits at z = -0.09 m, so it makes a ROLL moment, and the heave pair sits at x = 0, so
``tau[4] == 0`` identically. Pitch has no control authority at all. The observer's pitch
estimate is 6.0-9.8 N*m, and a moment that cannot be actuated is balanced only by the restoring
moment ``B * z_cob * sin(theta) = 34.29 * sin(theta)``:

| seed | \|Mx\| (actuated) | \|My\| (UNACTUATED) | predicted pitch eq. | measured tilt (obs on) |
|:--|--:|--:|--:|--:|
| 0 | 6.96 | 6.02 | 10.12 deg | 8.49 deg |
| 1 | 4.06 | 9.28 | 15.71 | 12.07 |
| 2 | 8.27 | 9.81 | 16.62 | 13.76 |

The ordering matches and the magnitude lands at 80-85% of prediction (|My| is an all-run mean of
an absolute value; the tilt metric is total tilt over settled heave legs). So **the stress-DR tilt
is an unactuated-axis equilibrium, not a control failure.** The observer buys 2.2-2.4 deg by
trimming the roll half, which IS actuated, and cannot touch the half that sets the floor.

Two consequences worth carrying forward:

* The standoff and saturation cost is the price of that roll trim. Roll is produced by the sway
  pair AND a heave differential — the same thrusters running the sway leg, the heave leg and the
  buoyancy offset — so trimming roll spends authority those axes needed.
* ``--w_roll`` raises ``werr[8]`` and ``werr[9]`` together, and the pitch weight can do nothing
  because there is no actuator behind it. That is the mechanical reason weight tuning was never
  going to reach this failure, independent of the sensitivity-at-saturation argument.

The shipped ``PKRCThrusterCfg`` puts that moment arm on the pitch row instead, which makes pitch
look actuated and roll unactuated. Per ``pkrc.py``'s own docstring the fixed TAM is the physically
correct one, so the shipped matrix was granting the controller authority the vehicle does not
have. Nominally this costs nothing (no pitch disturbance exists); under DR's +-5 cm CoB offset it
is exactly the binding constraint.

## What is estimated, and in which frame

Six numbers, matching ``mpc_reference.ND``::

    d[0:3]  residual FORCE, WORLD frame   (excess buoyancy is world-vertical and constant)
    d[3:6]  residual MOMENT, BODY frame   (a CoB/CoG trim moment rotates with the vehicle)

The frames are not interchangeable. Excess buoyancy ``dB`` acts along world +z whatever the
attitude, so it is constant only in the world frame — which is why ``_continuous_dynamics``
takes ``d_world`` and applies ``R.T @ d_world``. The trim moment is ``r_cob x f_buoy_body`` and
``f_buoy_body ~ (0, 0, B)`` at the small tilts this task runs at, so a CoB offset error
``(dx, dy)`` shows up as ``(dy*B, -dx*B, 0)`` — constant in the BODY frame.

## Estimator form: filtered acceleration residual, not an augmented EKF

The model says (mirroring ``_continuous_dynamics`` exactly)::

    mass * (v_dot + w x v) = tau[0:3] + hydro[0:3] + f_buoy + f_grav + R.T @ d_f_world
    I_rb  *  w_dot         = tau[3:6] + hydro[3:6] + m_buoy + d_m_body

so the residual is available in closed form from measured velocity and APPLIED thrust, with no
augmented state and no Jacobians::

    f_resid_body = mass * (v_dot_meas + w x v) - (modelled force)
    d_f_world   <- lowpass(R @ f_resid_body)

A first-order lowpass rather than a raw residual because ``v_dot_meas`` is a finite difference of
a noisy, zero-order-held DVL. The cutoff is what makes that safe: at ``lam_force = 1.0`` rad/s
the averaging window is ~1 s, while the DVL-A50's 15 Hz hold repeats a sample for only 3 control
steps (60 ms). The hold contributes a zero-mean stair pattern well inside the passband, and a
CONSTANT disturbance — the thing being estimated — passes unattenuated. This is deliberately the
cheapest form that can converge; an augmented-state EKF buys covariance bookkeeping that nothing
downstream consumes, since the MPC takes ``d`` as a certainty-equivalent parameter.

## Traps, and the two that matter most

* **Use the APPLIED (clipped) command, never the desired one.** Under DR the plant saturates 29%
  of steps. Feed the unclipped MPC output and the observer attributes the undeliverable part of
  the command to the environment, then the MPC trims against a disturbance it invented and winds
  up. ``u_newton`` must be the command that actually reached the thrusters.
* **The thruster lag is not modelled** (``tau_up = 0.1 s``), matching ``mpc_controller``'s own
  first-iteration choice. The lag makes applied thrust trail the command during reversals, which
  the 1 s lowpass mostly rejects; what survives is a small bias while a ramp is sustained. If a
  measurement ever pins residual error to reference reversals, augment the state instead of
  tightening the gains.
* **Bounds are anti-windup, not tuning.** The physically reachable range is bounded (+-34 N of
  buoyancy, ~11 N*m of trim from a 5 cm CoB offset at 228 N). Anything past that is a modelling
  failure elsewhere, and letting it accumulate turns a bad estimate into a commanded dive.
* The observer runs on the ESTIMATOR's velocities, so it inherits their provenance. As of
  2026-08-04 ``estimator_loop`` synthesizes DVL body x/y and gyro z but passes body ``v_z`` and
  ``w_x``/``w_y`` through from ground truth, so the heave-force channel — the one that carries
  buoyancy — is partly GT-fed. The DVL-A50 and the 3DM-GV7 both measure three axes, so this is
  an under-modelled sensor rather than a missing one; extending the synthesis is the honest
  follow-up, and until then any sim2real claim for ``d_f_world[2]`` is provisional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["WrenchObserverCfg", "WrenchObserver", "plant_wrench", "quat_to_rot"]

# Mirrors mpc_controller.GRAVITY; duplicated rather than imported so this module stays importable
# without acados (mpc_controller pulls in casadi at import time).
GRAVITY = 9.81


def quat_to_rot(quat) -> np.ndarray:
    """(3, 3) body->world rotation from a wxyz quaternion.

    Normalized on the way in for the same reason ``mpc_controller._error_expr`` does it: a
    non-unit quaternion does not produce a rotation, it produces a scaling by ``|q|^2``, and the
    estimator's quaternion is rebuilt from euler angles every step so its norm is only nominally
    one.
    """
    q = np.asarray(quat, float).reshape(4)
    q = q / max(math.sqrt(float(q @ q)), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def plant_wrench(prm, quat, v_b, w_b, u_newton) -> tuple[np.ndarray, np.ndarray]:
    """``(force_body, moment_body)`` the NOMINAL model predicts, excluding the disturbance.

    Numpy mirror of :func:`mpc_controller._continuous_dynamics`, term for term and in the same
    order. It has to be a mirror and not merely "the Fossen equations": that function encodes
    three deliberate departures from the textbook model to match THIS plant (no added mass in the
    mass matrix, ``C_RB`` computed with the cfg-fallback inertia while ``w_dot`` uses the PhysX
    tensor, explicit gravity), and an observer built on the textbook version would report those
    departures as environmental disturbance. ``tests/pkrc_wallscan/test_wrench_observer.py``
    cross-checks the two whenever casadi is importable.

    ``prm`` is a :class:`~marinelab.tasks.pkrc_wallscan.mpc_controller.PlantParams`; it is taken
    duck-typed so this module stays importable without acados.
    """
    v_b = np.asarray(v_b, float).reshape(3)
    w_b = np.asarray(w_b, float).reshape(3)
    nu = np.concatenate([v_b, w_b])

    D_l = np.asarray(prm.linear_damping, float)
    D_q = np.asarray(prm.quadratic_damping, float)
    A = np.asarray(prm.added_mass, float)
    I_cor = np.asarray(prm.coriolis_inertia, float)

    damping = D_l * nu + D_q * np.abs(nu) * nu

    c_rb_torque = np.cross(w_b, I_cor * w_b)
    ma = A * nu
    ma_lin, ma_ang = ma[0:3], ma[3:6]
    c_a_force = -np.cross(ma_lin, w_b)
    c_a_torque = -(np.cross(ma_lin, v_b) + np.cross(ma_ang, w_b))
    coriolis = np.concatenate([c_a_force, c_rb_torque + c_a_torque])

    hydro = -(coriolis + damping)

    R = quat_to_rot(quat)
    up_b = R.T @ np.array([0.0, 0.0, 1.0])
    f_buoy = prm.buoyancy_force * up_b
    m_buoy = np.cross(np.asarray(prm.center_of_buoyancy, float), f_buoy)
    f_grav = R.T @ np.array([0.0, 0.0, -prm.mass * GRAVITY])

    B = np.asarray(prm.allocation_matrix, float)
    tau = B @ np.asarray(u_newton, float).reshape(-1)

    return (tau[0:3] + hydro[0:3] + f_buoy + f_grav,
            tau[3:6] + hydro[3:6] + m_buoy)


@dataclass
class WrenchObserverCfg:
    """Gains and anti-windup bounds. See the module docstring for why each is set here."""

    # Lowpass cutoffs [rad/s]. Force is slower because its residual rides on the DVL's
    # zero-order hold; the moment channel is fed by the gyro, which is not held.
    lam_force: float = 1.0
    lam_moment: float = 2.0
    # Anti-windup bounds, per component. Force: Eval's volume_scale is +-34 N of buoyancy, so 60 N
    # already leaves headroom for a thrust-coefficient error on top. Moment: a 5 cm CoB/CoG
    # offset against 228 N of buoyancy is 11.4 N*m.
    max_force: float = 60.0
    max_moment: float = 20.0
    # Number of steps to ignore after a reset. The first residual needs a previous velocity
    # sample, and the two steps after a spawn are dominated by the release transient.
    warmup_steps: int = 3
    # Freeze integration while the command is saturated. Default OFF: with the APPLIED command as
    # input the residual stays valid under saturation, and freezing there would blind the observer
    # in exactly the regime it was built for. Exposed because the argument is a modelling claim,
    # not a proof.
    freeze_on_saturation: bool = False
    # Which of the six channels are EXPORTED to the MPC, ordered [Fx Fy Fz Mx My Mz]. All six are
    # always estimated (the diagnostics stay complete); a masked channel is simply not handed to
    # the solver.
    #
    # MEASURED 2026-08-04, stress DR, 8 envs x 3 seeds, paired per seed (``dwDRds`` = off,
    # ``dwDRobs`` = all, ``dwDRobsZM`` = z_moment, ``dwDRobsM`` = moment; settled window):
    #
    # | export | tilt heave | tilt sway | wall err | saturated |
    # |:--|--:|--:|--:|--:|
    # | off | 13.83 deg | 13.27 deg | 5.52 cm | 39% |
    # | all six | 11.44 | 11.09 | 9.64 | 67% |
    # | Fz + moments | 11.49 | 11.08 | 9.52 | 66% |
    # | moments only | 11.59 | 10.92 | 9.64 | 65% |
    #
    # **The three exports are indistinguishable, so the mask is not the lever it was built to be.**
    # Every effect -- the -2.2 to -2.4 deg of tilt, the +4.1 cm of standoff, the doubled saturation
    # -- comes from the MOMENT channels alone. The force half contributes nothing measurable, the
    # 16-22 N buoyancy residual on Fz included: z tracking was never what DR broke (weight 40 on a
    # directly measured channel already handles it), so correcting the model there changes no
    # decision the solver was making.
    #
    # A hypothesis this refutes, recorded because it looked well-supported: |Fx|, |Fy| come out at
    # 4-7 N mean, which is damping DR (x0.5-1.5) resolved through a tilted vehicle -- heaving at
    # 0.2 m/s while 10 deg off vertical puts 0.035 m/s on the lateral body axes and 97.79 N*s/m
    # turns that into ~3.4 N, matching the observed magnitude. Exporting a velocity-dependent
    # residual as a horizon constant IS wrong, and it predicted the standoff loss. Withholding
    # those channels changed the standoff by -0.12 cm. The radial axis is servoed at weight 40, so
    # a 4-7 N bias inside the model barely moves the closed-loop equilibrium; the +4.1 cm comes
    # from saturation stealing authority, not from a biased model.
    channel_mask: tuple[bool, bool, bool, bool, bool, bool] = (True, True, True, True, True, True)


class WrenchObserver:
    """Residual wrench for ONE vehicle: ``d = [force_world(3), moment_body(3)]``.

    Single-vehicle for the same reason as ``WallFrameEKF`` and ``WallFrameEstimator``: it holds
    one vehicle's filter state. Sharing an instance across envs mixes vehicles 20 m apart and is
    the bug that cost the most time in this project (see ``mpc_controller`` / CLAUDE.md); one
    instance per env, and reset only the envs that actually reset.
    """

    def __init__(self, prm, cfg: WrenchObserverCfg | None = None):
        self.prm = prm
        self.cfg = cfg or WrenchObserverCfg()
        self.d = np.zeros(6)
        self._mask = np.asarray(self.cfg.channel_mask, float).reshape(6)
        self._v_prev: np.ndarray | None = None
        self._w_prev: np.ndarray | None = None
        self._n = 0
        self.n_update = 0
        self.n_clipped = 0

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        """Forget the estimate and the velocity history.

        The disturbance is a per-episode property (buoyancy and CoB are re-drawn on every reset
        under DR), so carrying the estimate across an episode boundary would start the new
        episode trimming for the old vehicle.
        """
        self.d[:] = 0.0
        self._v_prev = self._w_prev = None
        self._n = 0

    @property
    def d_force_world(self) -> np.ndarray:
        """RAW estimate, mask not applied — for diagnostics, not for the solver."""
        return self.d[0:3]

    @property
    def d_moment_body(self) -> np.ndarray:
        """RAW estimate, mask not applied — for diagnostics, not for the solver."""
        return self.d[3:6]

    def exported(self) -> np.ndarray:
        """The (6,) vector to hand the solver: the estimate with ``channel_mask`` applied."""
        return self.d * self._mask

    # -- per step ----------------------------------------------------------
    def update(self, quat, v_b, w_b, u_newton, dt: float, *, saturated: bool = False) -> np.ndarray:
        """One observer step. Returns the current ``d`` (6,), world force then body moment.

        ``u_newton`` is the thrust that was APPLIED over the interval ending at this sample, in
        newtons — i.e. the clipped command times the thrust coefficient. Passing the desired
        command instead makes the observer wind up under saturation (see module docstring).
        """
        v = np.asarray(v_b, float).reshape(3)
        w = np.asarray(w_b, float).reshape(3)

        if self._v_prev is None:
            self._v_prev, self._w_prev = v.copy(), w.copy()
            self._n += 1
            return self.exported()

        v_dot = (v - self._v_prev) / dt
        w_dot = (w - self._w_prev) / dt
        self._v_prev, self._w_prev = v.copy(), w.copy()
        self._n += 1
        if self._n <= self.cfg.warmup_steps or (saturated and self.cfg.freeze_on_saturation):
            return self.exported()

        f_model, m_model = plant_wrench(self.prm, quat, v, w, u_newton)
        I_rb = np.asarray(self.prm.rigid_body_inertia, float)

        # Invert the same two lines the model integrates:
        #   v_dot = f_tot/mass - w x v        ->  f_tot = mass * (v_dot + w x v)
        #   w_dot = m_tot/I_rb               ->  m_tot = I_rb * w_dot
        f_meas = self.prm.mass * (v_dot + np.cross(w, v))
        m_meas = I_rb * w_dot

        R = quat_to_rot(quat)
        f_resid_world = R @ (f_meas - f_model)
        m_resid_body = m_meas - m_model

        a_f = 1.0 - math.exp(-self.cfg.lam_force * dt)
        a_m = 1.0 - math.exp(-self.cfg.lam_moment * dt)
        self.d[0:3] += a_f * (f_resid_world - self.d[0:3])
        self.d[3:6] += a_m * (m_resid_body - self.d[3:6])

        clipped = np.concatenate([
            np.clip(self.d[0:3], -self.cfg.max_force, self.cfg.max_force),
            np.clip(self.d[3:6], -self.cfg.max_moment, self.cfg.max_moment),
        ])
        if not np.array_equal(clipped, self.d):
            self.n_clipped += 1
        self.d[:] = clipped
        self.n_update += 1
        return self.exported()
