# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Weights-parametric wallscan NMPC (acados), the controller half of the Diff-WMPC port.

Ported from ``cylinder_orbit_mpc_diff.py`` in Underwater-Actor-Critic-Model-Predictive-Control
("Differentiable Weights-Varying Nonlinear MPC via Gradient-Based Policy Learning"). Same
two-solver workflow:

* a NOMINAL solver (NONLINEAR_LS, Gauss-Newton, SQP_RTI) whose diagonal cost weights are
  written per solve through ``cost.W``, and
* a SENSITIVITY solver (EXTERNAL cost with the weights in ``p_global``, EXACT Hessian,
  ``with_solution_sens_wrt_params=True``) that is loaded with the nominal iterate and
  factorized to return ``d z* / d p_global``.

The error vector, its ordering and the reference preview all come from :mod:`mpc_reference`
(pure torch, unit-tested); this module only mirrors them in CasADi and owns the solvers.

## Matching THIS plant rather than the orbit plant

The dynamics are not a copy of the orbit model — three differences would each cause a large
prediction error, all verified against ``marinelab.core.hydrodynamics`` and the PhysX values
dumped by ``isaaclab/logs/_probe_plant.py``:

1. **No added mass in the mass matrix.** ``PKRCHydrodynamicsCfg`` leaves
   ``apply_added_mass_force=False``, so ``M_A * v_dot`` is never applied to this plant; the
   added-mass coefficients enter only through the ``C_A`` velocity coupling. The orbit model
   uses ``inv(M_rb + A)``, which here would model the vehicle as 3-4x more sluggish in
   sway/heave than it is (A_sway = 64.65 vs m = 22.8).
2. **Rotational inertia comes from PhysX, not the cfg.** ``cfg.rigid_body_inertia`` is None,
   so ``default_rigid_inertia`` falls back to ``added_mass[3:6]*0.5 = (0.25, 0.25, 0.25)``
   — but that value is used ONLY inside the ``C_RB`` Coriolis term. PhysX integrates with
   the mesh-derived tensor ``diag(1.412, 1.406, 0.393)``. The model therefore uses the
   PhysX tensor for ``w_dot`` and reproduces the plant's ``C_RB`` with the cfg value: the
   goal is to match the plant as it is, not as it should be.
3. **Gravity is explicit.** PhysX applies weight to the real robot (``disable_gravity=False``)
   while ``HydrodynamicsModel`` contributes buoyancy only, so the model must carry both.

## Actuator model (optional, nx 13 -> 13 + nu)

``PlantParams.force_rate_limit = None`` (default) keeps the original instant-force model:
the OCP input is per-thruster force and the commanded force is assumed applied instantly.
That was a deliberate first-iteration choice, flagged here as "a known follow-up if the
baseline shows it" — and the 2026-08-18 depth-hold campaign showed it (bag 04_15_35,
``hw_bag_depthhold_0415_20260818.py``): the deployed chain slew-limits the VESC current
(teleop ramp, 17-30 A/s measured), so a full command reversal takes ~0.3 s that the model
did not know about. werr z=40 then acts as a relay against a lagged plant and the depth
loop settles into a +-5 cm / ~3 s describing-function limit cycle with heave 81% saturated.

Setting ``force_rate_limit`` (per-thruster N/s = ``newton_per_amp * teleop ramp``) switches
the model to a rate-limited integrator — the same structure as the hardware:

* state grows to ``[x13, F_act(nu)]``; thrust in the dynamics comes from the state ``F_act``,
* the OCP input becomes the NORMALIZED force rate ``r`` in [-1, 1], ``F_act_dot = rate * r``
  (so ``wu`` regularizes the rate — an u-rate cost for free; sim-tuned wu values do not
  carry over),
* ``|F_act|`` is box-constrained per thruster by ``thrust_limits`` (falls back to
  ``max_thrust``) — set these to the session's realizable force ``k*(amps_limit - I0)`` and
  the optimizer stops recruiting physically-neutered thrusters through attitude coupling,
* :meth:`WallScanMPC.solve` keeps the applied-force state between ticks (integrating its
  own plan at ``cfg.step_dt``; reset to zero on ``init_state_traj``) and returns the NEXT
  tick's planned force as ``u0``/``u0_cmd``, so every downstream consumer (thrust mapper,
  adapters, telemetry) still sees a force command. Publishing the one-tick-ahead force
  means the teleop ramp can always realize it exactly (the plan never exceeds the ramp).

The first-order sim lag (``tau_up = 0.1 s``) remains unmodelled — on the vehicle the teleop
current ramp dominates it, and the sim experiments (E1-E4) keep the legacy model for
reproducibility (their plant JSONs carry no ``force_rate_limit``). The friction deadzone
(zero thrust below ~0.7 A) is compensated affinely by the thrust mapper and its residual
(min realizable force 0.37 N) is accepted as the +-2-3 cm depth-ripple floor.

## Command-latency predictor (``PlantParams.command_latency_s``)

The deployed chain also carries ~0.4 s of round-trip DEAD TIME (solve overrun, mapper +
teleop hops, CAN, T200 spin-up) that no slew model captures. Field-identified from bag
2026-08-19 23:03:29: with the rate model AND the retuned weights live, the depth loop
still swung cap-to-cap at 4.2 s / 16 cm p-p, and the matched replay
(``isaaclab/logs/_probe_field_2303.py``) reproduces that signature exactly when 0.4 s of
input delay is injected — and collapses to ~1 cm when the measured state is rolled
forward through the in-flight command history before each solve. ``command_latency_s``
enables exactly that predictor inside :meth:`WallScanMPC.solve` (one RK4 tick of the 13-D
dynamics per in-flight command; history reset on ``init_state_traj``). Over-prediction is
benign (predicting 0.4 s against a zero-delay plant costs ~0.5 cm), so the shipped hw
JSON carries the full 0.4 s estimate.

## Sensitivity validity

``eval_solution_sensitivity`` is only meaningful while the solution is interior. Measured on
this host (``isaaclab/logs/_probe_acados.py``): sensitivities match finite differences to
6-7 digits at an interior optimum and are **identically zero** once the control saturates.
Keep the weight bounds low enough to stay off the thrust limits and skip saturated steps —
the learner does the latter.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import mpc_reference as mref
from .mpc_reference import ND, NE, NP_REF

try:
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

    _ACADOS_AVAILABLE = True
    _ACADOS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on the container having acados
    ca = None
    AcadosModel = AcadosOcp = AcadosOcpSolver = None
    _ACADOS_AVAILABLE = False
    _ACADOS_IMPORT_ERROR = exc

NX = 13
GRAVITY = 9.81


@dataclass
class PlantParams:
    """Everything the CasADi model needs, in one place so it cannot silently drift.

    Defaults are the shipped PKRC values; ``rigid_body_inertia`` is the PhysX tensor, which
    is NOT in any config file (it is derived from the USD mesh) — use :func:`from_env` to
    read it off a live env instead of trusting these numbers after an asset change.
    """

    mass: float = 22.8
    # PhysX diag(Ixx, Iyy, Izz) measured 2026-07-30; off-diagonals were <= 3% and dropped.
    rigid_body_inertia: tuple[float, float, float] = (1.412, 1.406, 0.393)
    # Used ONLY to reproduce the plant's C_RB term (see module docstring, point 2).
    coriolis_inertia: tuple[float, float, float] = (0.25, 0.25, 0.25)
    added_mass: tuple[float, ...] = (19.40, 64.65, 64.65, 0.5, 0.5, 0.5)
    linear_damping: tuple[float, ...] = (97.79, 119.44, 119.44, 15.0, 15.0, 4.0)
    quadratic_damping: tuple[float, ...] = (180.85, 38.51, 38.51, 30.0, 30.0, 8.0)
    buoyancy_force: float = 228.57          # rho * g * V
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.15)
    # Command +-1 maps to +-thrust_coefficient newtons, so THIS is the actuator bound the
    # env can actually realize (cfg.max_thrust = 51.5 N is a downstream clamp that a unit
    # command never reaches).
    max_thrust: float = 40.0
    allocation_matrix: tuple[tuple[float, ...], ...] = field(default_factory=tuple)
    # Per-thruster force slew limit (N/s) of the deployed actuator chain — on the vehicle
    # this is newton_per_amp * the teleop current ramp (17-30 A/s measured 2026-08-18;
    # use the conservative 17 so the plan is always realizable). None = legacy
    # instant-force model (nx 13, input = force) — the sim experiments' setting.
    force_rate_limit: tuple[float, ...] | None = None
    # Per-thruster |force| bound (N). None = max_thrust on every thruster. Set to the
    # session's realizable force k*(amps_limit - I0) when thrusters are operationally
    # clamped (e.g. the depth-hold scenario's horizontal pairs at the deadzone).
    thrust_limits: tuple[float, ...] | None = None
    # Round-trip command latency of the deployed chain (s): solve overrun + mapper/teleop
    # hops + CAN + T200 spin-up. When > 0, WallScanMPC rolls the measured state forward
    # through the in-flight command history before each solve (Smith-predictor style).
    # Field-identified 2026-08-19 (bag 23_03_29 + _probe_field_2303.py): ~0.4 s of
    # unmodeled dead time turned even the well-damped depth-hold weights into a
    # 16 cm / 4.2 s cap-to-cap limit cycle; the predictor collapses it to ~1 cm and is
    # robust to +-0.2 s misestimates (over-prediction with ZERO actual delay is benign).
    command_latency_s: float = 0.0

    def to_json(self, path: str) -> None:
        """Dump for the hardware side: the Jetson has no env to call :func:`from_env` on,
        so the sim exports the authoritative values once and the vehicle loads the file
        (``marinelab/config/pkrc_plant_fixed_tam.json`` is the committed export)."""
        import dataclasses
        import json

        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=1)

    @classmethod
    def from_json(cls, path: str) -> PlantParams:
        import json

        with open(path) as f:
            raw = json.load(f)
        raw["rigid_body_inertia"] = tuple(raw["rigid_body_inertia"])
        raw["coriolis_inertia"] = tuple(raw["coriolis_inertia"])
        raw["added_mass"] = tuple(raw["added_mass"])
        raw["linear_damping"] = tuple(raw["linear_damping"])
        raw["quadratic_damping"] = tuple(raw["quadratic_damping"])
        raw["center_of_buoyancy"] = tuple(raw["center_of_buoyancy"])
        raw["allocation_matrix"] = tuple(tuple(r) for r in raw["allocation_matrix"])
        # Optional actuator-model fields (absent in pre-2026-08-19 exports -> legacy model).
        for key in ("force_rate_limit", "thrust_limits"):
            if raw.get(key) is not None:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    @classmethod
    def from_env(cls, env) -> PlantParams:
        """Read the parameters off a constructed ``WallScanEnv`` (the authoritative source)."""
        h, t = env._hydro, env._thruster
        inertia = np.asarray(env._robot.data.default_inertia[0].cpu().numpy()).reshape(3, 3)
        return cls(
            mass=float(env._robot.data.default_mass[0].item()),
            rigid_body_inertia=(float(inertia[0, 0]), float(inertia[1, 1]), float(inertia[2, 2])),
            coriolis_inertia=tuple(float(v) for v in h.rigid_body_inertia[0].cpu().numpy()),
            added_mass=tuple(float(v) for v in h.added_mass_matrix[0].diagonal().cpu().numpy()),
            linear_damping=tuple(float(v) for v in h.linear_damping[0].cpu().numpy()),
            quadratic_damping=tuple(float(v) for v in h.quadratic_damping[0].cpu().numpy()),
            buoyancy_force=float(h.buoyancy_force[0].item()),
            center_of_buoyancy=tuple(float(v) for v in h.center_of_buoyancy[0].cpu().numpy()),
            max_thrust=float(t.cfg.thrust_coefficient),
            allocation_matrix=tuple(tuple(float(v) for v in row) for row in t.cfg.allocation_matrix),
        )


# ---------------------------------------------------------------------------
# CasADi helpers
# ---------------------------------------------------------------------------


def _quat_to_rot(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        ca.horzcat(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        ca.horzcat(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _omega_mat(w):
    p, q, r = w[0], w[1], w[2]
    return ca.vertcat(
        ca.horzcat(0, -p, -q, -r),
        ca.horzcat(p, 0, r, -q),
        ca.horzcat(q, -r, 0, p),
        ca.horzcat(r, q, -p, 0),
    )


def _quat_normalize(q):
    return q / ca.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2 + 1e-12)


def _continuous_dynamics(x, u, prm: PlantParams, B, d_world):
    """xdot = f(x, u), replicating ``HydrodynamicsModel.compute_forces`` + PhysX integration.

    The quaternion is normalized on the way IN, not just on the way out. See
    :func:`_error_expr` for why that matters — without it a >~50 deg attitude error makes the
    solver diverge.
    """
    quat, v_b, w_b = _quat_normalize(x[3:7]), x[7:10], x[10:13]
    nu = ca.vertcat(v_b, w_b)
    R = _quat_to_rot(quat)

    D_l = ca.DM(np.asarray(prm.linear_damping, dtype=float))
    D_q = ca.DM(np.asarray(prm.quadratic_damping, dtype=float))
    A = ca.DM(np.asarray(prm.added_mass, dtype=float))
    I_cor = ca.DM(np.asarray(prm.coriolis_inertia, dtype=float))
    I_rb = ca.DM(np.asarray(prm.rigid_body_inertia, dtype=float))

    damping = D_l * nu + D_q * ca.fabs(nu) * nu

    # C_RB: torque only, with the plant's (cfg-fallback) inertia. C_A: added-mass coupling.
    c_rb_torque = ca.cross(w_b, I_cor * w_b)
    ma = A * nu
    ma_lin, ma_ang = ma[0:3], ma[3:6]
    c_a_force = -ca.cross(ma_lin, w_b)
    c_a_torque = -(ca.cross(ma_lin, v_b) + ca.cross(ma_ang, w_b))
    coriolis = ca.vertcat(c_a_force, c_rb_torque + c_a_torque)

    hydro = -(coriolis + damping)

    up_b = R.T @ ca.DM([0.0, 0.0, 1.0])
    f_buoy = prm.buoyancy_force * up_b
    m_buoy = ca.cross(ca.DM(np.asarray(prm.center_of_buoyancy, dtype=float)), f_buoy)
    f_grav = R.T @ ca.DM([0.0, 0.0, -prm.mass * GRAVITY])
    f_dist = R.T @ d_world

    tau = ca.DM(np.asarray(B, dtype=float)) @ u
    f_tot = tau[0:3] + hydro[0:3] + f_buoy + f_grav + f_dist
    m_tot = tau[3:6] + hydro[3:6] + m_buoy

    # PhysX integrates world-frame velocity, so the body-frame derivative carries the
    # transport term; gyroscopic torque is absent because the asset sets
    # enable_gyroscopic_forces=False and the plant supplies C_RB externally instead.
    v_dot = f_tot / prm.mass - ca.cross(w_b, v_b)
    w_dot = m_tot / I_rb
    return ca.vertcat(R @ v_b, 0.5 * _omega_mat(w_b) @ quat, v_dot, w_dot)


def _error_expr(x, p, cfg: mref.WallScanMPCCfg):
    """CasADi mirror of :func:`mpc_reference.wallscan_errors`. Layout must stay identical.

    The quaternion is normalized here even though the dynamics emit a unit one. Nothing
    constrains the SQP's shooting variables to the unit sphere — only the (linearized)
    dynamics equality does, and that is satisfied approximately during iteration. Measured
    2026-07-30 (``isaaclab/logs/_probe_wallscan_mpc.py``): from a 100 deg heading error the
    mid-horizon ``|q|`` climbed to 1.082 and STAYED there, and every RTI iteration from the
    second on returned ``ACADOS_MINSTEP`` — because ``_quat_to_rot`` of a non-unit quaternion
    is not a rotation (it scales by ``|q|^2``), so the residuals and their Jacobians stop
    agreeing and the Hessian degenerates. The failure threshold sat between 40 deg (fine) and
    60 deg (broken). Normalizing on the way in makes the whole formulation invariant to the
    quaternion's scale, so the drift is harmless and the dynamics equality pulls it back.
    """
    r_des, z_ref, s_ref, v_tan_des, v_z_des, theta_anchor, s_anchor = (p[i] for i in range(NP_REF))
    pos, quat, v_b, w_b = x[0:3], _quat_normalize(x[3:7]), x[7:10], x[10:13]
    eps = 1e-6

    dist_xy = ca.sqrt(pos[0] ** 2 + pos[1] ** 2 + eps)
    h_out = ca.vertcat(pos[0], pos[1]) / dist_xy            # outward radial = the wall normal
    t_hat = ca.vertcat(-h_out[1], h_out[0])                 # +theta tangent

    R = _quat_to_rot(quat)
    h_act = ca.vertcat(R[0, 0], R[1, 0])
    h_act = h_act / ca.sqrt(h_act[0] ** 2 + h_act[1] ** 2 + eps)

    v_w = R @ v_b
    v_xy = ca.vertcat(v_w[0], v_w[1])

    theta = ca.atan2(pos[1], pos[0])
    dtheta = ca.atan2(ca.sin(theta - theta_anchor), ca.cos(theta - theta_anchor))
    s = s_anchor + dtheta * cfg.tank_radius

    roll = ca.atan2(2 * (quat[0] * quat[1] + quat[2] * quat[3]),
                    1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
    pitch = ca.asin(ca.fmax(-1.0, ca.fmin(1.0, 2 * (quat[0] * quat[2] - quat[3] * quat[1]))))

    return ca.vertcat(
        dist_xy - r_des,
        pos[2] - z_ref,
        s - s_ref,
        h_out.T @ v_xy,
        t_hat.T @ v_xy - v_tan_des,
        v_w[2] - v_z_des,
        h_act[0] - h_out[0],
        h_act[1] - h_out[1],
        roll,
        pitch,
        w_b[0],
        w_b[1],
    )


# ---------------------------------------------------------------------------
# OCP construction
# ---------------------------------------------------------------------------


def build_ocp(prm: PlantParams, cfg: mref.WallScanMPCCfg, *, N: int, sensitivity: bool,
              model_name: str, code_export_dir: str, werr_init, wu_init,
              solver_opts: dict | None = None, rk4_substeps: int = 1):
    if not _ACADOS_AVAILABLE:  # pragma: no cover
        raise ImportError("wallscan NMPC needs casadi + acados_template") from _ACADOS_IMPORT_ERROR

    B = np.asarray(prm.allocation_matrix, dtype=float)
    if B.ndim != 2 or B.shape[0] != 6:
        raise ValueError(f"allocation_matrix must be (6, nu); got {B.shape}")
    nu = int(B.shape[1])
    dt = cfg.dt_mpc

    # Actuator model switch (module docstring): rate mode augments the state with the
    # applied per-thruster force and turns the input into a normalized force rate.
    rate = None if prm.force_rate_limit is None else \
        np.asarray(prm.force_rate_limit, dtype=float).reshape(nu)
    if rate is not None and np.any(rate <= 0.0):
        raise ValueError(f"force_rate_limit must be positive; got {rate}")
    f_lim = np.full(nu, prm.max_thrust) if prm.thrust_limits is None else \
        np.asarray(prm.thrust_limits, dtype=float).reshape(nu)
    nx = NX + (nu if rate is not None else 0)

    ocp = AcadosOcp()
    model = AcadosModel()
    model.name = model_name
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    p = ca.SX.sym("p", NP_REF + ND)
    model.p = p
    d_world = p[NP_REF:NP_REF + ND]

    if rate is not None:
        def f(xx, uu):
            # thrust comes from the force STATE; the input uu is the normalized rate.
            x13_dot = _continuous_dynamics(xx[0:NX], xx[NX:], prm, B, d_world)
            return ca.vertcat(x13_dot, ca.DM(rate) * uu)
    else:
        def f(xx, uu):
            return _continuous_dynamics(xx, uu, prm, B, d_world)

    # Sub-stepped RK4. The yaw axis is stiff for a 0.05 s stage: Izz = 0.393 kg*m^2 against
    # 24 N*m of Mz authority is 61 rad/s^2, so one RK4 step per stage lands its four
    # evaluations at wildly different yaw rates (and the quadratic yaw damping, 8*|w|*w, is
    # 72 N*m at 3 rad/s -- larger than the torque producing it). That makes the stage map
    # inaccurate exactly where the solver needs a good linearization.
    h = dt / max(1, int(rk4_substeps))
    x_next = x
    for _ in range(max(1, int(rk4_substeps))):
        k1 = f(x_next, u)
        k2 = f(x_next + 0.5 * h * k1, u)
        k3 = f(x_next + 0.5 * h * k2, u)
        k4 = f(x_next + h * k3, u)
        x_next = x_next + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next = ca.vertcat(x_next[0:3], _quat_normalize(x_next[3:7]), x_next[7:])
    model.disc_dyn_expr = x_next
    model.x, model.u = x, u

    y_err = _error_expr(x[0:NX], p, cfg)

    if sensitivity:
        p_glob = ca.SX.sym("p_global", NE + nu)
        model.p_global = p_glob
        w_err, w_u = p_glob[0:NE], p_glob[NE:NE + nu]
        ocp.cost.cost_type = ocp.cost.cost_type_e = "EXTERNAL"
        model.cost_expr_ext_cost = 0.5 * (ca.dot(w_err, y_err * y_err) + ca.dot(w_u, u * u))
        model.cost_expr_ext_cost_e = 0.5 * ca.dot(w_err, y_err * y_err)
    else:
        ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
        model.cost_y_expr = ca.vertcat(y_err, u)
        model.cost_y_expr_e = y_err

    if rate is not None:
        # input = normalized force rate; the force itself is a box-constrained state.
        ocp.constraints.lbu = np.full(nu, -1.0)
        ocp.constraints.ubu = np.full(nu, 1.0)
        ocp.constraints.idxbx = NX + np.arange(nu)
        ocp.constraints.lbx = -f_lim
        ocp.constraints.ubx = f_lim
        ocp.constraints.idxbx_e = NX + np.arange(nu)
        ocp.constraints.lbx_e = -f_lim
        ocp.constraints.ubx_e = f_lim
    else:
        ocp.constraints.lbu = -f_lim
        ocp.constraints.ubu = f_lim
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.x0 = np.zeros(nx)
    ocp.constraints.x0[3] = 1.0  # unit quaternion, or the first solve starts from a singular R

    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * dt
    ocp.parameter_values = np.zeros(NP_REF + ND)

    w_full = np.concatenate([np.asarray(werr_init, float), np.asarray(wu_init, float)])
    if sensitivity:
        ocp.p_global_values = w_full
    else:
        ocp.cost.W = np.diag(w_full)
        ocp.cost.W_e = np.diag(w_full[:NE])
        ocp.cost.yref = np.zeros(NE + nu)
        ocp.cost.yref_e = np.zeros(NE)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_mu0 = 1e3
    if sensitivity:
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.qp_solver_ric_alg = 0
        ocp.solver_options.qp_solver_cond_ric_alg = 0
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = 0
        ocp.solver_options.with_solution_sens_wrt_params = True
    else:
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
    # Caller overrides last, so a robustness knob (globalization, Levenberg-Marquardt
    # regularization, a different QP backend) can be swept without editing this builder.
    for key, value in (solver_opts or {}).items():
        setattr(ocp.solver_options, key, value)

    ocp.code_export_directory = code_export_dir
    return ocp, nu


# Interior-friendly starting weights. Deliberately NOT the task's aggressive tracking-reward
# scales: large weights saturate the thrusters, and saturated solutions have zero (useless)
# sensitivities. Ordered as mpc_reference.ERROR_NAMES.
DEFAULT_WERR = (
    40.0,   # radial  - the mission constraint
    40.0,   # z
    40.0,   # s
    10.0,   # v_rad
    5.0,    # v_tan
    5.0,    # v_z
    20.0,   # head_x  - heading dominates wall-range accuracy (see mpc_reference docstring)
    20.0,   # head_y
    20.0,   # roll    - the sway leg's parasitic axis under the FIXED TAM
    20.0,   # pitch
    0.5,    # w_x
    0.5,    # w_y
)
DEFAULT_WU = 0.01


def advance_actuator(f_act, rate_cmd, rate_limit, f_lim, dt) -> np.ndarray:
    """One control tick of the applied-force bookkeeping (pure, natively tested).

    ``f_act`` (N) integrates the solver's normalized rate command at the modelled slew
    limit over the CONTROL tick (``cfg.step_dt``, not ``dt_mpc`` — the loop re-solves
    every tick), clipped to the per-thruster force bounds. Because the modelled slew is
    the conservative end of the measured teleop ramp, the published one-tick-ahead force
    is always realizable and this open-loop integration stays honest.
    """
    f = np.asarray(f_act, dtype=float) + \
        np.asarray(rate_limit, dtype=float) * np.clip(np.asarray(rate_cmd, dtype=float), -1.0, 1.0) * float(dt)
    return np.clip(f, -np.asarray(f_lim, dtype=float), np.asarray(f_lim, dtype=float))


class WallScanMPC:
    """Nominal + sensitivity acados solver pair for the wallscan task, single initial state."""

    def __init__(self, prm: PlantParams, cfg: mref.WallScanMPCCfg, *, N: int = 30,
                 sens_node: int | None = None, sens_nodes=None,
                 rti_iters: int = 8, werr_init=None, wu_init=None,
                 code_export_root: str | None = None, build: bool = True, generate: bool = True,
                 with_sensitivity: bool = True, verbose: bool = False,
                 solver_opts: dict | None = None, model_suffix: str = "",
                 rk4_substeps: int = 5):
        self.prm, self.cfg, self.N = prm, cfg, int(N)
        self.rti_iters = max(1, int(rti_iters))
        # State sensitivity can be evaluated at ANY stage, and Diff-WMPC needs several.
        # Measured 2026-07-31: with the loss at one node 1.0 s out, the learner optimized the
        # fast axes (attitude, ~0.1 s time constants) and drove w_radial down to 2-3, because
        # radial drift barely moves inside 1 s so its weight looks worthless to the gradient.
        # Summing over nodes rebalances that by construction: attitude error has converged by
        # ~0.3 s so late nodes contribute almost pure position/heading error, while the radial
        # contribution accumulates across every node.
        if sens_nodes is not None:
            self.sens_nodes = [int(k) for k in sens_nodes]
        elif sens_node is not None:
            self.sens_nodes = [int(sens_node)]
        else:
            self.sens_nodes = sorted({max(1, (self.N * f) // 6) for f in (1, 2, 4, 6)})
        for k in self.sens_nodes:
            if not 1 <= k <= self.N:
                raise ValueError(f"sens node {k} outside [1, {self.N}]")
        # Back-compat: single-node callers (the Phase 3 baseline) still read .sens_node.
        self.sens_node = self.sens_nodes[-1]

        root = code_export_root or os.path.join(os.getcwd(), "c_generated_code_wallscan")
        os.makedirs(root, exist_ok=True)
        werr = np.asarray(werr_init if werr_init is not None else DEFAULT_WERR, float)
        if werr.size != NE:
            raise ValueError(f"werr_init must have length {NE}; got {werr.size}")

        nom_ocp, self.nu = build_ocp(prm, cfg, N=self.N, sensitivity=False,
                                     model_name=f"pkrc_wallscan_mpc{model_suffix}",
                                     code_export_dir=os.path.join(root, f"nominal{model_suffix}"),
                                     werr_init=werr, wu_init=np.full(6, DEFAULT_WU),
                                     solver_opts=solver_opts, rk4_substeps=rk4_substeps)
        wu = np.asarray(wu_init if wu_init is not None else np.full(self.nu, DEFAULT_WU), float)
        self.n_pglobal = NE + self.nu
        # Actuator-rate mode bookkeeping (module docstring): the applied-force state lives
        # HERE so every caller (adapters, ROS node, sim runner) stays 13-D at the seam.
        self.rate_mode = prm.force_rate_limit is not None
        self.nx = NX + (self.nu if self.rate_mode else 0)
        self._rate_lim = None if not self.rate_mode else \
            np.asarray(prm.force_rate_limit, float).reshape(self.nu)
        self._f_lim = np.full(self.nu, prm.max_thrust) if prm.thrust_limits is None else \
            np.asarray(prm.thrust_limits, float).reshape(self.nu)
        self._tick_dt = float(cfg.step_dt)
        self._f_act = np.zeros(self.nu)
        # Command-latency predictor (PlantParams.command_latency_s): one RK4 tick of the
        # 13-D dynamics, applied over the in-flight command history before each solve.
        self._pred_n = int(round(max(0.0, float(prm.command_latency_s)) / self._tick_dt))
        self._pred_fn = None
        self._u_hist: list[np.ndarray] = []
        if self._pred_n > 0:
            x_sym = ca.SX.sym("x", NX)
            f_sym = ca.SX.sym("f", self.nu)
            B_arr = np.asarray(prm.allocation_matrix, dtype=float)
            h = self._tick_dt / 2
            xn = x_sym
            for _ in range(2):
                k1 = _continuous_dynamics(xn, f_sym, prm, B_arr, ca.DM.zeros(ND))
                k2 = _continuous_dynamics(xn + 0.5 * h * k1, f_sym, prm, B_arr, ca.DM.zeros(ND))
                k3 = _continuous_dynamics(xn + 0.5 * h * k2, f_sym, prm, B_arr, ca.DM.zeros(ND))
                k4 = _continuous_dynamics(xn + h * k3, f_sym, prm, B_arr, ca.DM.zeros(ND))
                xn = xn + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                xn = ca.vertcat(xn[0:3], _quat_normalize(xn[3:7]), xn[7:13])
            self._pred_fn = ca.Function("latency_pred", [x_sym, f_sym], [xn])
            self._u_hist = [np.zeros(self.nu) for _ in range(self._pred_n)]
        self.nominal = AcadosOcpSolver(nom_ocp, json_file=os.path.join(root, f"nominal{model_suffix}.json"),
                                       build=build, generate=generate, verbose=verbose)
        self.sens = None
        if with_sensitivity:
            sens_ocp, _ = build_ocp(prm, cfg, N=self.N, sensitivity=True,
                                    model_name="pkrc_wallscan_mpc_sens",
                                    code_export_dir=os.path.join(root, "sensitivity"),
                                    werr_init=werr, wu_init=wu, rk4_substeps=rk4_substeps)
            self.sens = AcadosOcpSolver(sens_ocp, json_file=os.path.join(root, "sensitivity.json"),
                                        build=build, generate=generate, verbose=verbose)
        self._warned = 0

    def param_matrix(self, ref: dict, theta_anchor: float, s_anchor: float,
                     d_world=None) -> np.ndarray:
        """(N+1, NP_REF+ND) per-stage parameters from a :func:`mpc_reference.reference_preview`.

        Stage-varying ``p`` is the whole point: the solver sees the ramp move over the horizon
        instead of a single frozen setpoint, which is what lets it decelerate INTO a phase
        endpoint rather than overshoot it.
        """
        d = np.zeros(ND) if d_world is None else np.asarray(d_world, float).reshape(ND)
        P = np.zeros((self.N + 1, NP_REF + ND))
        for k in range(self.N + 1):
            P[k, 0] = self.cfg.r_des
            P[k, 1] = float(ref["z_ref"][k])
            P[k, 2] = float(ref["s_ref"][k])
            P[k, 3] = float(ref["v_tan_des"][k])
            P[k, 4] = float(ref["v_z_des"][k])
            P[k, 5] = theta_anchor
            P[k, 6] = s_anchor
            P[k, 7:] = d
        return P

    def reset_actuator(self, f_act=None) -> None:
        """Re-anchor the applied-force state (rate mode) — zero matches a fresh enable,
        where the mapper watchdog has ramped the real currents down while disabled."""
        self._f_act = np.zeros(self.nu) if f_act is None else \
            np.asarray(f_act, float).reshape(self.nu).copy()

    def solve(self, x0, P: np.ndarray, weights, *, want_sensitivity: bool = True,
              init_state_traj: bool = False) -> dict:
        x0 = np.asarray(x0, float).reshape(-1)
        if self._pred_fn is not None:
            if init_state_traj:  # fresh enable: nothing is in flight (mapper held zeros)
                self._u_hist = [np.zeros(self.nu) for _ in range(self._pred_n)]
            x13 = x0[:NX]
            for f_in_flight in self._u_hist:  # oldest first
                x13 = np.asarray(self._pred_fn(x13, f_in_flight)).reshape(NX)
            x0 = np.concatenate([x13, x0[NX:]])
        if self.rate_mode:
            if init_state_traj:
                self.reset_actuator()
            if x0.size == NX:  # callers stay 13-D; the force state is ours to append
                x0 = np.concatenate([x0, self._f_act])
        w = np.asarray(weights, float).reshape(-1)
        if w.size != self.n_pglobal:
            raise ValueError(f"weights must have length {self.n_pglobal}; got {w.size}")

        W, W_e = np.diag(w), np.diag(w[:NE])
        for k in range(self.N):
            self.nominal.cost_set(k, "W", W)
            self.nominal.set(k, "p", P[k])
        self.nominal.cost_set(self.N, "W", W_e)
        self.nominal.set(self.N, "p", P[self.N])
        self.nominal.set(0, "lbx", x0)
        self.nominal.set(0, "ubx", x0)
        if init_state_traj:
            # A cold start leaves distant stages at zero, which is far from x0 and makes the
            # dynamics residual enormous (and the quaternion there is not even unit norm).
            for k in range(self.N + 1):
                self.nominal.set(k, "x", x0)
                if k < self.N:
                    self.nominal.set(k, "u", np.zeros(self.nu))

        status = 0
        for _ in range(self.rti_iters):
            status = self.nominal.solve()
        if status != 0 and self._warned < 10:
            print(f"[WallScanMPC] nominal solve status={status}")
            self._warned += 1

        u0 = np.asarray(self.nominal.get(0, "u"), float)
        x_traj = np.stack([np.asarray(self.nominal.get(k, "x"), float) for k in range(self.N + 1)])
        rate0 = None
        if self.rate_mode:
            # ``u0`` keeps FORCE semantics for every consumer: publish the one-tick-ahead
            # planned force and advance the internal applied-force state to match.
            rate0 = u0
            self._f_act = advance_actuator(self._f_act, rate0, self._rate_lim,
                                           self._f_lim, self._tick_dt)
            u0 = self._f_act.copy()
        if self._pred_fn is not None:  # the command just issued is now in flight
            self._u_hist.append(np.asarray(u0, float).copy())
            self._u_hist.pop(0)
        out = {
            "u0": u0,
            "u0_cmd": np.clip(u0 / self.prm.max_thrust, -1.0, 1.0),
            "rate0": rate0,
            "f_act": self._f_act.copy() if self.rate_mode else None,
            "x_node": x_traj[self.sens_node],
            "x_nodes": {k: x_traj[k] for k in self.sens_nodes},
            "x_traj": x_traj,
            "status": int(status),
            "sens_x": None,
            "sens_x_nodes": None,
            "sens_u": None,
        }
        if want_sensitivity and self.sens is not None and status == 0:
            try:
                self.sens.set_p_global_and_precompute_dependencies(w)
                for k in range(self.N + 1):
                    self.sens.set(k, "p", P[k])
                self.sens.set(0, "lbx", x0)
                self.sens.set(0, "ubx", x0)
                self.sens.load_iterate_from_flat_obj(self.nominal.store_iterate_to_flat_obj())
                # One factorization, then one cheap solve per requested node.
                self.sens.setup_qp_matrices_and_factorize()
                sens_x_nodes = {}
                for k in self.sens_nodes:
                    dx = self.sens.eval_solution_sensitivity(k, "p_global", return_sens_x=True,
                                                             return_sens_u=False, sanity_checks=True)
                    sens_x_nodes[k] = np.asarray(dx["sens_x"], float)
                # Control sensitivity only at stage 0: under full condensing acados returns
                # zero for intermediate stages, so asking for more would silently add nothing.
                du = self.sens.eval_solution_sensitivity(0, "p_global", return_sens_x=False,
                                                         return_sens_u=True, sanity_checks=True)
                out["sens_x_nodes"] = sens_x_nodes
                out["sens_x"] = sens_x_nodes[self.sens_node]
                out["sens_u"] = np.asarray(du["sens_u"], float)
            except Exception as exc:  # pragma: no cover
                if self._warned < 10:
                    print(f"[WallScanMPC] sensitivity failed: {exc!r}")
                    self._warned += 1
                out["sens_x"] = out["sens_x_nodes"] = out["sens_u"] = None
        return out
