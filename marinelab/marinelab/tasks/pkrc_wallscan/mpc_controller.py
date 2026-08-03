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

Not modelled: the first-order thruster lag (``tau_up = 0.1 s``, ``tau_down = 0.05 s``). The
commanded force is assumed to be applied instantly. That is a deliberate first-iteration
choice — it costs some overshoot at reference reversals, and the fix (augmenting the state
with the thruster state, nx 13 -> 19) is a known follow-up if the baseline shows it.

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

    ocp = AcadosOcp()
    model = AcadosModel()
    model.name = model_name
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", nu)
    p = ca.SX.sym("p", NP_REF + ND)
    model.p = p
    d_world = p[NP_REF:NP_REF + ND]

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
        x_next = ca.vertcat(x_next[0:3], _quat_normalize(x_next[3:7]), x_next[7:13])
    model.disc_dyn_expr = x_next
    model.x, model.u = x, u

    y_err = _error_expr(x, p, cfg)

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

    ocp.constraints.lbu = np.full(nu, -prm.max_thrust)
    ocp.constraints.ubu = np.full(nu, prm.max_thrust)
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.x0 = np.zeros(NX)
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

    def solve(self, x0, P: np.ndarray, weights, *, want_sensitivity: bool = True,
              init_state_traj: bool = False) -> dict:
        x0 = np.asarray(x0, float).reshape(-1)
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
        out = {
            "u0": u0,
            "u0_cmd": np.clip(u0 / self.prm.max_thrust, -1.0, 1.0),
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
