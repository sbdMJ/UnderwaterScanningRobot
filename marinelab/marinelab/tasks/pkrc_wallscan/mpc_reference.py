# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Reference generation + error vector for the wallscan NMPC (Diff-WMPC port).

Pure torch/numpy — no isaaclab, no pxr, no acados imports — so every sign convention
and every closed-form geometry relation below is unit-testable without the sim app or a
compiled solver. Same convention as ``geometry.py`` / ``sensors.py`` /
``scan_state_machine.py`` / ``eval_metrics.py``; see ``__init__.py``'s lazy map.

## Why the cylinder-orbit MPC ports here at all

The wallscan task is the *inside-out* version of the cylinder-orbit task in
Underwater-Actor-Critic-Model-Predictive-Control (``cylinder_orbit_mpc_diff.py``):

| | orbit | wallscan |
|:--|:--|:--|
| vehicle sits | OUTSIDE the cylinder | INSIDE the tank |
| holds | distance ``radius`` from the axis | wall clearance ``d_ref`` |
| faces | the axis (inward) | the wall (outward) |
| moves | tangentially at ``v_tan``, constant z | boustrophedon in (z, s) |

So the reference maps with three substitutions and no new math:

* ``center`` -> the tank axis (origin of the tank-local frame),
* ``r_des = tank_radius - d_ref`` (4.5 m for R=6, d_ref=1.5),
* ``yaw_offset = pi`` -> flips the orbit code's inward heading target to outward.

The last one is worth spelling out because it is the cheapest way to get the sign right:
the orbit error uses ``h_des = (center - pos)/|.|`` (inward) and then rotates it by
``yaw_offset``, so ``yaw_offset = pi`` yields exactly ``+pos_xy/|pos_xy|`` — the outward
radial, which on a cylinder *is* the wall normal. Referencing heading to the radial
(rather than to a bearing latched by the spin search) is closed-loop and cannot drift:
position turns -> radial turns. That is the same conclusion ``wallscan_env`` reached
empirically on 07-27 (``_yaw_ref_cur = theta_gt``), except an MPC can also see it coming
over the horizon and supply the curvature yaw rate ``v_tan / r`` without a feedforward term.

## Why heading dominates wall-distance accuracy (the reason this module exists)

A single-beam echo sounder returns ONE range along the beam. On a cylinder wall, a beam
tilted by ``phi`` off the outward radial reads (``sonar_range``, exact):

    t(r, phi) = -r*cos(phi) + sqrt(R^2 - r^2*sin^2(phi))
              ~ (R - r) * [1 + phi^2 * r/(2R)]        (small phi)

Two consequences drive the whole control design:

1. The error is **second order in phi**, so a tracking reward barely feels it — which is
   exactly how the RL policy could crab a full loop at "zero reward loss" (07-27 audit,
   bearing 5-84 deg off per episode).
2. The error is **one-sided**: ``t >= R - r`` always, with equality only at ``phi = 0``.
   A crabbing vehicle therefore believes it is FARTHER from the wall than it is and
   closes in. At R=6, clearance 1.5 m: +1.7 cm at 10 deg, +16.5 cm at 30 deg,
   +40 cm at 45 deg. It is a bias toward the wall, not a symmetric error.

``heading_offset_from_range`` inverts the relation in closed form, which is also the
observability backbone for a future (r, phi) estimator: given r, ONE range fixes |phi|.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Error-vector layout. This order IS the p_global column order in the acados
# controller and the weight-policy head order, so it must never be permuted.
# ---------------------------------------------------------------------------
ERROR_NAMES = (
    "radial",    # 0  |pos_xy| - r_des          (wall clearance;  + = too far from axis = too close to wall)
    "z",         # 1  pos_z - z_ref
    "s",         # 2  arc length - s_ref        (marker-frame, measured AT THE WALL like _s_gt)
    "v_rad",     # 3  outward radial speed      (target always 0)
    "v_tan",     # 4  tangential speed - target (sway leg: ramp rate; vertical leg: 0)
    "v_z",       # 5  heave speed - target      (vertical leg: ramp rate; sway leg: 0)
    "head_x",    # 6  body +x xy vs outward radial, x component
    "head_y",    # 7  ... y component
    "roll",      # 8
    "pitch",     # 9
    "w_x",       # 10
    "w_y",       # 11
)
NE = len(ERROR_NAMES)

# Number of per-stage reference parameters passed to the solver as model.p, in this order:
#   [r_des, z_ref, s_ref, v_tan_des, v_z_des, theta_anchor, s_anchor, d_world(3)]
# theta_anchor/s_anchor carry the env's *unwrapped* arc-length bookkeeping into the solve
# (see arc_length below); d_world is the disturbance-observer residual force.
NP_REF = 7
ND = 3


@dataclass
class WallScanMPCCfg:
    """Geometry + pacing constants the reference generator needs.

    Defaults mirror ``WallScanEnvCfg`` / ``ScanCfg`` so a mismatch is visible in one place.
    """

    tank_radius: float = 6.0
    d_ref: float = 1.5
    z_top: float = 8.5
    z_bottom: float = 1.0
    sway_step: float = 1.0
    # Ramp rates are per CONTROL step (0.02 s) in ScanCfg; the MPC stage is dt_mpc, so the
    # preview advances ref_step * (dt_mpc / step_dt) per stage. Kept as the raw ScanCfg
    # values plus the two dts so the conversion happens in exactly one place.
    ref_step: float = 0.004      # m per control step -> 0.2 m/s heave
    ref_step_s: float = 0.002    # m per control step -> 0.1 m/s sway
    step_dt: float = 0.02
    dt_mpc: float = 0.05

    @property
    def r_des(self) -> float:
        """Target distance from the tank axis (NOT the wall clearance)."""
        return self.tank_radius - self.d_ref

    @property
    def ramp_per_stage_z(self) -> float:
        return self.ref_step * (self.dt_mpc / self.step_dt)

    @property
    def ramp_per_stage_s(self) -> float:
        step = self.ref_step_s if self.ref_step_s > 0.0 else self.ref_step
        return step * (self.dt_mpc / self.step_dt)


# ---------------------------------------------------------------------------
# Sonar geometry: the exact range/heading coupling and its inverse
# ---------------------------------------------------------------------------


def sonar_range(r: torch.Tensor, phi: torch.Tensor, tank_radius: float) -> torch.Tensor:
    """Range a radial-mounted single beam reads at axis distance ``r``, beam offset ``phi``.

    Exact solution of |p + t*d|^2 = R^2 for the positive root, with ``phi`` the angle
    between the beam and the OUTWARD radial. Reduces to ``R - r`` at ``phi = 0`` and is
    monotonically increasing in |phi| — see the module docstring for why the one-sidedness
    matters. ``r`` and ``phi`` broadcast.
    """
    R = tank_radius
    return -r * torch.cos(phi) + torch.sqrt((R * R - (r * torch.sin(phi)) ** 2).clamp(min=0.0))


def heading_offset_from_range(
    measured: torch.Tensor, r: torch.Tensor, tank_radius: float, eps: float = 1e-9
) -> torch.Tensor:
    """|beam offset| implied by a measured range at known axis distance ``r``.

    Closed-form inverse of :func:`sonar_range`: squaring ``t + r*cos(phi) = sqrt(R^2 -
    r^2 sin^2 phi)`` cancels every trig term but one, leaving

        cos(phi) = (R^2 - r^2 - t^2) / (2*t*r)

    Returns |phi| in [0, pi] (the sign is unobservable from one beam). This is why a
    single echo sounder is not hopeless: with ``r`` from dead reckoning, ONE range pins
    |phi| exactly, and DVL motion over time resolves the remaining ambiguity in (r, phi).
    """
    R = tank_radius
    cos_phi = (R * R - r * r - measured * measured) / (2.0 * measured.clamp(min=eps) * r.clamp(min=eps))
    return torch.arccos(cos_phi.clamp(-1.0, 1.0))


def clearance_bias(r: torch.Tensor, phi: torch.Tensor, tank_radius: float) -> torch.Tensor:
    """How much a perpendicular-assuming controller over-estimates its wall clearance."""
    return sonar_range(r, phi, tank_radius) - (tank_radius - r)


# ---------------------------------------------------------------------------
# Arc length in the virtual-marker frame
# ---------------------------------------------------------------------------


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def arc_length(
    pos_xy: torch.Tensor, theta_anchor: torch.Tensor, s_anchor: torch.Tensor, tank_radius: float
) -> torch.Tensor:
    """Continuous marker-frame arc length, measured AT THE WALL (matches ``_s_gt``).

    ``wallscan_env`` keeps s continuous across full loops by accumulating GT angle
    INCREMENTS (``:418``), which is stateful and therefore not expressible inside a solver.
    Instead the caller passes the current ``(theta_anchor, s_anchor)`` pair each solve —
    the angle and arc length at the present step — and the horizon measures relative to
    it. A 1.5-3 s horizon covers ~0.3 m ~ 0.05 rad, so the wrap at the antipode can never
    be reached inside one solve and the wrapped form is exact.

    Note the multiplier is ``tank_radius`` (6.0), not the vehicle's own radius: s is the
    swath position along the WALL, which is the quantity the scan pattern covers.
    """
    theta = torch.atan2(pos_xy[..., 1], pos_xy[..., 0])
    return s_anchor + _wrap_to_pi(theta - theta_anchor) * tank_radius


# ---------------------------------------------------------------------------
# Reference preview over the horizon
# ---------------------------------------------------------------------------


def ramp_preview(
    ramp0: torch.Tensor, target: torch.Tensor, per_stage: float, n_stages: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Saturated ramp from ``ramp0`` toward ``target``, sampled over ``n_stages + 1`` nodes.

    ``scan_state_machine.step`` slews the emitted reference by at most ``ref_step`` per
    control step; over a horizon with a fixed phase that is a pure deterministic sequence,
    so it has a closed form (no python loop, and differentiable/batched for free):

        ramp_k = target - sign(target - ramp0) * max(0, |target - ramp0| - k*per_stage)

    Returns ``(ramp[..., n_stages+1], disp[..., n_stages+1])`` where ``disp`` is the
    reference DISPLACEMENT per stage (metres, not m/s) — divide by ``dt_mpc`` to get the
    feedforward speed target, which is what :func:`reference_preview` does. It goes to
    zero once the ramp has arrived, so an arrived reference commands zero speed.

    Assumes the phase does NOT advance inside the horizon — true for tens-of-seconds
    phases and a few-second horizon, wrong for the handful of steps straddling a
    transition (a refinement, not a correctness issue: the terminal node simply keeps
    holding the old endpoint).
    """
    k = torch.arange(n_stages + 1, dtype=ramp0.dtype, device=ramp0.device)
    delta = target - ramp0
    remaining = (delta.abs().unsqueeze(-1) - k * per_stage).clamp(min=0.0)
    ramp = target.unsqueeze(-1) - torch.sign(delta).unsqueeze(-1) * remaining
    disp = torch.zeros_like(ramp)
    disp[..., :-1] = ramp[..., 1:] - ramp[..., :-1]
    disp[..., -1] = disp[..., -2]  # terminal node has no successor; hold the last displacement
    return ramp, disp


def phase_targets(phase: torch.Tensor, s_ref: torch.Tensor, z_hold: torch.Tensor, cfg: WallScanMPCCfg):
    """(z_target, s_target) the current phase is ramping toward.

    Mirrors ``scan_state_machine.step``'s emitted reference exactly: SWAY phases hold the
    latched depth and slew s toward ``s_ref``; vertical phases hold s and slew z toward
    ``z_bottom`` (DESCEND) or ``z_top`` (ASCEND).
    """
    is_sway = (phase == 1) | (phase == 3)
    is_ascend = phase == 2
    z_target = torch.where(
        is_sway, z_hold, torch.where(is_ascend, torch.full_like(z_hold, cfg.z_top), torch.full_like(z_hold, cfg.z_bottom))
    )
    return z_target, s_ref


def reference_preview(
    phase: torch.Tensor,
    z_ramp0: torch.Tensor,
    s_ramp0: torch.Tensor,
    s_ref: torch.Tensor,
    z_hold: torch.Tensor,
    cfg: WallScanMPCCfg,
    n_stages: int,
) -> dict[str, torch.Tensor]:
    """Per-stage reference the MPC should track over its horizon.

    This is the structural advantage an MPC has over the RL policy here: the policy only
    ever observes the CURRENT ``z_ref``/``s_ref``, so buoyancy + momentum carried its
    ascent 0.1-0.2 m past the band (rollout traces; the ``z > tank_height + 0.2`` kill
    bound had to be widened for exactly this). With the ramp previewed over the horizon
    the solver starts decelerating before the endpoint instead of after it.

    Returns dict of [..., n_stages+1] tensors: ``z_ref``, ``s_ref``, ``v_z_des``, ``v_tan_des``.
    """
    z_target, s_target = phase_targets(phase, s_ref, z_hold, cfg)
    z_ref, z_disp = ramp_preview(z_ramp0, z_target, cfg.ramp_per_stage_z, n_stages)
    s_ref_p, s_disp = ramp_preview(s_ramp0, s_target, cfg.ramp_per_stage_s, n_stages)
    inv_dt = 1.0 / cfg.dt_mpc
    return {
        "z_ref": z_ref,
        "s_ref": s_ref_p,
        "v_z_des": z_disp * inv_dt,
        "v_tan_des": s_disp * inv_dt,
    }


# ---------------------------------------------------------------------------
# Error vector (torch reference implementation)
# ---------------------------------------------------------------------------


def _quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz -> (..., 3, 3) body->world rotation."""
    q = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def _quat_to_rp(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    return roll, pitch


def wallscan_errors(
    x: torch.Tensor,
    *,
    z_ref: torch.Tensor,
    s_ref: torch.Tensor,
    v_tan_des: torch.Tensor,
    v_z_des: torch.Tensor,
    theta_anchor: torch.Tensor,
    s_anchor: torch.Tensor,
    cfg: WallScanMPCCfg,
    eps: float = 1e-6,
) -> torch.Tensor:
    """The NE-dim wallscan error vector. ``x`` is [..., 13] = [pos(3), quat(wxyz,4), v_b(3), w_b(3)].

    Layout is :data:`ERROR_NAMES`. Every entry is an ERROR (target already subtracted), so
    the MPC cost is a plain weighted sum of squares and the Diff-WMPC weight policy maps
    one weight per entry.

    Sign conventions worth stating once:

    * ``radial`` is ``|pos_xy| - r_des``, so POSITIVE means the vehicle is farther from the
      axis, i.e. CLOSER to the wall. Wall clearance is ``tank_radius - |pos_xy|``.
    * ``head_*`` compares the body +x axis projected on xy against the OUTWARD radial, so
      zero error means the sonar looks straight at the wall (``phi = 0`` in
      :func:`sonar_range`) and the range it reads is the true clearance.
    * ``v_tan`` is positive in the ``+theta`` (counter-clockwise) direction, matching the
      sign of ``arc_length`` and hence of ``sway_dir = +1``.
    """
    pos = x[..., 0:3]
    quat = x[..., 3:7]
    v_b = x[..., 7:10]
    w_b = x[..., 10:13]

    pos_xy = pos[..., 0:2]
    dist_xy = torch.sqrt((pos_xy * pos_xy).sum(-1) + eps)
    h_out = pos_xy / dist_xy.unsqueeze(-1)                      # outward radial (the wall normal)
    t_hat = torch.stack([-h_out[..., 1], h_out[..., 0]], dim=-1)  # +theta tangent

    R = _quat_to_rot(quat)
    h_act = torch.stack([R[..., 0, 0], R[..., 1, 0]], dim=-1)     # body +x in world, xy part
    h_act = h_act / h_act.norm(dim=-1, keepdim=True).clamp(min=eps)

    v_w = torch.einsum("...ij,...j->...i", R, v_b)
    v_xy = v_w[..., 0:2]
    roll, pitch = _quat_to_rp(quat)

    return torch.stack(
        [
            dist_xy - cfg.r_des,
            pos[..., 2] - z_ref,
            arc_length(pos_xy, theta_anchor, s_anchor, cfg.tank_radius) - s_ref,
            (h_out * v_xy).sum(-1),
            (t_hat * v_xy).sum(-1) - v_tan_des,
            v_w[..., 2] - v_z_des,
            h_act[..., 0] - h_out[..., 0],
            h_act[..., 1] - h_out[..., 1],
            roll,
            pitch,
            w_b[..., 0],
            w_b[..., 1],
        ],
        dim=-1,
    )


def sway_tilt_equilibrium(v_sway: float, tam_arm: float, buoy_force: float, cob_z: float,
                          lin_damp_y: float, quad_damp_y: float) -> float:
    """Steady tilt [rad] a sustained sway leg forces, given the TAM's parasitic moment arm.

    The sway thrusters are the only source of the parasitic moment, so the required lateral
    force is set purely by drag and the tilt by the buoyancy restoring couple:

        F_y = lin*v + quad*v^2            (drag the sway leg must overcome)
        M    = tam_arm * F_y              (parasitic moment, TAM row entry)
        sin(theta) = M / (buoy_force * cob_z)

    For the shipped PKRC numbers this predicts 2.30 deg at the measured 0.123 m/s sway,
    against 2.20 deg measured by ``eval_metrics`` — i.e. the trained policy already sits at
    this equilibrium. Whether that equilibrium is escapable depends on WHICH axis the
    parasitic moment lands on: as shipped it is ``My`` (pitch), and pitch has no other
    actuator, so it is a hard floor. If the arm belongs on ``Mx`` (roll) instead — which is
    what a pure +y force at ``z = -0.09`` gives, and what the TAM's own Fy/Mz rows imply —
    the heave differential (``Mx = 0.16 * dF``) can cancel it for ``dF = -tam_arm/0.16 * F_y``,
    about 8.6 N at 0.123 m/s, well inside a 40 N thruster. Verify against the Stonefish
    ``pkrc_tam.yaml`` before trusting either reading.
    """
    f_y = lin_damp_y * v_sway + quad_damp_y * v_sway * v_sway
    return math.asin(min(1.0, tam_arm * f_y / (buoy_force * cob_z)))
