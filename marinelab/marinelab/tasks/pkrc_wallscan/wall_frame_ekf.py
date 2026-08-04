# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Wall-relative state estimator: the (r, phi, s) filter the wallscan NMPC actually needs.

Pure numpy — no isaaclab, no acados — so the observability arguments below are unit-testable
without the sim app. Same convention as ``mpc_reference`` / ``geometry`` / ``sensors``.

## The restructuring that makes this tractable

``mpc_controller`` formulates the MPC over an absolute tank-frame state, which makes the
estimator look impossible: the heading reference is the outward radial, the outward radial
needs absolute xy, and the only absolute-xy sensor is UKF-M (surface ArUco marker), whose
validity is depth-limited while the scan runs from z = 8.5 m down to z = 1 m.

But go through ``mpc_reference.ERROR_NAMES`` one entry at a time and the absolute frame never
actually appears:

| error entry | what it needs |
|:---|:---|
| ``radial`` | ``r - r_des``, and ``r = tank_radius - clearance`` — a sonar quantity |
| ``head_x/y`` | chord ``2*sin(phi/2)`` of the beam offset — ``phi`` itself, not yaw |
| ``s`` | accumulated ``d(theta)``, i.e. ``integral(v_tan / r)`` — an increment, not a bearing |
| ``v_rad``, ``v_tan`` | body velocity rotated by ``phi``, NOT by absolute yaw |
| ``z``, ``roll``, ``pitch``, ``w_x``, ``w_y`` | depth sensor and INS, directly |

So three wall-relative states suffice: ``r`` (distance from the tank axis), ``phi = yaw -
theta`` (beam offset from the outward radial) and ``s`` (arc length along the wall). Absolute
xy and absolute yaw are an artifact of the MPC's state parameterization, not a requirement of
the task.

That reframing is what makes the filter small and auditable, but it does NOT make UKF-M
optional — the measurements below show the opposite: the sonar carries ``r`` well and heading
barely at all, so UKF-M is load-bearing for ``phi``.

## Observability, and where it runs out

The sonar gives ONE range, which by the closed form in
``mpc_reference.heading_offset_from_range`` constrains the pair::

    cos(phi) = (R^2 - r^2 - t^2) / (2*t*r)

— a 1-D manifold in ``(r, phi)``, so a single range at a single instant is not enough. Motion
resolves it: with DVL velocity and gyro rate as inputs, the range's evolution separates the
two. Concretely ``dr/dt = v_rad`` depends on ``phi`` through the rotation, so tangential
motion at a nonzero ``phi`` changes the range in a way pure radial motion cannot mimic.

The practical consequence is that the excitation is **phase-dependent**:

* SWAY legs translate tangentially -> the range moves -> ``(r, phi)`` are well observed.
* DESCEND/ASCEND legs move vertically, which changes neither ``r`` nor ``phi`` -> no new
  information, and the filter coasts on gyro integration for the whole ~39 s leg.

MEASURED 2026-07-31 (``scripts/replay_wall_frame_ekf.py`` over a logged 180 s run, 5 seeds).
The gyro was the obvious suspect and it is NOT the binding constraint: dropping the bias from
1e-3 rad/s to exactly zero leaves phi RMSE at 5.99 deg, so something else sets that floor.

It is the sonar's own information content. The range signature of a small beam offset is
second order, so at r = 4.5 m::

    phi =  1 deg ->  0.17 mm      phi = 10 deg -> 17.3 mm
    phi =  5 deg ->  4.29 mm      phi = 20 deg -> 70.6 mm

against 50 mm of white noise and — the killer — a per-episode sonar bias of up to 100 mm,
which never averages out. A 5 deg misalignment moves the range by 1/23 of the bias. **A single
echo sounder cannot observe small heading offsets, and no filter tuning changes that.**

What makes the design work anyway is UKF-M, which covers most of the scan depth band. With the
corrected validity gate (see below) phi RMSE is 1.71 deg, and 0.96 deg at a realistic gyro
spec — better than the trained RL policy's 1.50 deg crab. So UKF-M is load-bearing here, and
the sonar's job is ``r``, which it does well (RMSE 0.04 m).

## The UKF-M validity gate in sensors.py is inverted

``sensors.py:91`` gates on ``|z| < ukfm_valid_max_depth``, and wallscan's ``z`` is HEIGHT above
the tank floor (spawn 9.5, ``z_bottom`` 1.0). As written the fix is therefore valid in the
lower 8 m and invalid in the top 2 m. The physical setup is the opposite: the ArUco marker
sits at the water surface and the camera looks up, so validity is bounded by DEPTH BELOW THE
SURFACE. Two other places in this codebase already assume the correct direction —
``wallscan_env.py:702`` ("Spawn at the water surface ... UKF-M marker view also valid there")
and ``:485`` (the operating ceiling is surface - 1 m because *too close* to the marker
degrades the fix). The correct condition with the shipped 8.0 m parameter is
``tank_height - z < 8`` i.e. ``z > 2.0``, optionally with a near limit ``z < 9.0``.

Impact of the inversion on existing results is small but real: both gates cover most of the
7.5 m scan band and differ only in WHICH end goes blind (as shipped: the top, including the
ASCEND endpoint at 8.5; corrected: the bottom, including the DESCEND endpoint at 1.0).
Measured phi RMSE 1.56 deg vs 1.71 deg. Fixing it is still worth doing — a controller that
loses its absolute fix exactly at a phase endpoint is not what anyone designed.

"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["WallFrameEKF", "WallFrameEKFCfg", "sonar_range", "sonar_jacobian",
           "mounted_sonar_range", "mounted_sonar_jacobian", "dvl_hold_steps",
           "gyro_drift_over_leg", "range_information"]


def sonar_range(r: float, phi: float, tank_radius: float) -> float:
    """Range a beam AT THE BODY ORIGIN reads at axis distance ``r``, beam offset ``phi``.

    Scalar mirror of ``mpc_reference.sonar_range``. Kept for the idealised geometry arguments;
    the filter uses :func:`mounted_sonar_range`, because the real transducer is not at the
    body origin and pretending otherwise cost a measurable bias — see below.
    """
    inner = max(tank_radius**2 - (r * math.sin(phi)) ** 2, 0.0)
    return -r * math.cos(phi) + math.sqrt(inner)


def _sensor_pose(r: float, phi: float, mount_xy, yaw_offset: float):
    """Wall-relative pose of the TRANSDUCER given the vehicle's ``(r, phi)``.

    The beam does not start at the body origin: PKRC's sonar sits at body ``(0.10, 0.0)`` m
    (``SensorCfg.sonar_mount_pos``). Working in the outward-radial / tangential basis of the
    VEHICLE, the mount resolves as

        radial      r_s = r + mx*cos(phi) - my*sin(phi)
        tangential  a_s =     mx*sin(phi) + my*cos(phi)

    (body +x makes angle ``phi`` with the outward radial, so ``body_x . h = cos phi``,
    ``body_x . t = sin phi``, ``body_y . h = -sin phi``, ``body_y . t = cos phi``.)

    From there the sensor has its OWN axis distance and its own outward radial, rotated by
    ``delta`` from the vehicle's, so the beam offset measured at the sensor is
    ``phi + yaw_offset - delta``. Feeding those two numbers to the body-origin formula gives
    the exact mounted range — no new geometry, just the right arguments.

    Returns ``(r_sensor, phi_sensor, r_s, a_s)``; the last two are handed to the Jacobian so it
    does not recompute them.
    """
    mx, my = float(mount_xy[0]), float(mount_xy[1])
    c, s = math.cos(phi), math.sin(phi)
    r_s = r + mx * c - my * s
    a_s = mx * s + my * c
    r_sen = math.sqrt(r_s * r_s + a_s * a_s)
    delta = math.atan2(a_s, r_s)
    return r_sen, phi + yaw_offset - delta, r_s, a_s


def mounted_sonar_range(r: float, phi: float, tank_radius: float,
                        mount_xy=(0.0, 0.0), yaw_offset: float = 0.0,
                        beam_half_angle: float = 0.0) -> float:
    """Range the beam reads with the transducer at body-frame ``mount_xy``, azimuth ``yaw_offset``.

    Equivalent to ``geometry.sonar_wall_distance`` (the Cartesian model the env samples the
    truth from), expressed in the filter's wall-relative state. Cross-checked against it in
    ``tests/pkrc_wallscan/test_wall_frame_ekf.py``.

    MEASURED cost of omitting this (2026-07-31): with the mount left out, the filter reconciles
    a range that is ~10 cm shorter than its model predicts by inflating ``r``, and it settled at
    +5.3 cm of ``r`` bias — i.e. the controller held station 5 cm farther from the wall than
    asked. Worse, the error GREW when the sonar got better: replacing the placeholder noise
    (50 mm) with the Ping1D figure (7.5 mm) made the filter trust the sonar more and pulled
    ``r`` further toward the missing offset, taking r-RMSE from ~5 cm to ~10 cm. An imprecise
    sensor had been masking the modelling gap.
    """
    r_sen, phi_sen, _, _ = _sensor_pose(r, phi, mount_xy, yaw_offset)
    return sonar_range(r_sen, _effective_beam_offset(phi_sen, beam_half_angle), tank_radius)


def _effective_beam_offset(phi_sen: float, beam_half_angle: float) -> float:
    """Angle the SHORTEST return inside a ``2*beam_half_angle`` cone actually comes from.

    The tank wall is concave from the inside and range grows with |angle off the radial|, so an
    echo sounder reporting the nearest return sees ``max(0, |phi| - half_angle)``: zero for any
    misalignment the cone still spans.

    So inside the cone the range carries no DIRECT heading information — ``d(eff)/d(phi) = 0``,
    and the Jacobian's ``phi_sensor`` term vanishes. It is not exactly zero overall, though: with
    a non-zero mount, rotating the vehicle swings the transducer sideways and changes its own
    axis distance, leaking a little heading sensitivity through the lever arm. Measured at
    phi = 5 deg with the 0.10 m mount: 0.0085 m/rad, ~40x smaller than just outside the cone.
    """
    if beam_half_angle <= 0.0:
        return phi_sen
    return max(abs(phi_sen) - beam_half_angle, 0.0)


def mounted_sonar_jacobian(r: float, phi: float, tank_radius: float,
                           mount_xy=(0.0, 0.0), yaw_offset: float = 0.0,
                           beam_half_angle: float = 0.0) -> tuple[float, float]:
    """``(dt/dr, dt/dphi)`` of :func:`mounted_sonar_range`, by the chain rule through
    ``(r_sensor, phi_sensor)``.

    Analytic rather than finite-difference because the filter runs at 50 Hz and a wrong
    Jacobian is silent — it just makes the update the wrong size. Verified against central
    differences in the tests, the same way the acados sensitivities were.
    """
    mx, my = float(mount_xy[0]), float(mount_xy[1])
    r_sen, phi_sen, r_s, a_s = _sensor_pose(r, phi, mount_xy, yaw_offset)
    eff = _effective_beam_offset(phi_sen, beam_half_angle)
    dt_drs, dt_dps = sonar_jacobian(r_sen, eff, tank_radius)
    # d(eff)/d(phi_sen): 0 inside the cone, +-1 outside. Inside, the measurement is blind to the
    # beam offset, so the filter must not credit the update to phi through this path. (A residual
    # dependence survives via dr_sen/dphi when the mount is non-zero -- see _effective_beam_offset.)
    if beam_half_angle > 0.0:
        dt_dps *= 0.0 if abs(phi_sen) <= beam_half_angle else (1.0 if phi_sen > 0 else -1.0)

    r_sen2 = max(r_sen * r_sen, 1e-12)
    # d r_s/dphi = -a_s ,  d a_s/dphi = r_s - r  (differentiate the two lines in _sensor_pose)
    drsen_dr = r_s / max(r_sen, 1e-12)
    drsen_dphi = -a_s * r / max(r_sen, 1e-12)
    # delta = atan2(a_s, r_s) -> d(atan2)/dx = (r_s da_s - a_s dr_s)/r_sen^2
    ddelta_dr = -a_s / r_sen2
    ddelta_dphi = 1.0 - r * r_s / r_sen2
    dphisen_dr = -ddelta_dr
    dphisen_dphi = 1.0 - ddelta_dphi
    del mx, my
    return (dt_drs * drsen_dr + dt_dps * dphisen_dr,
            dt_drs * drsen_dphi + dt_dps * dphisen_dphi)


def sonar_jacobian(r: float, phi: float, tank_radius: float) -> tuple[float, float]:
    """``(dt/dr, dt/dphi)`` of :func:`sonar_range`.

    ``dt/dphi`` vanishes at ``phi = 0`` — the range is stationary in the beam offset, which is
    the same second-order insensitivity that let the RL policy crab a whole loop at nearly no
    reward cost. It means a single range is a WEAK heading measurement near alignment and the
    filter has to lean on the motion model there.
    """
    s, c = math.sin(phi), math.cos(phi)
    root = math.sqrt(max(tank_radius**2 - (r * s) ** 2, 1e-12))
    dt_dr = -c - r * s * s / root
    dt_dphi = r * s - r * r * s * c / root
    return dt_dr, dt_dphi


def range_information(r: float, phi: float, tank_radius: float) -> float:
    """|dt/dphi| — how much heading information one range carries at this configuration.

    Zero at ``phi = 0`` and growing with |phi|: the sonar can tell you that you are badly
    misaligned much better than it can confirm that you are aligned.
    """
    return abs(sonar_jacobian(r, phi, tank_radius)[1])


def gyro_drift_over_leg(gyro_bias: float, leg_seconds: float) -> float:
    """Heading error [rad] accumulated by open-loop gyro integration over an unexcited leg.

    The vertical legs give the range no new information, so ``phi`` is propagated by the gyro
    alone for their whole duration. At the shipped placeholder bias (0.02 rad/s) a 39 s
    descend leg accumulates 0.78 rad = 45 deg, which would reproduce exactly the crab-walk
    failure this project is trying to remove. At a real 3DM-GV7's ~10 deg/hr it is 0.1 deg.
    """
    return gyro_bias * leg_seconds


@dataclass
class WallFrameEKFCfg:
    tank_radius: float = 6.0
    # Process noise (per sqrt(second)) on r, phi, s. phi's entry has to cover the unmodelled
    # gyro bias, since a bias is not white and the filter has no state for it.
    q_r: float = 0.01
    q_phi: float = 0.02
    q_s: float = 0.02
    # Measurement noise: sonar_noise from sensors.py, and the UKF-M reset when it is valid.
    r_sonar: float = 0.05
    # Transducer pose in the body frame. Defaults mirror SensorCfg.sonar_mount_pos /
    # sonar_yaw_offset; leaving them at (0, 0) reproduces the pre-fix body-origin model.
    sonar_mount_pos: tuple[float, float] = (0.10, 0.0)
    sonar_yaw_offset: float = 0.0
    # Beam half-angle [rad]; 0 = single-ray (legacy). Ping1D is 25 deg full width -> 0.2182.
    sonar_beam_half_angle: float = 0.0
    r_ukfm_r: float = 0.05
    r_ukfm_phi: float = 0.05
    # Innovation gate in sonar-sigma units: a first return off a curved wall at a large beam
    # offset is unreliable, so reject outliers rather than let one bad ping move the state.
    gate_sigma: float = 4.0
    p0: tuple[float, float, float] = (0.25, 0.25, 0.25)
    initial: tuple[float, float, float] = field(default=(4.5, 0.0, 0.0))


class WallFrameEKF:
    """EKF over ``x = [r, phi, s]`` driven by DVL + gyro, corrected by sonar (and UKF-M).

    Single-vehicle (not batched) on purpose: it pairs with the sequential acados solve, and
    keeping it plain numpy makes the Jacobians auditable.
    """

    def __init__(self, cfg: WallFrameEKFCfg | None = None):
        self.cfg = cfg or WallFrameEKFCfg()
        self.x = np.asarray(self.cfg.initial, float).copy()
        self.P = np.diag(np.asarray(self.cfg.p0, float)).copy()
        self.n_gated = 0
        self.n_sonar = 0
        self.n_ukfm = 0

    # -- accessors ---------------------------------------------------------
    @property
    def r(self) -> float:
        return float(self.x[0])

    @property
    def phi(self) -> float:
        return float(self.x[1])

    @property
    def s(self) -> float:
        return float(self.x[2])

    @property
    def clearance(self) -> float:
        """Estimated TRUE wall clearance — the quantity a perpendicular-assuming controller
        gets wrong (always over-estimating) whenever ``phi != 0``."""
        return self.cfg.tank_radius - self.r

    def velocity_components(self, v_bx: float, v_by: float) -> tuple[float, float]:
        """Body planar velocity -> (outward radial, +theta tangential), rotated by ``phi``.

        Derivation: body +x makes angle ``phi`` with the outward radial, so
        ``body_x . h_out = cos(phi)``, ``body_x . t_hat = sin(phi)``,
        ``body_y . h_out = -sin(phi)``, ``body_y . t_hat = cos(phi)``.
        """
        c, s = math.cos(self.phi), math.sin(self.phi)
        return v_bx * c - v_by * s, v_bx * s + v_by * c

    # -- filter ------------------------------------------------------------
    def predict(self, v_bx: float, v_by: float, gyro_z: float, dt: float) -> None:
        """Propagate with DVL body velocity and gyro yaw rate over ``dt``."""
        r, phi = self.x[0], self.x[1]
        v_rad, v_tan = self.velocity_components(v_bx, v_by)
        r_safe = max(r, 1e-3)
        theta_dot = v_tan / r_safe

        self.x[0] = r + v_rad * dt
        self.x[1] = _wrap(phi + (gyro_z - theta_dot) * dt)
        self.x[2] = self.x[2] + self.cfg.tank_radius * theta_dot * dt
        self.x[0] = float(np.clip(self.x[0], 0.05, self.cfg.tank_radius - 0.01))

        c, s = math.cos(phi), math.sin(phi)
        dvrad_dphi = -v_bx * s - v_by * c
        dvtan_dphi = v_bx * c - v_by * s
        F = np.eye(3)
        F[0, 1] = dvrad_dphi * dt
        F[1, 0] = (v_tan / (r_safe * r_safe)) * dt          # d(-v_tan/r)/dr
        F[1, 1] = 1.0 - (dvtan_dphi / r_safe) * dt
        F[2, 0] = -self.cfg.tank_radius * (v_tan / (r_safe * r_safe)) * dt
        F[2, 1] = self.cfg.tank_radius * (dvtan_dphi / r_safe) * dt

        Q = np.diag([self.cfg.q_r**2, self.cfg.q_phi**2, self.cfg.q_s**2]) * dt
        self.P = F @ self.P @ F.T + Q

    def update_sonar(self, measured_range: float) -> bool:
        """Correct with one echo-sounder return. Returns False when the gate rejected it."""
        r, phi = self.x[0], self.x[1]
        mount, yaw_off = self.cfg.sonar_mount_pos, self.cfg.sonar_yaw_offset
        beam = self.cfg.sonar_beam_half_angle
        pred = mounted_sonar_range(r, phi, self.cfg.tank_radius, mount, yaw_off, beam)
        dt_dr, dt_dphi = mounted_sonar_jacobian(r, phi, self.cfg.tank_radius, mount, yaw_off, beam)
        H = np.array([[dt_dr, dt_dphi, 0.0]])
        R = np.array([[self.cfg.r_sonar**2]])
        innov = measured_range - pred
        S = float((H @ self.P @ H.T + R)[0, 0])
        if abs(innov) > self.cfg.gate_sigma * math.sqrt(max(S, 1e-12)):
            self.n_gated += 1
            return False
        K = (self.P @ H.T) / S
        self.x = self.x + (K * innov).reshape(-1)
        self.x[1] = _wrap(self.x[1])
        self.P = (np.eye(3) - K @ H) @ self.P
        self.n_sonar += 1
        return True

    def update_ukfm(self, r_meas: float, phi_meas: float) -> None:
        """Absolute reset from the surface-marker fix. Only call while UKF-M reports valid.

        This is the only place absolute information enters, and the whole point of the
        wall-relative formulation is that it is optional: without it the filter still runs on
        sonar + DVL + gyro, it just accumulates gyro bias over the unexcited vertical legs.
        """
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        R = np.diag([self.cfg.r_ukfm_r**2, self.cfg.r_ukfm_phi**2])
        innov = np.array([r_meas - self.x[0], _wrap(phi_meas - self.x[1])])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.x[1] = _wrap(self.x[1])
        self.P = (np.eye(3) - K @ H) @ self.P
        self.n_ukfm += 1

    def step(self, *, v_bx: float, v_by: float, gyro_z: float, dt: float,
             sonar: float | None = None, ukfm: tuple[float, float] | None = None) -> None:
        """One predict + available corrections, in the order a real loop would run them."""
        self.predict(v_bx, v_by, gyro_z, dt)
        if sonar is not None:
            self.update_sonar(sonar)
        if ukfm is not None:
            self.update_ukfm(*ukfm)


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
