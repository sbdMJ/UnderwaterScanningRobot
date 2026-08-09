# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Sensor synthesis + wall-frame estimation, shared by the NMPC runner and the Diff-WMPC trainer.

Pure torch/numpy — no isaaclab — so it is unit-testable and both entry scripts can hold the
identical sensor model.

## Why this is a shared module and not copied code

Diff-WMPC learns cost weights for the state quality it is trained against. Training on ground
truth and evaluating on an estimate is a train/test mismatch, and it showed up as a measurable
regression: under GT + placeholder sensors the learned policy beat hand-tuned weights on crab,
but re-measured under the estimator with datasheet sensors it came out **0.074 +- 0.022 deg
WORSE** (decided across 3 paired seeds). Closing that gap means the trainer must drive the same
estimator the evaluator does.

Which is only trustworthy if it is literally the same code. The first version of this logic
lived only in ``run_wallscan_mpc.py`` and silently omitted the INS attitude bias, so every tilt
figure reported from it was optimistic until that was caught. Two copies would drift again.

## What it does per step

1. Synthesize the sensor stream from ground truth using ``SensorCfg`` (or
   ``SensorCfgDatasheet``): sonar with the finite beam cone, DVL with scale-factor error and
   zero-order hold at the published ping rate, INS attitude/rate with their separate noise and
   bias channels, depth, and UKF-M gated by its validity band.
2. Feed :class:`~marinelab.tasks.pkrc_wallscan.wall_frame_ekf.WallFrameEKF`.
3. Rebuild the 13-element MPC state from the ESTIMATE, so nothing downstream sees truth.

Step 3 is where the bearing comes from ``theta_hat = theta_0 + s_hat / R`` rather than a second
integrator: the arc-length state already carries it, and a parallel integrator would drift away
from it for no reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from . import geometry
from .sensors import SensorCfg, att_noise, dvl_hold_steps, gyro_noise, ukfm_in_range
from .wall_frame_ekf import WallFrameEKF, WallFrameEKFCfg

__all__ = ["WallFrameEstimator", "EstimatorOutput"]


def _quat_from_rpy(roll: float, pitch: float, yaw: float, device) -> torch.Tensor:
    """(1, 4) wxyz from XYZ euler angles. Local so this module stays isaaclab-free."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return torch.tensor([[cr * cp * cy + sr * sp * sy,
                          sr * cp * cy - cr * sp * sy,
                          cr * sp * cy + sr * cp * sy,
                          cr * cp * sy - sr * sp * cy]], device=device)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class EstimatorOutput:
    """One step of estimated state, in the shapes the MPC and the scan machine want."""

    pos: torch.Tensor        # (1, 3) tank-local
    quat: torch.Tensor       # (1, 4) wxyz
    v_b: torch.Tensor        # (1, 3) body linear velocity
    w_b: torch.Tensor        # (1, 3) body angular velocity
    theta_anchor: torch.Tensor   # (1,) bearing the arc length is measured from
    s_anchor: torch.Tensor       # (1,) arc length at that bearing
    err_r: float = 0.0
    err_phi: float = 0.0
    err_s: float = 0.0


@dataclass
class WallFrameEstimator:
    """Owns the sensor realization, the DVL hold and the wall-frame EKF for ONE vehicle."""

    scfg: SensorCfg
    tank_radius: float
    step_dt: float
    sonar_mount_nom: torch.Tensor        # (1, 2) body-frame mount, as the env holds it
    sonar_yaw_nom: float
    gyro_bias: float
    rng: np.random.Generator
    ekf: WallFrameEKF | None = None
    # TRUE mount, when it differs from nominal. The stress-DR task jitters the physical mount
    # per episode (+-8 cm / +-0.04 rad) while any real filter still has to predict with the
    # surveyed value, so the range is SYNTHESIZED from these and the EKF keeps ``*_nom``. Leave
    # them None and the two coincide, which is the no-DR case.
    sonar_mount_true: torch.Tensor | None = None
    sonar_yaw_true: float | None = None
    # Model the instruments as the 3-AXIS devices they actually are. Left False by default
    # because every published number in this repo was produced with the passthrough below, and
    # turning it on makes the estimate strictly worse (which is the point).
    #
    # The passthrough: `v_est = v_b.clone()` replaces only [:2], and `w_est = w_b.clone()` only
    # [2], so body v_z and the roll/pitch rates reach the controller as GROUND TRUTH. A DVL-A50
    # is a 4-beam instrument and a 3DM-GV7 a 3-axis IMU -- neither has a privileged axis -- so
    # this is under-modelling, and it flatters two things in particular: the vertical axis is the
    # wallscan's primary motion, and `wrench_observer`'s force channel is computed FROM v_z, so
    # any policy conditioned on the observer is being fed a partly ground-truth signal.
    full_3axis: bool = False
    _bias: dict = field(default_factory=dict)
    _dvl_held: np.ndarray | None = None
    _theta0: float = 0.0
    n_ukfm: int = 0
    n_gated: int = 0

    # -- lifecycle ---------------------------------------------------------
    def reset(self, pos: torch.Tensor, quat: torch.Tensor, yaw: float) -> None:
        """Start a new episode/segment: fresh filter, fresh per-episode biases.

        The filter is seeded AT TRUTH on purpose. This measures how an estimate DEGRADES over a
        mission, not how it converges from a cold start — the spin search would own that, and
        the controllers here skip it.
        """
        if self.ekf is not None:
            self.n_ukfm += self.ekf.n_ukfm
            self.n_gated += self.ekf.n_gated
        r0 = float(torch.linalg.norm(pos[0, :2]))
        th0 = float(torch.atan2(pos[0, 1], pos[0, 0]))
        self._theta0 = th0
        self.ekf = WallFrameEKF(WallFrameEKFCfg(
            tank_radius=self.tank_radius, r_sonar=self.scfg.sonar_noise,
            sonar_mount_pos=self.scfg.sonar_mount_pos,
            sonar_yaw_offset=self.scfg.sonar_yaw_offset,
            sonar_beam_half_angle=self.scfg.sonar_beam_half_angle,
            initial=(r0, _wrap(yaw - th0), 0.0),
        ))
        self._dvl_held = None
        s = self.scfg
        u = self.rng.uniform
        self._bias = {
            "sonar": float(u(-s.sonar_bias_dr, s.sonar_bias_dr)) if s.sonar_bias_dr else 0.0,
            "depth": float(u(-s.depth_bias_dr, s.depth_bias_dr)) if s.depth_bias_dr else 0.0,
            "dvl": u(-s.dvl_bias_dr, s.dvl_bias_dr, size=2) if s.dvl_bias_dr else np.zeros(2),
            "dvl_scale": u(-s.dvl_scale_dr, s.dvl_scale_dr, size=2) if s.dvl_scale_dr else np.zeros(2),
            "att": u(-s.ins_att_bias_dr, s.ins_att_bias_dr, size=2) if s.ins_att_bias_dr else np.zeros(2),
            # UKF-M is the only ABSOLUTE fix in the filter, so a constant offset on it is not
            # averaged away like noise -- it pulls (r, phi) off by that much for the episode.
            "ukfm": u(-s.ukfm_bias_dr, s.ukfm_bias_dr, size=2) if s.ukfm_bias_dr else np.zeros(2),
        }

    def harvest(self) -> tuple[int, int]:
        """Totals across segments; the per-segment filter is replaced on every reset."""
        n_u, n_g = self.n_ukfm, self.n_gated
        if self.ekf is not None:
            n_u += self.ekf.n_ukfm
            n_g += self.ekf.n_gated
        return n_u, n_g

    # -- per step ----------------------------------------------------------
    def step(self, i: int, pos: torch.Tensor, quat: torch.Tensor, v_b: torch.Tensor,
             w_b: torch.Tensor, roll: float, pitch: float, yaw: float,
             s_gt: float) -> EstimatorOutput:
        """Synthesize sensors from truth, update the filter, return the ESTIMATED state."""
        s, dev = self.scfg, pos.device
        r_true = float(torch.linalg.norm(pos[0, :2]))
        theta = float(torch.atan2(pos[0, 1], pos[0, 0]))
        phi_true = _wrap(yaw - theta)

        # Sonar: the env's own Cartesian model, so truth and the filter's prediction come from
        # the same geometry, including the finite beam cone.
        # Synthesized from the TRUE mount; the filter predicts with the nominal one, so a mount
        # DR shows up as the unmodeled range error a real vehicle would actually have.
        mount = self.sonar_mount_nom if self.sonar_mount_true is None else self.sonar_mount_true
        m_yaw = self.sonar_yaw_nom if self.sonar_yaw_true is None else self.sonar_yaw_true
        sonar_true = float(geometry.sonar_wall_distance(
            pos[:, :2], torch.tensor([yaw], device=dev), mount,
            m_yaw, self.tank_radius, s.sonar_beam_half_angle)[0])
        sonar = sonar_true + float(self.rng.normal(0, s.sonar_noise)) + self._bias["sonar"]

        # DVL: scale-factor error (the A50 spec is a percentage) then zero-order hold at the
        # published ping rate, because a 15 Hz reading cannot be re-averaged every 50 Hz step.
        hold = dvl_hold_steps(s, self.step_dt)
        n_ax = 3 if self.full_3axis else 2
        if self._dvl_held is None or i % hold == 0:
            # Same scale-factor error, noise and hold on every measured axis: the spec is per
            # instrument, not per axis. Both per-episode draws are 2-vectors historically, so the
            # third axis takes the mean of the drawn pair rather than silently running unbiased.
            def _ax(name):
                v = np.asarray(self._bias[name], dtype=float).reshape(-1)
                return v[:n_ax] if v.size >= n_ax else np.append(v, float(v.mean()))
            self._dvl_held = (v_b[0, :n_ax].cpu().numpy() * (1.0 + _ax("dvl_scale"))
                              + self.rng.normal(0, s.dvl_noise, n_ax) + _ax("dvl"))
        v_meas = self._dvl_held

        gyro = float(w_b[0, 2]) + float(self.rng.normal(0, gyro_noise(s))) + self.gyro_bias
        w_xy_meas = None
        if self.full_3axis:
            w_xy_meas = (w_b[0, :2].cpu().numpy()
                         + self.rng.normal(0, gyro_noise(s), 2) + self.gyro_bias)
        z_meas = float(pos[0, 2]) + float(self.rng.normal(0, s.depth_noise)) + self._bias["depth"]

        ukfm = None
        if bool(ukfm_in_range(pos[:, 2], s)[0]):
            ukfm = (r_true + float(self.rng.normal(0, s.ukfm_noise)) + float(self._bias["ukfm"][0]),
                    phi_true + float(self.rng.normal(0, s.ukfm_noise)) + float(self._bias["ukfm"][1]))
        self.ekf.step(v_bx=float(v_meas[0]), v_by=float(v_meas[1]), gyro_z=gyro,
                      dt=self.step_dt, sonar=sonar, ukfm=ukfm)

        # Rebuild the MPC state from the estimate alone.
        th_hat = self._theta0 + self.ekf.s / self.tank_radius
        roll_m = roll + float(self.rng.normal(0, att_noise(s))) + float(self._bias["att"][0])
        pitch_m = pitch + float(self.rng.normal(0, att_noise(s))) + float(self._bias["att"][1])
        pos_est = torch.tensor([[self.ekf.r * math.cos(th_hat),
                                 self.ekf.r * math.sin(th_hat), z_meas]], device=dev)
        v_est = v_b.clone()
        v_est[0, :len(v_meas)] = torch.as_tensor(v_meas, dtype=v_b.dtype, device=dev)
        w_est = w_b.clone()
        w_est[0, 2] = gyro
        if w_xy_meas is not None:
            w_est[0, :2] = torch.as_tensor(w_xy_meas, dtype=w_b.dtype, device=dev)
        return EstimatorOutput(
            pos=pos_est,
            quat=_quat_from_rpy(roll_m, pitch_m, th_hat + self.ekf.phi, dev).to(quat.dtype),
            v_b=v_est, w_b=w_est,
            theta_anchor=torch.tensor([th_hat], device=dev),
            s_anchor=torch.tensor([self.ekf.s], device=dev),
            err_r=self.ekf.r - r_true,
            err_phi=_wrap(self.ekf.phi - phi_true),
            err_s=self.ekf.s - s_gt,
        )
