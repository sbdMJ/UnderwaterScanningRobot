# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Sensor-to-state estimation chain: raw sensor samples in, ``VehicleState`` out.

This is the half of the sim2real seam that faces the sensors. In simulation the experiment
runner synthesizes :class:`SensorSample` from ground truth plus the sensor model (exactly as
``run_wallscan_mpc.py --state ekf`` does); on the vehicle a ROS node fills the same fields
from the DVL / INS / Ping1D / pressure / UKF-M drivers. Everything downstream — the EKF, the
state reconstruction, every controller — is byte-for-byte the same code in both worlds.

The reconstruction mirrors ``run_wallscan_mpc.py``: the wall-frame EKF estimates
``(r, phi, s)``; absolute wall angle is re-anchored as ``theta_hat = theta0 + s / R`` so no
second integrator can drift against the arc-length state, and absolute xy/yaw are rebuilt
from it for the MPC's tank-frame state vector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .types import VehicleState


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(w, x, y, z) from XYZ Euler angles — matches ``isaaclab.utils.math``."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


@dataclass
class SensorSample:
    """One control tick's raw sensor readings, body-frame where applicable."""

    v_bx: float  # DVL surge velocity
    v_by: float  # DVL sway velocity
    gyro_z: float  # INS yaw rate
    sonar: float  # Ping1D wall range
    depth: float  # pressure depth (tank z)
    roll: float  # INS attitude
    pitch: float  # INS attitude
    v_bz: float = 0.0  # DVL heave velocity (unused by the EKF, passed through to the state)
    gyro_x: float = 0.0  # INS roll rate (passed through)
    gyro_y: float = 0.0  # INS pitch rate (passed through)
    ukfm: tuple[float, float] | None = None  # (r, phi) surface-marker fix when visible
    stamp: float = 0.0


class WallFrameStateEstimator:
    """Wall-frame EKF wrapped into the ``SensorSample -> VehicleState`` contract."""

    def __init__(self, ekf_cfg=None):
        self._cfg = ekf_cfg
        self._ekf = None
        self._theta0 = 0.0

    def reset(self, *, r0: float, phi0: float, theta0: float) -> None:
        """Start a fresh filter seeded at (r0, phi0), anchored at absolute wall angle theta0."""
        from marinelab.tasks.pkrc_wallscan.wall_frame_ekf import WallFrameEKF, WallFrameEKFCfg

        if self._cfg is None:
            self._cfg = WallFrameEKFCfg()
        # dataclasses.replace would lose subclasses; mutate a per-reset copy of initial only.
        cfg = self._cfg
        cfg.initial = (float(r0), float(phi0), 0.0)
        self._ekf = WallFrameEKF(cfg)
        self._theta0 = float(theta0)

    @property
    def ekf(self):
        return self._ekf

    @property
    def theta_hat(self) -> float:
        """Absolute wall angle, re-derived from the arc-length state."""
        return self._theta0 + self._ekf.s / self._ekf.cfg.tank_radius

    @property
    def s_hat(self) -> float:
        return self._ekf.s

    def step(self, sample: SensorSample, dt: float) -> VehicleState:
        if self._ekf is None:
            raise RuntimeError("call reset(r0=..., phi0=..., theta0=...) before step()")
        self._ekf.step(v_bx=sample.v_bx, v_by=sample.v_by, gyro_z=sample.gyro_z, dt=dt,
                       sonar=sample.sonar, ukfm=sample.ukfm)
        th = self.theta_hat
        yaw = th + self._ekf.phi
        r = self._ekf.r
        return VehicleState(
            pos_w=np.array([r * math.cos(th), r * math.sin(th), sample.depth]),
            quat_wb=_quat_from_euler_xyz(sample.roll, sample.pitch, yaw),
            lin_vel_b=np.array([sample.v_bx, sample.v_by, sample.v_bz]),
            ang_vel_b=np.array([sample.gyro_x, sample.gyro_y, sample.gyro_z]),
            stamp=sample.stamp,
        )
