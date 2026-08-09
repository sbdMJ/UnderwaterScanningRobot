# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Plain-data types shared by every wallscan controller (sim and hardware).

Everything here is numpy-only on purpose: these dataclasses are the sim2real seam. In
simulation the experiment runner fills them from the Isaac env; on the vehicle a ROS node
fills them from driver topics. Controllers must consume nothing richer than this module.

The 13-D state layout matches ``mpc_controller`` exactly:
``x = [pos_w(3), quat_wb(4, wxyz), lin_vel_b(3), ang_vel_b(3)]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

X13 = 13
NU = 6


@dataclass
class VehicleState:
    """Vehicle state in the tank frame (world origin on the tank axis, z up from floor)."""

    pos_w: np.ndarray  # (3,)
    quat_wb: np.ndarray  # (4,) w, x, y, z — body-to-world
    lin_vel_b: np.ndarray  # (3,) body frame
    ang_vel_b: np.ndarray  # (3,) body frame
    stamp: float = 0.0

    def to_x13(self) -> np.ndarray:
        """Flatten to the MPC state vector."""
        return np.concatenate([
            np.asarray(self.pos_w, float).reshape(3),
            np.asarray(self.quat_wb, float).reshape(4),
            np.asarray(self.lin_vel_b, float).reshape(3),
            np.asarray(self.ang_vel_b, float).reshape(3),
        ])

    @classmethod
    def from_x13(cls, x: np.ndarray, stamp: float = 0.0) -> VehicleState:
        x = np.asarray(x, float).reshape(X13)
        return cls(pos_w=x[0:3], quat_wb=x[3:7], lin_vel_b=x[7:10], ang_vel_b=x[10:13], stamp=stamp)


@dataclass
class ScanReference:
    """One control step's reference, previewed over the MPC horizon.

    The four arrays are (N+1,) — one entry per MPC stage — in the exact keys
    ``WallScanMPC.param_matrix`` reads. A frozen setpoint (the preview-off ablation) is the
    same object with constant arrays, so controllers cannot tell the difference and the
    ablation stays a runner-side choice.
    """

    z_ref: np.ndarray  # (N+1,)
    s_ref: np.ndarray  # (N+1,)
    v_tan_des: np.ndarray  # (N+1,)
    v_z_des: np.ndarray  # (N+1,)
    theta_anchor: float = 0.0  # absolute wall angle the s-error is measured from
    s_anchor: float = 0.0  # arc length at the anchor
    phase: int = 0  # scan_state_machine phase, for weight-policy features

    def as_dict(self) -> dict[str, np.ndarray]:
        """Keys/layout consumed by ``WallScanMPC.param_matrix``."""
        return {"z_ref": self.z_ref, "s_ref": self.s_ref,
                "v_tan_des": self.v_tan_des, "v_z_des": self.v_z_des}

    @classmethod
    def frozen(cls, n_stages: int, *, z_ref: float, s_ref: float, v_tan_des: float = 0.0,
               v_z_des: float = 0.0, theta_anchor: float = 0.0, s_anchor: float = 0.0,
               phase: int = 0) -> ScanReference:
        """Constant-setpoint reference (no preview) over ``n_stages`` MPC stages."""
        ones = np.ones(n_stages + 1)
        return cls(z_ref=z_ref * ones, s_ref=s_ref * ones, v_tan_des=v_tan_des * ones,
                   v_z_des=v_z_des * ones, theta_anchor=theta_anchor, s_anchor=s_anchor,
                   phase=phase)


@dataclass
class ControlOutput:
    """One control step's result, identical in sim and on hardware."""

    u_cmd: np.ndarray  # (6,) normalized thruster command in [-1, 1]
    solve_ms: float = 0.0  # controller compute time (acados solve / policy forward)
    status: int = 0  # 0 = ok; nonzero mirrors the acados status
    u_newton: np.ndarray | None = None  # (6,) per-thruster force before normalization
    aux: dict = field(default_factory=dict)  # method-specific diagnostics (weights, ...)
