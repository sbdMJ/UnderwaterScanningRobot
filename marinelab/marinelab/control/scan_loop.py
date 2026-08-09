# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""The controller-owned wallscan closed loop — scan state machine + reference preview +
controller step — as one pure object.

Extracted from ``scripts/experiments/_sim_loop.run_mpc_cell`` (itself a port of
``run_wallscan_mpc.main``) so simulation and the vehicle drive the SAME call sequence:
``_sim_loop`` feeds it env ground truth or the sim EKF, the ``wallscan_controller`` ROS
node feeds it ``/wallscan/state``. Torch/numpy only — no isaaclab, no rclpy — so the §8
closed-loop guarantee ("one full tick with no simulator") is a native test.

Inputs per tick are deliberately scalar: ``(VehicleState, s_hat, theta_hat)``. The state
machine's sway timing runs on the arc-length estimate and the reference anchor runs on the
absolute wall bearing — exactly the two quantities the estimator owns (GT in the sim's
Phase-2a runs, wall-frame EKF elsewhere), so no caller can accidentally re-derive them from
a different source.
"""
from __future__ import annotations

import numpy as np
import torch

from marinelab.tasks.pkrc_wallscan import mpc_reference as mref
from marinelab.tasks.pkrc_wallscan import scan_state_machine as ssm

from .types import ControlOutput, ScanReference, VehicleState


class WallScanControlLoop:
    """One vehicle's scan brain: phase machine state + ramped references + the controller."""

    def __init__(self, ctl, scan_cfg: "ssm.ScanCfg", mpc_cfg: "mref.WallScanMPCCfg",
                 horizon: int = 30, device: str = "cpu"):
        self.ctl = ctl
        self.scan_cfg = scan_cfg
        self.mpc_cfg = mpc_cfg
        self.horizon = int(horizon)
        self.device = device
        self.sm = ssm.ScanState(1, device=device)
        self.cycles = 0
        self._z_ref = self._s_ref = 0.0

    @property
    def phase(self) -> int:
        return int(self.sm.phase[0])

    def reset(self, veh: VehicleState | None = None, z0: float | None = None) -> None:
        """Mirror of ``_sim_loop.reset_internal``: phase 0, refs re-anchored at current z.

        ``z0`` overrides the ramp anchor independently of ``veh`` — the sim runner anchors
        at ground-truth z while historically resetting the controller with a zero state,
        and regression-parity with the recorded e5_ekf cells depends on keeping that.
        """
        if z0 is None:
            z0 = float(veh.pos_w[2]) if veh is not None else 0.0
        self.sm.phase[:] = 0
        self.sm.s_ref[:] = 0.0
        self.sm.sway_dir[:] = 1.0
        self.sm.z_hold[:] = 0.0
        self.sm._hold[:] = 0
        self.sm.z_ramp[:] = z0
        self.sm.s_ramp[:] = 0.0
        self.cycles = 0
        self.ctl.reset(veh if veh is not None else VehicleState.from_x13(np.zeros(13)))

    def step(self, veh: VehicleState, s_hat: float, theta_hat: float) -> ControlOutput:
        """One control tick: advance the phase machine, build the preview, run the controller.

        ``s_hat``/``theta_hat``: the estimator's arc length and absolute wall bearing (a
        consistent pair — ``theta_hat = theta0 + s_hat / R``).
        """
        z_sm = torch.tensor([float(veh.pos_w[2])], device=self.device)
        s_sm = torch.tensor([float(s_hat)], device=self.device)
        z_ref, s_ref, _phase_sc, advanced = ssm.step(
            self.sm, z_sm, s_sm, self.scan_cfg, z_latch=z_sm)
        self.cycles += int(bool((advanced & (self.sm.phase == 0))[0]))
        self._z_ref, self._s_ref = float(z_ref[0]), float(s_ref[0])

        preview = mref.reference_preview(
            self.sm.phase, self.sm.z_ramp, self.sm.s_ramp, self.sm.s_ref, self.sm.z_hold,
            self.mpc_cfg, self.horizon,
        )
        ref = ScanReference(
            z_ref=preview["z_ref"][0].cpu().numpy(), s_ref=preview["s_ref"][0].cpu().numpy(),
            v_tan_des=preview["v_tan_des"][0].cpu().numpy(),
            v_z_des=preview["v_z_des"][0].cpu().numpy(),
            theta_anchor=float(theta_hat), s_anchor=float(s_hat), phase=self.phase,
        )
        return self.ctl.step(veh, ref)

    @property
    def refs(self) -> tuple[float, float]:
        """(z_ref, s_ref) the state machine emitted on the last step — telemetry/scoring."""
        return self._z_ref, self._s_ref
