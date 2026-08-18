# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Diff-WMPC adapter — E1 method ⑤: learned WeightPolicy drives the fixed-NMPC solver.

Inference-only composition of two existing pieces, both unmodified:
:class:`marinelab.algorithms.diff_wmpc.WeightPolicy` (checkpoint from
``train_diff_wmpc_wallscan.py``) and the ``WallScanMPC`` solver via
:class:`~marinelab.control.fixed_nmpc.FixedWeightNMPC`. The feature vector fed to the
policy replicates ``run_wallscan_mpc.py --policy_ckpt``: the NE-dim wallscan error at the
current stage plus (sin, cos) of the scan phase.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .fixed_nmpc import FixedWeightNMPC
from .types import ControlOutput, ScanReference, VehicleState


class DiffWMPCController(FixedWeightNMPC):
    name = "diff"

    def __init__(self, *, mpc_cfg, ckpt_path: str | None = None, policy=None, **nmpc_kwargs):
        super().__init__(mpc_cfg=mpc_cfg, **nmpc_kwargs)
        self.name = "diff"  # FixedWeightNMPC may have renamed itself "bo" via params_json
        self._mref_cfg = mpc_cfg
        if policy is None:
            if ckpt_path is None:
                raise ValueError("pass ckpt_path= (trained checkpoint) or policy= (built WeightPolicy)")
            from marinelab.algorithms.diff_wmpc import WeightPolicy
            from marinelab.tasks.pkrc_wallscan.mpc_reference import NE

            # The checkpoint is the architecture authority (hidden/history/preview spec
            # travel inside it) — building with ctor defaults and strict-loading was how
            # the pre-buffer legacy checkpoint became silently unloadable.
            state = torch.load(ckpt_path, map_location="cpu")
            policy = WeightPolicy.from_state_dict(state, NE, self._mpc.nu,
                                                  werr_init=self._weights[:NE],
                                                  wu_init=self._weights[NE:])
        policy.eval()
        self._policy = policy

    def reset(self, state: VehicleState) -> None:
        super().reset(state)
        self._policy.reset_history()

    def _current_weights(self, state: VehicleState, ref: ScanReference) -> np.ndarray:
        from marinelab.tasks.pkrc_wallscan import mpc_reference as mref

        with torch.no_grad():
            x = torch.as_tensor(state.to_x13(), dtype=torch.float32).unsqueeze(0)
            e_now = mref.wallscan_errors(
                x,
                z_ref=torch.tensor([float(ref.z_ref[0])]),
                s_ref=torch.tensor([float(ref.s_ref[0])]),
                v_tan_des=torch.tensor([float(ref.v_tan_des[0])]),
                v_z_des=torch.tensor([float(ref.v_z_des[0])]),
                theta_anchor=torch.tensor([float(ref.theta_anchor)]),
                s_anchor=torch.tensor([float(ref.s_anchor)]),
                cfg=self._mref_cfg,
            )[0]
            from marinelab.algorithms.diff_wmpc import policy_features
            feat = policy_features(e_now, ref.phase, z_ref=ref.z_ref, s_ref=ref.s_ref,
                                   preview_nodes=self._policy.preview_nodes.tolist())
            return self._policy(feat).numpy()

    def step(self, state: VehicleState, ref: ScanReference,
             obs: np.ndarray | None = None) -> ControlOutput:
        return super().step(state, ref, obs)
