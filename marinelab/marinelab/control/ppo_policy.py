# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""PPO policy runner — E1 method ③, and the Jetson inference path.

Runs the *exported* policy (``checkpoints/exported/policy.pt`` TorchScript or
``policy.onnx``, both written by ``play.py``), not the rsl_rl runner: the same file and the
same class deploy to the vehicle, so the sim comparison and the hardware run share one
inference implementation. The 31-D observation vector is assembled by the caller (the
experiment runner in sim, the sensor pipeline on hardware) and passed via ``obs``.
"""
from __future__ import annotations

import time

import numpy as np

from .base import WallScanController
from .types import ControlOutput, ScanReference, VehicleState


class PPOPolicyController(WallScanController):
    name = "ppo"

    def __init__(self, policy_path: str, obs_dim: int = 31):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self._backend = "onnx" if policy_path.endswith(".onnx") else "torchscript"
        if self._backend == "onnx":
            import onnxruntime as ort

            self._sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
            self._input_name = self._sess.get_inputs()[0].name
        else:
            import torch

            self._torch = torch
            self._module = torch.jit.load(policy_path, map_location="cpu")
            self._module.eval()

    def reset(self, state: VehicleState) -> None:
        pass  # exported policy is a feedforward MLP; nothing episodic to clear

    def step(self, state: VehicleState, ref: ScanReference,
             obs: np.ndarray | None = None) -> ControlOutput:
        if obs is None:
            raise ValueError("PPOPolicyController needs the 31-D observation vector via obs=")
        x = np.asarray(obs, np.float32).reshape(1, self.obs_dim)
        t0 = time.perf_counter()
        if self._backend == "onnx":
            action = self._sess.run(None, {self._input_name: x})[0][0]
        else:
            with self._torch.no_grad():
                action = self._module(self._torch.from_numpy(x))[0].numpy()
        solve_ms = 1e3 * (time.perf_counter() - t0)
        u_cmd = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
        out = ControlOutput(u_cmd=u_cmd, solve_ms=solve_ms, status=0)
        self.stats.record(out)
        return out
