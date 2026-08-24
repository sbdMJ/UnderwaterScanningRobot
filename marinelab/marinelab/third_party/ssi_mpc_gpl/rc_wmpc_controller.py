# SPDX-License-Identifier: GPL-3.0-only — see LICENSE in this directory.
"""RC-WMPC: Residual-Conditioned Weights-varying MPC (proposed method, 2026-08-24).

Unifies the two adaptation axes this framework already has, per the RC-WMPC proposal:

- **model channel** (inherited from :class:`SSIMPCController`): the RFF online learner's
  residual is injected as the OCP's per-stage ``d_world`` disturbance, with the
  field-validated guards (latency-aligned regression pairs, clamp, injection low-pass);
- **objective channel** (the Diff-WMPC weight policy): a learned ``WeightPolicy`` emits
  the cost weights every step — here CONDITIONED on the learner's adaptation state via
  :func:`marinelab.algorithms.diff_wmpc.rc_context` (``d_world`` = what the model already
  compensates, ``pred_err`` = what remains uncompensated, ``|alpha|`` = maturity).

Step ordering matters and is inherited, not reimplemented: ``SSIMPCController.step``
updates the learner and sets ``self._d_world`` / ``self._d_filt`` BEFORE delegating to
``FixedWeightNMPC.step``, which is what calls :meth:`_current_weights` — so the ctx this
class feeds the policy is exactly the adaptation state of the solve it parameterizes,
including the in-flight-command roll-forward (``x_eval``) semantics.

A checkpoint trained WITHOUT ctx (``ctx_dim == 0`` — any pre-RC Diff-WMPC checkpoint)
degrades this controller to the naive-stacking composition: SSI injection + unconditioned
weight policy. That is deliberately supported — it IS ablation A1 of the proposal.

This file lives in the GPL-isolated package because it subclasses the SSI-MPC port;
everything under this directory is GPL-3.0 (see LICENSE), unlike the BSD rest of the repo.
"""
from __future__ import annotations

import numpy as np
import torch

from marinelab.control.types import ControlOutput, ScanReference, VehicleState

from .ssi_controller import SSIMPCController


class RCWMPCController(SSIMPCController):
    name = "rc"

    def __init__(self, *, mpc_cfg, ckpt_path: str | None = None, policy=None,
                 ctx_d_scale: float = 10.0, ctx_e_scale: float = 0.5,
                 ctx_a_scale: float = 5.0, **ssi_kwargs):
        super().__init__(mpc_cfg=mpc_cfg, **ssi_kwargs)
        self.name = "rc"  # params_json load renames to "bo"; the method identity wins
        self._mref_cfg = mpc_cfg
        self._ctx_scales = (float(ctx_d_scale), float(ctx_e_scale), float(ctx_a_scale))
        if policy is None:
            if ckpt_path is None:
                raise ValueError("pass ckpt_path= (trained checkpoint) or policy= (built WeightPolicy)")
            from marinelab.algorithms.diff_wmpc import WeightPolicy
            from marinelab.tasks.pkrc_wallscan.mpc_reference import NE

            # The checkpoint is the architecture authority (hidden/history/preview/ctx spec
            # travel inside it) — same rule as the Diff-WMPC adapter.
            state = torch.load(ckpt_path, map_location="cpu")
            policy = WeightPolicy.from_state_dict(state, NE, self._mpc.nu,
                                                  werr_init=self._weights[:NE],
                                                  wu_init=self._weights[NE:])
        policy.eval()
        self._policy = policy
        if int(policy.ctx_dim) == 0:
            # Naive stacking (ablation A1): SSI injection + a ctx-blind weight policy.
            # Named distinctly so a results table can never pass one off as the other.
            self.name = "rc_naive"

    def reset(self, state: VehicleState) -> None:
        super().reset(state)
        self._policy.reset_history()

    def _current_weights(self, state: VehicleState, ref: ScanReference) -> np.ndarray:
        from marinelab.algorithms.diff_wmpc import RC_CTX_DIM, policy_features, rc_context
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
            ctx = None
            if int(self._policy.ctx_dim):
                if int(self._policy.ctx_dim) != RC_CTX_DIM:
                    raise ValueError(f"checkpoint ctx_dim {int(self._policy.ctx_dim)} != "
                                     f"rc_context's {RC_CTX_DIM}")
                d_sc, e_sc, a_sc = self._ctx_scales
                # _d_world is the post-clamp, post-LPF injected value set earlier THIS step
                # by SSIMPCController.step — the ctx must describe what the OCP actually
                # sees, not the raw learner output.
                d = np.zeros(3) if self._d_world is None else self._d_world
                ctx = rc_context(d, self._learner.last_pred_err,
                                 float(np.linalg.norm(self._learner.alpha)),
                                 d_scale=d_sc, e_scale=e_sc, a_scale=a_sc)
            feat = policy_features(e_now, ref.phase, z_ref=ref.z_ref, s_ref=ref.s_ref,
                                   preview_nodes=self._policy.preview_nodes.tolist(),
                                   ctx=ctx)
            return self._policy(feat).numpy()

    def step(self, state: VehicleState, ref: ScanReference,
             obs: np.ndarray | None = None) -> ControlOutput:
        out = super().step(state, ref, obs)
        # w(t) telemetry for the E3 soften->re-stiffen analysis rides on aux["werr"]
        # (set by FixedWeightNMPC.step); add the ctx the policy saw this step.
        if int(self._policy.ctx_dim):
            out.aux["rc_pred_err"] = self._learner.last_pred_err
        return out
