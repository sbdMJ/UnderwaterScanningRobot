# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Diff-WMPC: learn NMPC cost weights by backpropagating through the solver's sensitivity.

Port of the learning core from Underwater-Actor-Critic-Model-Predictive-Control
(``marinegym/learning/diff_wmpc_cylinder.py``), which implements *"Differentiable
Weights-Varying Nonlinear MPC via Gradient-Based Policy Learning"*. No RL, no critic, no
reward: a small network maps the current situation to the MPC's diagonal cost weights, the
solver reports ``d z* / d p_global``, and a differentiable task loss is pushed back through
that Jacobian into the network.

One learning step::

    w        = policy(feat)                                  # differentiable in policy params
    out      = mpc.solve(x0, P, w.detach())                   # acados + forward sensitivity
    x_node, u0 = torch leaves from out
    L        = task_loss(x_node, u0)                          # differentiable in x_node, u0
    dL/dx, dL/du = autograd.grad(L, (x_node, u0))
    gW       = clip(sens_x^T dL/dx + sens_u^T dL/du)          # -> dL/d p_global
    autograd.grad(w, params, grad_outputs=gW)                 # accumulate
    every batch_size steps: Adam step

Pure torch — the acados half lives in ``tasks.pkrc_wallscan.mpc_controller``, and this module
only consumes the sensitivity matrices it returns. That keeps the learner unit-testable
natively (feed it hand-made Jacobians) without a compiled solver.

## Two invariants the port must preserve

1. **Skip steps where the solve failed or the control saturated.** Measured on this host
   (``isaaclab/logs/_probe_acados.py``): ``eval_solution_sensitivity`` matches finite
   differences to 6-7 digits at an interior optimum and is *identically zero* at a bound. A
   saturated step therefore contributes a silently wrong (zero) gradient, not noise.
2. **The control term of the loss must use the NORMALIZED command** ``u/max_thrust``. The
   reference implementation flags the alternative as a "learn to do nothing" pathology: with
   forces in newtons the effort term dwarfs the tracking terms and the optimum is to stop
   moving.

## Weight bounds are wider than the reference

The orbit port capped ``w_err`` at 500. Wallscan needs more: a hand sweep on 2026-07-30
found the roll weight has to reach ~2000 before the solver will spend heave differential on
cancelling the sway leg's parasitic moment (sway tilt 1.11 deg at w=20 -> 0.14 deg at
w=2000, measured closed-loop). A ceiling of 500 would put the single most valuable weight
outside the search space, so the default upper bound here is 5000.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

__all__ = ["WeightPolicy", "DiffWMPCLearner", "WallScanLossCfg", "wallscan_loss"]


def _inv_sigmoid(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = y.clamp(eps, 1.0 - eps)
    return torch.log(y) - torch.log1p(-y)


def _as_bound(value, n: int) -> torch.Tensor:
    """Scalar or per-entry bound -> (n,) tensor, so a single weight can be floored alone."""
    t = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    return t.expand(n).clone() if t.numel() == 1 else t


def _map_to_range(raw: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor, *,
                  log_scale: bool) -> torch.Tensor:
    s = torch.sigmoid(raw)
    if log_scale:
        return torch.exp(torch.log(lb) + (torch.log(ub) - torch.log(lb)) * s)
    return lb + (ub - lb) * s


class WeightPolicy(nn.Module):
    """Situation -> bounded diagonal MPC cost weights ``[w_err(ne), w_u(nu)]``.

    Bounded outputs are not cosmetic: unbounded weights drive the MPC solution onto the
    thrust limits, where the solution sensitivity is zero and the learning signal vanishes
    (and, on this problem, where the QP itself starts failing).

    The input is the current feature vector concatenated with a short history, so the network
    can infer unmodelled disturbances (current, actuator-gain error, buoyancy trim) from how
    the recent state evolved — the same trick as the reference implementation.

    ``head`` is initialized to the ``*_init`` weights (orthogonal gain 0.01 on the last layer
    plus a bias solving for the requested values), so training starts from a hand-tuned
    controller that already works rather than from a random one.
    """

    def __init__(self, feat_dim: int, ne: int, nu: int, *, history_len: int = 4, hidden: int = 128,
                 werr_init: np.ndarray | None = None, wu_init: np.ndarray | None = None,
                 werr_lb: float = 0.1, werr_ub: float = 5000.0,
                 wu_lb: float = 5e-3, wu_ub: float = 5.0, log_scale: bool = True):
        super().__init__()
        self.feat_dim, self.ne, self.nu = int(feat_dim), int(ne), int(nu)
        self.history_len = int(history_len)
        # Bounds may be per-entry: the 2026-07-31 collapse was w_radial falling to 2-3, and a
        # floor on just that entry is cheap insurance that costs nothing when it is not binding.
        #
        # Registered as BUFFERS so they travel in state_dict. The bounds are part of the
        # output mapping, not decoration: loading a checkpoint into a policy built with
        # different bounds yields different cost weights from the identical network, which is
        # a silent behaviour change with no error anywhere. Round-tripping them removes that
        # failure mode entirely.
        self.register_buffer("werr_lb", _as_bound(werr_lb, self.ne))
        self.register_buffer("werr_ub", _as_bound(werr_ub, self.ne))
        self.register_buffer("wu_lb", _as_bound(wu_lb, self.nu))
        self.register_buffer("wu_ub", _as_bound(wu_ub, self.nu))
        self.log_scale = bool(log_scale)

        self.fc1 = nn.Linear(self.feat_dim * (self.history_len + 1), hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, self.ne + self.nu)
        self._history: list[torch.Tensor] = []
        self._init_head(werr_init, wu_init)

    def _init_head(self, werr_init, wu_init) -> None:
        for m in (self.fc1, self.fc2):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)

        werr = np.full(self.ne, 40.0) if werr_init is None else np.asarray(werr_init, float)
        wu = np.full(self.nu, 0.01) if wu_init is None else np.asarray(wu_init, float)
        we = torch.as_tensor(werr, dtype=torch.float32).clamp(self.werr_lb, self.werr_ub)
        wc = torch.as_tensor(wu, dtype=torch.float32).clamp(self.wu_lb, self.wu_ub)
        if self.log_scale:
            se = (torch.log(we) - torch.log(self.werr_lb)) / (torch.log(self.werr_ub) - torch.log(self.werr_lb))
            su = (torch.log(wc) - torch.log(self.wu_lb)) / (torch.log(self.wu_ub) - torch.log(self.wu_lb))
        else:
            se = (we - self.werr_lb) / (self.werr_ub - self.werr_lb)
            su = (wc - self.wu_lb) / (self.wu_ub - self.wu_lb)
        with torch.no_grad():
            self.head.bias.copy_(torch.cat([_inv_sigmoid(se), _inv_sigmoid(su)]))

    def reset_history(self) -> None:
        self._history = []

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """``feat``: (feat_dim,) detached features for this step. Returns (ne+nu,) weights."""
        feat = feat.reshape(-1).float().detach()
        if not self._history:
            self._history = [feat.clone() for _ in range(self.history_len)]
        x_in = torch.cat([feat] + self._history, dim=0)
        self._history = ([feat.clone()] + self._history)[: self.history_len]

        h = torch.tanh(self.fc1(x_in))
        h = torch.tanh(self.fc2(h))
        raw = self.head(h)
        w_err = _map_to_range(raw[: self.ne], self.werr_lb, self.werr_ub, log_scale=self.log_scale)
        w_u = _map_to_range(raw[self.ne:], self.wu_lb, self.wu_ub, log_scale=self.log_scale)
        return torch.cat([w_err, w_u], dim=0)


@dataclass
class WallScanLossCfg:
    """Task-level loss weights. NOT the MPC cost weights — these define what "good" means.

    Priorities follow the two problems this port exists to solve: heading (a single-beam
    echo sounder reads the true clearance only when the beam is normal to the wall, and the
    error is second-order in the offset so it is easy to miss) and attitude during the sway
    leg. ``l_u`` stays small; see the module docstring on the normalized-command requirement.

    KNOWN FAILURE, measured 2026-07-31 (8000 steps, then scored with
    ``scripts/run_wallscan_mpc.py --policy_ckpt``): with the loss evaluated at a SINGLE
    shooting node 1.0 s ahead, the learner perfected attitude (sway tilt 0.07 deg, better
    than the best hand-tuned 0.14) and **abandoned radial/heading tracking** — it drove
    ``w_radial`` down to 2-3 and the closed-loop run came out at crab 150.7 deg and 7.5 m of
    wall-distance error. The mechanism is visible in the timescales: attitude has ~0.1 s time
    constants so a 1 s lookahead shows a large sensitivity, while radial drift barely moves
    in 1 s, so its weight looks worthless to the gradient. The orbit port warns about exactly
    this ("sens_node ~ 2N/3, a far node so slow radial drift is visible; node 1 is too
    myopic") — 2N/3 is simply not far enough for wallscan's much slower scan timescale.

    Fixes, cheapest first: (1) sum the loss over SEVERAL nodes so fast and slow axes both
    register, (2) lengthen the horizon (N=60 at dt=0.05 puts the node 2 s out, at ~2x the
    solve cost), (3) raise ``l_radial``/``l_heading`` and lift the per-entry weight FLOOR for
    radial/heading so the search cannot collapse them.
    """

    l_radial: float = 2.0
    l_z: float = 1.0
    l_s: float = 1.0
    l_v_rad: float = 0.2
    l_v_tan: float = 0.2
    # 2.0 (2026-07-31), was 0.2. The multi-node learner beat the hand-tuned weights on sway
    # tilt (0.14 -> 0.07 deg) and sway speed (0.092 -> 0.101 m/s, target 0.100) but pushed the
    # heave leg FURTHER over its 0.20 m/s target (0.233 -> 0.244), the one confirmed
    # regression. Arithmetic explains it: at the measured 0.17 m ramp lag the z-position term
    # contributes 1.0 * 0.17^2 = 0.029 while the 0.044 m/s rate error contributes only
    # 0.2 * 0.044^2 = 3.9e-4 -- 75x smaller, so overshooting the rate to close the lag was
    # nearly free. Raising this to l_radial's scale makes the rate target actually compete.
    l_v_z: float = 2.0
    l_heading: float = 2.0
    l_rollpitch: float = 2.0
    l_omega: float = 0.05
    l_u: float = 1e-3
    max_thrust: float = 40.0


def wallscan_loss(errors: torch.Tensor, u0: torch.Tensor, cfg: WallScanLossCfg) -> torch.Tensor:
    """Differentiable task loss from a wallscan error vector and the applied control.

    ``errors`` is the (12,) vector from ``mpc_reference.wallscan_errors`` evaluated at the
    predicted state, in ``ERROR_NAMES`` order; it must be a tensor that requires grad so the
    caller can pull ``dL/dx`` out of it.
    """
    e = errors.reshape(-1)
    return (
        cfg.l_radial * e[0] ** 2
        + cfg.l_z * e[1] ** 2
        + cfg.l_s * e[2] ** 2
        + cfg.l_v_rad * e[3] ** 2
        + cfg.l_v_tan * e[4] ** 2
        + cfg.l_v_z * e[5] ** 2
        + cfg.l_heading * (e[6] ** 2 + e[7] ** 2)
        + cfg.l_rollpitch * (e[8] ** 2 + e[9] ** 2)
        + cfg.l_omega * (e[10] ** 2 + e[11] ** 2)
        + cfg.l_u * ((u0 / float(cfg.max_thrust)) ** 2).mean()
    )


class DiffWMPCLearner:
    """Owns the weight policy + optimizer and applies the sensitivity-chained gradient."""

    def __init__(self, policy: WeightPolicy, *, n_pglobal: int, lr: float = 5e-4,
                 betas: tuple[float, float] = (0.5, 0.99), grad_clip: float = 0.1,
                 batch_size: int = 10, saturation_thresh: float = 0.98,
                 device: str = "cpu"):
        self.policy = policy.to(device)
        self.device = device
        self.n_pglobal = int(n_pglobal)
        self.grad_clip = float(grad_clip)
        self.batch_size = max(1, int(batch_size))
        self.saturation_thresh = float(saturation_thresh)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self._in_batch = 0
        self.last_loss = float("nan")
        self.n_updates = 0
        self.n_skipped_status = 0
        self.n_skipped_sat = 0
        self.n_skipped_nan = 0

    def compute_weights(self, feat: torch.Tensor) -> torch.Tensor:
        return self.policy(feat.to(self.device))

    @property
    def n_skipped(self) -> int:
        return self.n_skipped_status + self.n_skipped_sat + self.n_skipped_nan

    def learn_step(self, weights: torch.Tensor, mpc_out: dict, error_fn,
                   loss_cfg: WallScanLossCfg, node_weights: dict | None = None):
        """Accumulate one Diff-WMPC gradient; step Adam every ``batch_size`` accepted calls.

        The loss is summed over EVERY node the solver returned a state sensitivity for::

            L  = sum_k a_k * loss(errors(x*_k, ref_k), u*_0)
            gW = sum_k a_k * Sx_k^T dL_k/dx_k  +  Su^T dL/du

        Summing is the fix for the single-node failure measured 2026-07-31, where the learner
        perfected attitude and abandoned radial tracking (w_radial fell to 2-3; closed loop
        came out at crab 150 deg). Attitude has ~0.1 s time constants so it has converged by
        the later nodes, which therefore carry almost pure position/heading error, while the
        radial contribution accumulates across all of them.

        Args:
            weights: the DIFFERENTIABLE tensor from :meth:`compute_weights` (not a copy).
            mpc_out: dict from ``WallScanMPC.solve(..., want_sensitivity=True)``; uses
                ``sens_x_nodes``/``x_nodes`` when present and falls back to the single
                ``sens_x``/``x_node`` pair otherwise.
            error_fn: ``(x_tensor, node) -> (ne,)`` errors, closing over the reference THAT
                node tracks. Passing the stage-0 reference for every node would quietly
                reintroduce the myopia this method exists to remove.
            loss_cfg: task loss weights.
            node_weights: optional per-node scale ``a_k`` (default 1.0).

        Returns:
            The summed step loss, or None when the step was skipped.
        """
        su = mpc_out.get("sens_u")
        nodes = mpc_out.get("sens_x_nodes")
        if nodes is None and mpc_out.get("sens_x") is not None:
            nodes = {int(mpc_out.get("sens_node", 0)): mpc_out["sens_x"]}
        if mpc_out.get("status", 1) != 0 or not nodes or su is None:
            self.n_skipped_status += 1
            return None
        if float(np.abs(mpc_out["u0_cmd"]).max()) > self.saturation_thresh:
            # At a thrust bound the sensitivity is exactly zero, so this step would teach
            # "these weights do not matter" -- a wrong lesson, not a noisy one.
            self.n_skipped_sat += 1
            return None

        x_nodes = mpc_out.get("x_nodes") or {int(mpc_out.get("sens_node", 0)): mpc_out["x_node"]}
        u0 = torch.tensor(mpc_out["u0"], dtype=torch.float32, device=self.device,
                          requires_grad=True)
        gW = torch.zeros(self.n_pglobal, device=self.device)
        total = 0.0
        usable = [k for k in sorted(nodes) if k in x_nodes]
        first_node = min(usable) if usable else None
        for k in usable:
            a_k = 1.0 if node_weights is None else float(node_weights.get(k, 1.0))
            if a_k == 0.0:
                continue
            x_k = torch.tensor(x_nodes[k], dtype=torch.float32, device=self.device,
                               requires_grad=True)
            # The control-effort term rides on the first node ONLY. Passing a DETACHED u0 to
            # the others would keep it out of the gradient but still add its value to the loss
            # once per node, so the reported number would disagree with what is optimized;
            # zeroing it counts the effort exactly once in both.
            first = k == first_node
            u_k = u0 if first else torch.zeros_like(u0)
            loss_k = wallscan_loss(error_fn(x_k, k), u_k, loss_cfg)
            if not loss_k.requires_grad:
                # This node's loss is disconnected from the graph (a term that is identically
                # zero here). It contributes value but no gradient; do not ask autograd.
                total += a_k * float(loss_k.detach())
                continue
            targets = (x_k, u0) if first else (x_k,)
            grads = torch.autograd.grad(loss_k, targets, allow_unused=True)
            # grads[0] can legitimately be None when this node's loss does not depend on the
            # state (e.g. an error term that is identically zero there); contribute nothing
            # rather than crashing on a None operand.
            if grads[0] is not None:
                Sx_k = torch.as_tensor(nodes[k], dtype=torch.float32, device=self.device)
                gW = gW + a_k * (Sx_k.t() @ grads[0])
            if first and len(grads) > 1 and grads[1] is not None:
                Su = torch.as_tensor(su, dtype=torch.float32, device=self.device)
                gW = gW + Su.t() @ grads[1]
            total += a_k * float(loss_k.detach())

        # Check BEFORE clipping: clipping first turns an inf/nan sensitivity entry into a
        # plausible-looking +-grad_clip and feeds garbage into Adam.
        if not torch.isfinite(gW).all():
            self.n_skipped_nan += 1
            return None
        gW = torch.clamp(gW, -self.grad_clip, self.grad_clip)

        params = [p for p in self.policy.parameters() if p.requires_grad]
        grads = torch.autograd.grad(weights, params, grad_outputs=gW, allow_unused=True)
        for p, g in zip(params, grads):
            if g is None:
                continue
            p.grad = g.detach().clone() if p.grad is None else p.grad + g.detach()

        self._in_batch += 1
        self.last_loss = total
        if self._in_batch >= self.batch_size:
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self._in_batch = 0
            self.n_updates += 1
        return self.last_loss

    def state_dict(self) -> dict:
        return {"policy": self.policy.state_dict(), "opt": self.opt.state_dict(),
                "n_updates": self.n_updates}

    def load_state_dict(self, state: dict) -> None:
        self.policy.load_state_dict(state["policy"])
        if "opt" in state:
            self.opt.load_state_dict(state["opt"])
        self.n_updates = int(state.get("n_updates", 0))
