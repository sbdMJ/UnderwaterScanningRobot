# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DORAEMON: Domain Randomization with Entropy Maximization (ICLR 2024).

Adaptive Beta distribution scheduling for DR, replacing linear curriculum.
Reference: Tiboni et al., "Domain Randomization via Entropy Maximization", ICLR 2024.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
from scipy.optimize import minimize

from isaaclab.utils import configclass

logger = logging.getLogger(__name__)


def _escape_inference_mode(fn):
    """Re-enable autograd for the SLSQP optimizers when called from inside
    torch.inference_mode() — rsl_rl collects rollouts (and hence runs the env's
    _reset_idx -> scheduler.step()) under inference mode, where backward() raises
    'element 0 of tensors does not require grad'. The except in the optimizers
    swallowed it, so every DORAEMON update silently no-oped for a full 3000-iter
    run (07-25 s_train). Tensor args are cloned so buffer tensors recorded under
    inference mode don't poison the autograd graph."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with torch.inference_mode(False), torch.enable_grad():
            args = tuple(a.detach().clone() if isinstance(a, torch.Tensor) else a for a in args)
            return fn(self, *args, **kwargs)

    return wrapper


# Configuration
# ---------------------------------------------------------------------------


@configclass
class DoraemonCfg:
    """DORAEMON scheduler configuration.

    Reference: Tiboni et al., "Domain Randomization via Entropy Maximization", ICLR 2024.
    """

    enable: bool = True
    performance_lb: float = 80.0  # J_LB: episode return threshold for binary success
    alpha: float = 0.5  # Desired IS-estimated success rate for constraint (Ĝ >= alpha)
    kl_ub: float = 0.5  # Trust region KL upper bound per step (ref default=1.0)
    init_concentration: float = 30.0  # Initial Beta(a,b) concentration (a+b)
    step_interval: int = 250  # RL iterations between DORAEMON updates (ref: train_until_converged)
    buffer_size: int = 2000  # Maximum episode buffer capacity
    min_episodes: int = 200  # Minimum episodes before first update
    min_ess_ratio: float = 0.01  # Minimum ESS/buffer_size to accept update (relaxed for kl_ub=2.0)
    hard_performance_constraint: bool = True  # Use inverted problem when infeasible
    param_overrides: dict[str, tuple[float, float]] = {}  # Per-param bound overrides {name: (lo, hi)}
    replay_curriculum_path: str = ""  # if set, env loads a recorded curriculum for frozen replay


# Parameter Specification
# ---------------------------------------------------------------------------


class ParamSpec(NamedTuple):
    """Single DR parameter specification."""

    name: str
    min_bound: float
    max_bound: float
    nominal: float


def build_param_specs(dr_cfg, param_defs, nominal_overrides=None) -> list[ParamSpec]:
    """Build param specs from a DR config, auto-syncing physics bounds.

    Physics param bounds are read directly from the DR config tuple fields. The caller
    supplies ``param_defs`` (the robot's parameter list) and optional ``nominal_overrides``
    so this engine is robot-agnostic.

    Args:
        dr_cfg: object exposing each ``field_name`` as a ``(lo, hi)`` tuple attribute.
        param_defs: list of ``(param_name, field_name, default_lo, default_hi)`` tuples.
        nominal_overrides: optional ``{param_name: nominal}``; defaults to midpoint of [lo, hi].
    """
    nominal_overrides = nominal_overrides or {}
    specs: list[ParamSpec] = []
    for param_name, field_name, _, _ in param_defs:
        lo, hi = getattr(dr_cfg, field_name)
        nominal = nominal_overrides.get(param_name, (lo + hi) / 2.0)
        specs.append(ParamSpec(param_name, lo, hi, nominal))
    return specs


_MIN_BETA_PARAM = 1.0
_MAX_BETA_PARAM = 500.0
_IS_LOG_CLAMP = 5.0  # IS log-weight clamp: exp(5) ~ 148. Prevents weight explosion in ring buffer.


def _compute_kl(flat_new: np.ndarray, flat_prev: np.ndarray) -> float:
    """Compute KL(new || prev) for independent Beta distributions.

    Uses reverse KL matching Algorithm 1: KL(phi || phi_i) < epsilon.
    Reverse KL penalizes expansion more heavily, providing conservative
    trust region suitable for slow-adapting TRPO policy.
    """
    a_b_new = torch.from_numpy(flat_new.copy()).reshape(-1, 2).double()
    a_b_prev = torch.from_numpy(flat_prev.copy()).reshape(-1, 2).double()
    new = torch.distributions.Beta(
        a_b_new[:, 0].clamp(min=_MIN_BETA_PARAM),
        a_b_new[:, 1].clamp(min=_MIN_BETA_PARAM),
    )
    prev = torch.distributions.Beta(
        a_b_prev[:, 0].clamp(min=_MIN_BETA_PARAM),
        a_b_prev[:, 1].clamp(min=_MIN_BETA_PARAM),
    )
    return torch.distributions.kl_divergence(new, prev).sum().item()


# Beta Distribution
# ---------------------------------------------------------------------------


class BetaDistribution:
    """Independent Beta distributions over DR parameter space, mapped to physical bounds."""

    def __init__(self, params: list[ParamSpec], device: torch.device, concentration: float = 200.0) -> None:
        self.params = params
        self.ndims = len(params)
        self.device = device

        self._mins = torch.tensor([p.min_bound for p in params], dtype=torch.float64)
        self._maxs = torch.tensor([p.max_bound for p in params], dtype=torch.float64)
        self._ranges = self._maxs - self._mins

        self._a = torch.zeros(self.ndims, dtype=torch.float64)
        self._b = torch.zeros(self.ndims, dtype=torch.float64)

        for i, p in enumerate(params):
            mu = (p.nominal - p.min_bound) / (p.max_bound - p.min_bound)
            mu = max(0.01, min(0.99, mu))
            a_raw = mu * concentration
            b_raw = (1.0 - mu) * concentration
            if a_raw < _MIN_BETA_PARAM or b_raw < _MIN_BETA_PARAM:
                # Preserve mean: derive the other param from the clamped one
                if mu <= 0.5:
                    self._a[i] = _MIN_BETA_PARAM
                    self._b[i] = _MIN_BETA_PARAM * (1.0 - mu) / mu
                else:
                    self._b[i] = _MIN_BETA_PARAM
                    self._a[i] = _MIN_BETA_PARAM * mu / (1.0 - mu)
            else:
                self._a[i] = a_raw
                self._b[i] = b_raw

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample n parameter vectors and their log probabilities."""
        a_gpu = self._a.float().to(self.device)
        b_gpu = self._b.float().to(self.device)
        dist = torch.distributions.Beta(a_gpu, b_gpu)

        xi_unit = dist.sample((n,))
        log_probs = dist.log_prob(xi_unit).sum(dim=-1)

        mins_gpu = self._mins.float().to(self.device)
        ranges_gpu = self._ranges.float().to(self.device)
        xi_physical = mins_gpu + xi_unit * ranges_gpu
        return xi_physical, log_probs

    def log_prob(self, xi_physical: torch.Tensor) -> torch.Tensor:
        """Compute log probability of physical-scale samples."""
        mins_gpu = self._mins.float().to(self.device)
        ranges_gpu = self._ranges.float().to(self.device)
        xi_unit = (xi_physical - mins_gpu) / ranges_gpu
        xi_unit = xi_unit.clamp(1e-6, 1.0 - 1e-6)

        a_gpu = self._a.float().to(self.device)
        b_gpu = self._b.float().to(self.device)
        dist = torch.distributions.Beta(a_gpu, b_gpu)
        return dist.log_prob(xi_unit).sum(dim=-1)

    def entropy(self) -> float:
        """Total entropy: sum of per-dim Beta entropies + log(range) scaling."""
        dist = torch.distributions.Beta(self._a, self._b)
        return (dist.entropy() + self._ranges.log()).sum().item()

    def kl_divergence(self, other: BetaDistribution) -> float:
        """KL(self || other): sum of per-dim KL divergences."""
        p = torch.distributions.Beta(self._a, self._b)
        q = torch.distributions.Beta(other._a, other._b)
        return torch.distributions.kl_divergence(p, q).sum().item()

    def get_flat_params(self) -> np.ndarray:
        """Return [a0, b0, a1, b1, ...] as float64 numpy for scipy."""
        flat = torch.stack([self._a, self._b], dim=-1).flatten()
        return flat.numpy()

    def set_flat_params(self, flat: np.ndarray) -> None:
        """Update from scipy result: [a0, b0, a1, b1, ...]."""
        t = torch.from_numpy(flat.copy()).reshape(self.ndims, 2).double()
        self._a = t[:, 0].clamp(min=_MIN_BETA_PARAM, max=_MAX_BETA_PARAM)
        self._b = t[:, 1].clamp(min=_MIN_BETA_PARAM, max=_MAX_BETA_PARAM)

    def clone(self) -> BetaDistribution:
        """Deep copy."""
        new = BetaDistribution.__new__(BetaDistribution)
        new.params = self.params
        new.ndims = self.ndims
        new.device = self.device
        new._mins = self._mins.clone()
        new._maxs = self._maxs.clone()
        new._ranges = self._ranges.clone()
        new._a = self._a.clone()
        new._b = self._b.clone()
        return new

    def get_stats(self) -> dict[str, float]:
        """Per-dimension mean and std in physical space for logging."""
        stats = {}
        for i, p in enumerate(self.params):
            a, b = self._a[i].item(), self._b[i].item()
            mean_unit = a / (a + b)
            var_unit = (a * b) / ((a + b) ** 2 * (a + b + 1))
            std_unit = var_unit**0.5
            mean_phys = p.min_bound + mean_unit * (p.max_bound - p.min_bound)
            std_phys = std_unit * (p.max_bound - p.min_bound)
            stats[f"mean/{p.name}"] = mean_phys
            stats[f"std/{p.name}"] = std_phys
        return stats


# Episode Buffer
# ---------------------------------------------------------------------------


@dataclass
class EpisodeBuffer:
    """Ring buffer for completed episode statistics."""

    capacity: int
    ndims: int
    device: torch.device

    xi: torch.Tensor = field(init=False)
    returns: torch.Tensor = field(init=False)
    success: torch.Tensor = field(init=False)
    log_probs: torch.Tensor = field(init=False)
    _count: int = field(init=False, default=0)
    _write_idx: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.xi = torch.zeros(self.capacity, self.ndims, device=self.device)
        self.returns = torch.zeros(self.capacity, device=self.device)
        self.success = torch.zeros(self.capacity, device=self.device)
        self.log_probs = torch.zeros(self.capacity, device=self.device)

    def add(
        self,
        xi: torch.Tensor,
        returns: torch.Tensor,
        success: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> None:
        """Batch insert episodes. Wraps around if capacity exceeded."""
        n = xi.shape[0]
        if n == 0:
            return
        if n > self.capacity:
            tail = n - self.capacity
            xi = xi[tail:]
            returns = returns[tail:]
            success = success[tail:]
            log_probs = log_probs[tail:]
            n = self.capacity
        start = self._write_idx % self.capacity
        if start + n <= self.capacity:
            self.xi[start : start + n] = xi
            self.returns[start : start + n] = returns
            self.success[start : start + n] = success
            self.log_probs[start : start + n] = log_probs
        else:
            first = self.capacity - start
            self.xi[start:] = xi[:first]
            self.returns[start:] = returns[:first]
            self.success[start:] = success[:first]
            self.log_probs[start:] = log_probs[:first]
            self.xi[: n - first] = xi[first:]
            self.returns[: n - first] = returns[first:]
            self.success[: n - first] = success[first:]
            self.log_probs[: n - first] = log_probs[first:]
        self._write_idx += n
        self._count = min(self._count + n, self.capacity)

    def get_all(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return valid entries."""
        n = self._count
        return self.xi[:n], self.returns[:n], self.success[:n], self.log_probs[:n]

    def clear(self) -> None:
        """Reset buffer."""
        self._count = 0
        self._write_idx = 0


# DORAEMON Scheduler
# ---------------------------------------------------------------------------


class DoraemonScheduler:
    """DORAEMON DR distribution scheduler.

    Aligned with Tiboni et al., ICLR 2024 reference implementation.
    Single constrained optimization: max H(phi) s.t. Ghat >= alpha, KL <= eps.
    When infeasible (hard_performance_constraint): inverted problem first.
    Binary success: episode_return >= performance_lb.
    Unnormalized IS with stored per-episode log probs.
    """

    def __init__(
        self,
        cfg: DoraemonCfg,
        device: torch.device,
        dr_cfg=None,
        param_defs=None,
        nominal_overrides=None,
    ) -> None:
        self.cfg = cfg
        self.device = device

        if dr_cfg is not None and param_defs is not None:
            specs = build_param_specs(dr_cfg, param_defs, nominal_overrides)
        elif param_defs is not None:
            specs = [ParamSpec(n, lo, hi, (nominal_overrides or {}).get(n, (lo + hi) / 2.0))
                     for n, _f, lo, hi in param_defs]
        else:
            raise ValueError(
                "DoraemonScheduler requires param_defs (and usually dr_cfg). The engine is "
                "robot-agnostic; pass the caller's parameter definitions."
            )

        if cfg.param_overrides:
            name_to_idx = {s.name: i for i, s in enumerate(specs)}
            for name, (lo, hi) in cfg.param_overrides.items():
                if name not in name_to_idx:
                    logger.warning("[DORAEMON] param_overrides: unknown parameter '%s', skipping.", name)
                    continue
                idx = name_to_idx[name]
                old = specs[idx]
                specs[idx] = ParamSpec(old.name, lo, hi, (lo + hi) / 2.0)
                logger.info(
                    "[DORAEMON] Override %s bounds: (%.2f, %.2f) -> (%.2f, %.2f)",
                    name,
                    old.min_bound,
                    old.max_bound,
                    lo,
                    hi,
                )

        ndims = len(specs)
        self.dist = BetaDistribution(specs, device, cfg.init_concentration)
        self.buffer = EpisodeBuffer(cfg.buffer_size, ndims, device)

        self._step_count = 0
        self._total_episodes = 0
        self._trajectory: list[dict] = []

        logger.info(
            "[DORAEMON] Initialized: alpha=%.2f, kl_ub=%.4f, performance_lb=%.1f, "
            "%d parameters, concentration=%.0f, hard_constraint=%s",
            cfg.alpha, cfg.kl_ub, cfg.performance_lb, ndims,
            cfg.init_concentration, cfg.hard_performance_constraint,
        )

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample DR parameters from current Beta distribution."""
        return self.dist.sample(n)

    def record_episodes(
        self,
        xi: torch.Tensor,
        returns: torch.Tensor,
        success: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> None:
        """Record completed episodes into the buffer."""
        self.buffer.add(xi, returns, success, log_probs)
        self._total_episodes += xi.shape[0]

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self, iteration: int | None = None) -> dict[str, float]:
        """Run one DORAEMON distribution optimization step.

        Reference structure (Tiboni et al., ICLR 2024):
        1. Estimate success rate under current distribution via unnormalized IS
        2. If feasible (Ghat >= alpha): main optimization
        3. If infeasible + hard_constraint: inverted problem -> main optimization
        """
        xi, _returns, success, log_probs = self.buffer.get_all()
        n = xi.shape[0]

        metrics: dict[str, float] = {}
        metrics["buffer_size"] = float(n)
        metrics["total_episodes"] = float(self._total_episodes)

        if n < self.cfg.min_episodes:
            metrics["skipped"] = 1.0
            metrics["entropy"] = self.dist.entropy()
            metrics["success_rate"] = success.mean().item() if n > 0 else 0.0
            self._step_count += 1
            return metrics

        # Always report metrics; only optimize every step_interval iterations.
        # Between updates the policy trains on the current DR distribution,
        # matching the reference's "train for N steps then update" structure.
        metrics["success_rate"] = success.mean().item()
        metrics["entropy_before"] = self.dist.entropy()

        # Per-parameter distribution stats (always logged)
        for k, v in self.dist.get_stats().items():
            metrics[k] = v

        if self._step_count % self.cfg.step_interval != 0:
            metrics["entropy_after"] = self.dist.entropy()
            metrics["kl_step"] = 0.0
            self._step_count += 1
            return metrics

        prev_dist = self.dist.clone()

        # Estimate success rate under current distribution (unnormalized IS)
        # Use prev_dist as IS denominator: ring buffer data is mostly from recent
        # distributions close to prev_dist, matching reference's fresh-data approach.
        current_success_rate = self._estimate_success_rate(xi, success, prev_dist)
        metrics["success_rate"] = current_success_rate

        if self.cfg.hard_performance_constraint and current_success_rate < self.cfg.alpha:
            # Infeasible: inverted problem to find feasible starting point
            feasible_flat, inv_kl, inv_ok = self._find_feasible_start(prev_dist, xi, success)

            if inv_ok:
                self.dist.set_flat_params(feasible_flat)
                new_sr = self._estimate_success_rate(xi, success, prev_dist)

                if new_sr >= self.cfg.alpha:
                    # Feasible point found -> run main optimization from here
                    # Use original prev_dist as IS/KL reference (matches reference impl)
                    self._optimize_entropy(prev_dist, xi, success)
                    metrics["mode"] = 1.0  # inverted + optimize
                else:
                    # Max-success point still infeasible -> keep it, skip main opt
                    metrics["mode"] = -2.0  # kept max-success dist
            else:
                # Inverted problem failed -> revert
                self.dist = prev_dist
                metrics["mode"] = -3.0  # optimization error
        else:
            # Feasible (or soft constraint): main optimization directly
            self._optimize_entropy(prev_dist, xi, success)
            metrics["mode"] = 0.0  # normal

        metrics["entropy_after"] = self.dist.entropy()
        metrics["kl_step"] = self.dist.kl_divergence(prev_dist)

        # ESS validation: revert if IS estimator quality is too low
        ess, ess_ratio = self._compute_ess(xi, prev_dist, n)
        metrics["ess"] = ess
        metrics["ess_ratio"] = ess_ratio
        if ess < self.cfg.min_ess_ratio * n:
            self.dist = prev_dist
            metrics["reverted"] = 1.0
            metrics["entropy_after"] = self.dist.entropy()
            metrics["kl_step"] = 0.0
            logger.warning("[DORAEMON] Reverted: ESS=%.0f (%.1f%% of %d)", ess, 100 * ess_ratio, n)

        # Record the committed distribution for curriculum replay.
        rec_iter = iteration if iteration is not None else self._step_count
        self._trajectory.append(
            {
                "iter": int(rec_iter),
                "a": self.dist._a.tolist(),
                "b": self.dist._b.tolist(),
            }
        )

        self._step_count += 1
        return metrics

    # ------------------------------------------------------------------
    # IS estimation (unnormalized, using stored per-episode log probs)
    # ------------------------------------------------------------------

    def _estimate_success_rate(
        self,
        xi: torch.Tensor,
        success: torch.Tensor,
        ref_dist: BetaDistribution,
    ) -> float:
        """Unnormalized IS estimate: (1/K) sum [p_current(xi_k) / p_ref(xi_k)] * sigma_k.

        Uses ref_dist (typically prev_dist / current dist) as IS denominator instead
        of stored per-episode log_probs.  Ring buffer data is mostly from recent
        distributions close to ref_dist, so this is a good approximation and keeps
        IS weights near 1.0 -- matching the reference implementation which always
        uses current_distr as denominator with fresh data.
        """
        current_lp = self.dist.log_prob(xi)
        ref_lp = ref_dist.log_prob(xi)
        log_ratio = (current_lp - ref_lp).clamp(-_IS_LOG_CLAMP, _IS_LOG_CLAMP)
        IS_weights = torch.exp(log_ratio)
        return torch.mean(IS_weights * success).item()

    def _compute_ess(
        self,
        xi: torch.Tensor,
        prev_dist: BetaDistribution,
        n: int,
    ) -> tuple[float, float]:
        """Effective Sample Size between current and previous distribution."""
        current_lp = self.dist.log_prob(xi)
        ref_lp = prev_dist.log_prob(xi)
        log_ratio = current_lp - ref_lp
        weights = torch.exp(log_ratio - log_ratio.max())
        weights = weights / weights.sum()
        ess = (1.0 / (weights**2).sum()).item()
        return ess, ess / max(n, 1)

    # ------------------------------------------------------------------
    # Shared helpers for scipy optimization
    # ------------------------------------------------------------------

    def _precompute_is_data(
        self,
        prev_flat: np.ndarray,
        xi: torch.Tensor,
        success: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute CPU tensors for scipy IS optimization.

        Returns (xi_unit, success_t, ref_lp) -- all on CPU, float64.
        """
        mins = self.dist._mins
        ranges = self.dist._ranges
        xi_unit = ((xi.detach().cpu().float() - mins.float()) / ranges.float()).clamp(1e-6, 1 - 1e-6).double()
        success_t = success.detach().cpu().double()
        prev_a_b = torch.from_numpy(prev_flat.copy()).reshape(self.dist.ndims, 2).double()
        prev_dist_ref = torch.distributions.Beta(
            prev_a_b[:, 0].clamp(min=_MIN_BETA_PARAM), prev_a_b[:, 1].clamp(min=_MIN_BETA_PARAM)
        )
        ref_lp = prev_dist_ref.log_prob(xi_unit).sum(dim=-1).detach()
        return xi_unit, success_t, ref_lp

    @staticmethod
    def _make_kl_constraint(
        prev_flat: np.ndarray,
        kl_budget: float,
    ) -> dict:
        """Build SLSQP KL inequality constraint: kl_budget - KL(new || prev) >= 0."""
        ndims = prev_flat.shape[0] // 2

        def kl_ineq(log_flat: np.ndarray) -> float:
            flat = np.exp(log_flat).clip(min=_MIN_BETA_PARAM, max=_MAX_BETA_PARAM)
            return kl_budget - _compute_kl(flat, prev_flat)

        def kl_ineq_jac(log_flat: np.ndarray) -> np.ndarray:
            log_t = torch.from_numpy(log_flat.copy()).double().requires_grad_(True)
            a_b = torch.exp(log_t).reshape(ndims, 2)
            new_d = torch.distributions.Beta(a_b[:, 0].clamp(min=_MIN_BETA_PARAM), a_b[:, 1].clamp(min=_MIN_BETA_PARAM))
            prev_a_b = torch.from_numpy(prev_flat.copy()).reshape(ndims, 2).double()
            prev_d = torch.distributions.Beta(
                prev_a_b[:, 0].clamp(min=_MIN_BETA_PARAM), prev_a_b[:, 1].clamp(min=_MIN_BETA_PARAM)
            )
            kl = torch.distributions.kl_divergence(new_d, prev_d).sum()
            kl.backward()
            assert log_t.grad is not None
            return -log_t.grad.numpy().copy()

        return {"type": "ineq", "fun": kl_ineq, "jac": kl_ineq_jac}

    # ------------------------------------------------------------------
    # Main optimization: max H(phi) s.t. Ghat >= alpha, KL <= eps
    # ------------------------------------------------------------------

    @_escape_inference_mode
    def _optimize_entropy(
        self,
        prev_dist: BetaDistribution,
        xi: torch.Tensor,
        success: torch.Tensor,
    ) -> None:
        """Maximize entropy subject to success >= alpha and KL trust region.

        Uses SLSQP with analytical gradients in log-space parameterization.
        IS denominator: prev_dist (not stored log_probs) for ring buffer stability.
        """
        prev_flat = prev_dist.get_flat_params()
        xi_unit, success_t, ref_lp = self._precompute_is_data(prev_flat, xi, success)
        ranges = self.dist._ranges

        x0_flat = self.dist.get_flat_params()
        log_x0 = np.log(x0_flat.clip(min=_MIN_BETA_PARAM))

        # Objective: minimize negative entropy (log-space)
        def objective_fn(log_flat: np.ndarray) -> tuple[float, np.ndarray]:
            log_t = torch.from_numpy(log_flat.copy()).double().requires_grad_(True)
            a_b = torch.exp(log_t).reshape(self.dist.ndims, 2)
            a = a_b[:, 0].clamp(min=_MIN_BETA_PARAM)
            b = a_b[:, 1].clamp(min=_MIN_BETA_PARAM)
            dist = torch.distributions.Beta(a, b)
            neg_entropy = -(dist.entropy() + ranges.double().log()).sum()
            neg_entropy.backward()
            assert log_t.grad is not None
            return neg_entropy.item(), log_t.grad.numpy().copy()

        # Performance constraint: Ghat >= alpha (SLSQP 'ineq' = g(x) >= 0)
        def perf_ineq(log_flat: np.ndarray) -> float:
            log_t = torch.from_numpy(log_flat.copy()).double()
            a_b = torch.exp(log_t).reshape(self.dist.ndims, 2)
            new_dist = torch.distributions.Beta(
                a_b[:, 0].clamp(min=_MIN_BETA_PARAM), a_b[:, 1].clamp(min=_MIN_BETA_PARAM)
            )
            new_lp = new_dist.log_prob(xi_unit).sum(dim=-1)
            IS_w = torch.exp((new_lp - ref_lp).clamp(-_IS_LOG_CLAMP, _IS_LOG_CLAMP))
            return torch.mean(IS_w * success_t).item() - self.cfg.alpha

        def perf_ineq_jac(log_flat: np.ndarray) -> np.ndarray:
            log_t = torch.from_numpy(log_flat.copy()).double().requires_grad_(True)
            a_b = torch.exp(log_t).reshape(self.dist.ndims, 2)
            new_dist = torch.distributions.Beta(
                a_b[:, 0].clamp(min=_MIN_BETA_PARAM), a_b[:, 1].clamp(min=_MIN_BETA_PARAM)
            )
            new_lp = new_dist.log_prob(xi_unit).sum(dim=-1)
            IS_w = torch.exp((new_lp - ref_lp).clamp(-_IS_LOG_CLAMP, _IS_LOG_CLAMP))
            ghat = torch.mean(IS_w * success_t)
            ghat.backward()
            assert log_t.grad is not None
            return log_t.grad.numpy().copy()

        kl_constraint = self._make_kl_constraint(prev_flat, self.cfg.kl_ub)
        constraints = [
            {"type": "ineq", "fun": perf_ineq, "jac": perf_ineq_jac},
            kl_constraint,
        ]

        try:
            result = minimize(
                objective_fn,
                log_x0,
                method="SLSQP",
                jac=True,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-8},
            )
            flat_result = np.exp(result.x).clip(min=_MIN_BETA_PARAM, max=_MAX_BETA_PARAM)
            init_obj = objective_fn(log_x0)[0]
            kl_val = self.cfg.kl_ub - kl_constraint["fun"](result.x)
            # Accept a non-converged SLSQP result only if ALL constraints hold (perf floor + KL) AND entropy
            # improved -- mirrors reference doraemon.py:920 "all constraints satisfied AND improved, else keep old".
            # Without the perf re-check, the OR-branch could accept a sub-floor (success < alpha) distribution when
            # the solver stalls; harmless while success >> alpha, but a real leak once DR is widened toward the floor.
            perf_ok = perf_ineq(result.x) >= 0.0
            if result.success or (perf_ok and result.fun < init_obj and kl_val <= self.cfg.kl_ub):
                self.dist.set_flat_params(flat_result)
                logger.info(
                    "[DORAEMON] Entropy opt: neg_H %.4f -> %.4f, KL=%.4f, success=%s, perf_ok=%s",
                    init_obj,
                    result.fun,
                    kl_val,
                    result.success,
                    perf_ok,
                )
            else:
                logger.warning(
                    "[DORAEMON] Entropy opt rejected: %s (neg_H %.4f -> %.4f, KL=%.4f)",
                    result.message,
                    init_obj,
                    result.fun,
                    kl_val,
                )
        except Exception as e:
            logger.warning("[DORAEMON] Entropy optimization failed: %s", e)

    # ------------------------------------------------------------------
    # Inverted problem: max Ghat(phi) s.t. KL <= eps
    # ------------------------------------------------------------------

    @_escape_inference_mode
    def _find_feasible_start(
        self,
        prev_dist: BetaDistribution,
        xi: torch.Tensor,
        success: torch.Tensor,
    ) -> tuple[np.ndarray | None, float, bool]:
        """Find a feasible starting distribution by maximizing success rate within trust region.

        Returns (flat_params, kl_step, success_flag).
        """
        if success.sum().item() < 1.0:
            logger.debug("[DORAEMON] Inverted problem skipped: no successful episodes.")
            return None, 0.0, False

        prev_flat = prev_dist.get_flat_params()
        xi_unit, success_t, ref_lp = self._precompute_is_data(prev_flat, xi, success)

        log_x0 = np.log(prev_flat.clip(min=_MIN_BETA_PARAM))

        # Objective: minimize -Ghat (maximize success rate)
        def neg_success_fn(log_flat: np.ndarray) -> tuple[float, np.ndarray]:
            log_t = torch.from_numpy(log_flat.copy()).double().requires_grad_(True)
            a_b = torch.exp(log_t).reshape(self.dist.ndims, 2)
            new_dist = torch.distributions.Beta(
                a_b[:, 0].clamp(min=_MIN_BETA_PARAM), a_b[:, 1].clamp(min=_MIN_BETA_PARAM)
            )
            new_lp = new_dist.log_prob(xi_unit).sum(dim=-1)
            IS_w = torch.exp((new_lp - ref_lp).clamp(-_IS_LOG_CLAMP, _IS_LOG_CLAMP))
            neg_sr = -torch.mean(IS_w * success_t)
            neg_sr.backward()
            assert log_t.grad is not None
            return neg_sr.item(), log_t.grad.numpy().copy()

        # Tighter KL margin (1e-5) to leave room for subsequent main optimization
        kl_constraint = self._make_kl_constraint(prev_flat, self.cfg.kl_ub - 1e-5)
        constraints = [kl_constraint]

        try:
            result = minimize(
                neg_success_fn,
                log_x0,
                method="SLSQP",
                jac=True,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-8},
            )
            flat_result = np.exp(result.x).clip(min=_MIN_BETA_PARAM, max=_MAX_BETA_PARAM)
            kl_val = (self.cfg.kl_ub - 1e-5) - kl_constraint["fun"](result.x)

            if result.success:
                logger.info(
                    "[DORAEMON] Inverted problem converged: sr=%.4f, KL=%.4f",
                    -result.fun,
                    kl_val,
                )
                return flat_result, kl_val, True
            else:
                init_obj = neg_success_fn(log_x0)[0]
                if result.fun < init_obj and kl_val <= self.cfg.kl_ub:
                    logger.info(
                        "[DORAEMON] Inverted: not converged but improved (sr %.4f -> %.4f, KL=%.4f). Accepting.",
                        -init_obj,
                        -result.fun,
                        kl_val,
                    )
                    return flat_result, kl_val, True
                logger.warning(
                    "[DORAEMON] Inverted problem failed: %s (sr %.4f -> %.4f, KL=%.4f)",
                    result.message,
                    -init_obj,
                    -result.fun,
                    kl_val,
                )
                return None, 0.0, False
        except Exception as e:
            logger.warning("[DORAEMON] Inverted problem exception: %s", e)
            return None, 0.0, False

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def export_trajectory(self) -> list[dict]:
        """Recorded (iter, a, b) entries, one per DORAEMON update. Replay uses this."""
        return self._trajectory

    def export_recording(self) -> dict:
        """Full recording: param metadata + trajectory, for CurriculumReplayer."""
        return {
            "param_names": [p.name for p in self.dist.params],
            "param_bounds": [[p.min_bound, p.max_bound] for p in self.dist.params],
            "trajectory": self._trajectory,
        }

    def state_dict(self) -> dict:
        """Serialize scheduler state for checkpoint persistence."""
        return {
            "dist_a": self.dist._a.clone(),
            "dist_b": self.dist._b.clone(),
            "step_count": self._step_count,
            "total_episodes": self._total_episodes,
            "buffer_xi": self.buffer.xi[: self.buffer._count].clone(),
            "buffer_returns": self.buffer.returns[: self.buffer._count].clone(),
            "buffer_success": self.buffer.success[: self.buffer._count].clone(),
            "buffer_log_probs": self.buffer.log_probs[: self.buffer._count].clone(),
            "buffer_write_idx": self.buffer._write_idx,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from a checkpoint."""
        ndims = self.dist.ndims
        if state["dist_a"].shape[0] != ndims:
            raise ValueError(
                f"DORAEMON checkpoint has {state['dist_a'].shape[0]} dims but current config expects {ndims}."
            )
        # dist._a/_b must stay CPU float64 (the invariant _mins/_maxs/_ranges hold);
        # a checkpoint loaded with map_location on GPU would otherwise leave them on
        # CUDA, causing a device mismatch against _ranges in entropy()/get_flat_params().
        self.dist._a = state["dist_a"].double().cpu()
        self.dist._b = state["dist_b"].double().cpu()

        self._step_count = state["step_count"]
        self._total_episodes = state["total_episodes"]

        # Restore buffer
        n = state["buffer_xi"].shape[0]
        self.buffer.xi[:n] = state["buffer_xi"].to(self.device)
        self.buffer.returns[:n] = state["buffer_returns"].to(self.device)
        self.buffer.success[:n] = state["buffer_success"].to(self.device)
        self.buffer.log_probs[:n] = state["buffer_log_probs"].to(self.device)
        self.buffer._count = n
        self.buffer._write_idx = state["buffer_write_idx"]


# Curriculum Replay
# ---------------------------------------------------------------------------


class CurriculumReplayer:
    """Replays a recorded DORAEMON curriculum. Duck-types DoraemonScheduler.

    Exposes the same sample()/step()/record_episodes()/state_dict() surface the env
    and runner consume, so swapping it in needs no change to their control flow. The
    distribution at iteration ``t`` is the last recorded update with ``iter <= t``
    (hold-last step function). No buffer, no IS, no optimization.
    """

    def __init__(
        self,
        recording: dict,
        param_specs: list[ParamSpec],
        device: torch.device,
        cfg: DoraemonCfg | None = None,
    ) -> None:
        self._validate(recording, param_specs)
        # The env reads self.cfg.performance_lb (to compute the binary success that
        # record_episodes then discards). Expose a cfg so duck typing stays complete;
        # default to a stock DoraemonCfg when the recording run's cfg is not supplied.
        self.cfg = cfg if cfg is not None else DoraemonCfg()
        self.dist = BetaDistribution(param_specs, device, concentration=2.0)  # overwritten below
        self._schedule = sorted(recording["trajectory"], key=lambda e: int(e["iter"]))
        if not self._schedule:
            raise ValueError("CurriculumReplayer: recording has an empty trajectory.")
        self._last_iter = int(self._schedule[-1]["iter"])
        self._warned_past_end = False
        self._apply(0)

    @staticmethod
    def _validate(recording: dict, param_specs: list[ParamSpec]) -> None:
        names = recording.get("param_names")
        bounds = recording.get("param_bounds")
        if names is None or bounds is None:
            raise ValueError("CurriculumReplayer: recording missing param_names/param_bounds.")
        if len(names) != len(param_specs):
            raise ValueError(
                f"CurriculumReplayer: recording has {len(names)} dims but config expects "
                f"{len(param_specs)}."
            )
        for spec, name, (lo, hi) in zip(param_specs, names, bounds):
            if spec.name != name:
                raise ValueError(
                    f"CurriculumReplayer: param name mismatch at dim: '{name}' vs '{spec.name}'."
                )
            if abs(spec.min_bound - lo) > 1e-9 or abs(spec.max_bound - hi) > 1e-9:
                raise ValueError(
                    f"CurriculumReplayer: bounds mismatch for '{name}': "
                    f"recording ({lo}, {hi}) vs config ({spec.min_bound}, {spec.max_bound})."
                )

    def _apply(self, iteration: int) -> None:
        """Set self.dist to the last recorded update with iter <= iteration."""
        entry = self._schedule[0]
        for e in self._schedule:
            if int(e["iter"]) <= iteration:
                entry = e
            else:
                break
        self.dist._a = torch.tensor(entry["a"], dtype=torch.float64)
        self.dist._b = torch.tensor(entry["b"], dtype=torch.float64)

    # --- DoraemonScheduler-compatible surface ---

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dist.sample(n)

    def step(self, iteration: int | None = None) -> dict[str, float]:
        it = 0 if iteration is None else int(iteration)
        if it > self._last_iter and not self._warned_past_end:
            logger.warning(
                "[REPLAY] iteration %d past recorded end %d; holding final distribution.",
                it, self._last_iter,
            )
            self._warned_past_end = True
        self._apply(it)
        return {"replay/cur_iter": float(it), "replay/entropy": self.dist.entropy()}

    def record_episodes(self, *args, **kwargs) -> None:
        """No-op: replay ignores episode statistics."""
        return

    def state_dict(self) -> dict:
        """Current distribution only (checkpoint compatibility)."""
        return {"dist_a": self.dist._a.clone(), "dist_b": self.dist._b.clone()}

    def load_state_dict(self, state: dict) -> None:
        """Replay distribution is schedule-driven; ignore checkpointed dist."""
        return

    def export_trajectory(self) -> list[dict]:
        """A replay run does not re-record."""
        return []
