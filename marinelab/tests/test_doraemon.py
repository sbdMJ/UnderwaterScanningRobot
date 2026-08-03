# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the robot-agnostic DORAEMON engine in marinelab.algorithms.

Imports go through the conftest shim (bare ``marinelab`` package + isaaclab mock)
so the heavy ``marinelab/__init__.py`` (Isaac Sim) never runs. The engine takes
injected ``param_defs``/``nominal_overrides`` -- no robot-specific module globals.
"""

import dataclasses
import sys

import pytest
import torch


def _configclass_with_factory(cls):
    """configclass stand-in that wraps mutable defaults in default_factory.

    The conftest mock uses a bare ``dataclasses.dataclass``, which rejects
    ``DoraemonCfg.param_overrides: dict = {}`` (mutable default). The real
    ``isaaclab.utils.configclass`` handles this; replicate just that here.
    """
    for name, value in list(vars(cls).items()):
        if name in getattr(cls, "__annotations__", {}) and isinstance(value, (dict, list, set)):
            default = value
            setattr(cls, name, dataclasses.field(default_factory=lambda d=default: type(d)(d)))
    return dataclasses.dataclass(cls)


# Override the conftest's minimal configclass before importing the engine.
sys.modules["isaaclab.utils"].configclass = _configclass_with_factory

from marinelab.algorithms.doraemon import (  # noqa: E402
    BetaDistribution,
    DoraemonCfg,
    DoraemonScheduler,
    EpisodeBuffer,
    ParamSpec,
    build_param_specs,
)

DEVICE = torch.device("cpu")

# Synthetic 3-param DR config (mass / damping / current) -- deliberately NOT
# constrained-albc's _PARAM_DEFS: the engine must be robot-agnostic.
PARAM_DEFS = [
    ("mass", "mass_range", 1.0, 3.0),
    ("damping", "damping_range", 10.0, 20.0),
    ("current", "current_range", 0.0, 0.5),
]


class _StubDrCfg:
    """Exposes each field_name as a (lo, hi) tuple, as build_param_specs expects."""

    mass_range = (1.0, 3.0)
    damping_range = (10.0, 20.0)
    current_range = (0.0, 0.5)


# build_param_specs
# ---------------------------------------------------------------------------


def test_build_param_specs_reads_bounds_and_midpoint_nominal():
    specs = build_param_specs(_StubDrCfg(), PARAM_DEFS)

    assert [s.name for s in specs] == ["mass", "damping", "current"]  # ordering preserved
    assert specs[0] == ParamSpec("mass", 1.0, 3.0, 2.0)  # midpoint nominal
    assert specs[1] == ParamSpec("damping", 10.0, 20.0, 15.0)
    assert specs[2] == ParamSpec("current", 0.0, 0.5, 0.25)


def test_build_param_specs_applies_nominal_override():
    specs = build_param_specs(_StubDrCfg(), PARAM_DEFS, nominal_overrides={"damping": 12.0})

    # overridden param uses the override, others stay at midpoint
    assert specs[1].nominal == 12.0
    assert specs[0].nominal == 2.0
    assert specs[2].nominal == 0.25


# BetaDistribution
# ---------------------------------------------------------------------------


def test_beta_distribution_sample_shape_bounds_and_self_kl():
    specs = build_param_specs(_StubDrCfg(), PARAM_DEFS)
    dist = BetaDistribution(specs, DEVICE, concentration=200.0)

    assert dist.ndims == 3

    n = 256
    xi, logp = dist.sample(n)
    assert xi.shape == (n, 3)
    assert logp.shape == (n,)

    eps = 1e-4
    mins = torch.tensor([s.min_bound for s in specs])
    maxs = torch.tensor([s.max_bound for s in specs])
    assert torch.all(xi >= mins - eps)
    assert torch.all(xi <= maxs + eps)

    # KL of a distribution against itself is ~0
    assert abs(dist.kl_divergence(dist)) < 1e-6


# EpisodeBuffer
# ---------------------------------------------------------------------------


def _episode_batch(values, ndims=3):
    """Build a (len(values), ndims) xi batch plus matching 1D stat tensors.

    Each episode's xi row is filled with its scalar tag so we can identify
    which episodes survived the ring wrap.
    """
    k = len(values)
    tags = torch.tensor(values, dtype=torch.float32)
    xi = tags.unsqueeze(1).expand(k, ndims).contiguous()
    return xi, tags, tags, tags  # xi, returns, success, log_probs


def test_episode_buffer_ring_caps_and_keeps_newest():
    buf = EpisodeBuffer(capacity=4, ndims=3, device=DEVICE)

    buf.add(*_episode_batch([0.0, 1.0, 2.0]))  # 3 episodes
    xi, returns, success, log_probs = buf.get_all()
    assert xi.shape[0] == 3

    # add 3 more -> total 6 written, capacity 4: oldest two (0,1) evicted
    buf.add(*_episode_batch([3.0, 4.0, 5.0]))
    xi, returns, success, log_probs = buf.get_all()
    assert xi.shape[0] == 4  # capped at capacity

    # the four retained episodes are the newest: tags {2,3,4,5}
    retained = set(returns.tolist())
    assert retained == {2.0, 3.0, 4.0, 5.0}

    buf.clear()
    xi, _, _, _ = buf.get_all()
    assert xi.shape[0] == 0


# DoraemonScheduler
# ---------------------------------------------------------------------------


def test_scheduler_builds_from_injected_param_defs_and_exposes_override():
    cfg = DoraemonCfg(init_concentration=50.0, buffer_size=128)
    sched = DoraemonScheduler(
        cfg,
        DEVICE,
        dr_cfg=_StubDrCfg(),
        param_defs=PARAM_DEFS,
        nominal_overrides={"current": 0.1},
    )

    assert sched.dist.ndims == 3
    specs_by_name = {s.name: s for s in sched.dist.params}
    assert specs_by_name["current"].nominal == 0.1  # override surfaced in dist specs

    xi, logp = sched.sample(16)
    assert xi.shape == (16, 3)
    assert logp.shape == (16,)


def test_scheduler_without_param_defs_raises():
    cfg = DoraemonCfg()
    with pytest.raises(ValueError, match="param_defs"):
        DoraemonScheduler(cfg, DEVICE)


def test_scheduler_optimizes_under_inference_mode():
    """Regression (07-25 s_train): rsl_rl runs env resets under torch.inference_mode(),
    where the SLSQP backward() raised and the swallowed exception silently no-oped every
    DORAEMON update for a whole run. The optimizers must escape inference mode."""
    cfg = DoraemonCfg(init_concentration=50.0, buffer_size=512, min_episodes=64, step_interval=1)
    sched = DoraemonScheduler(cfg, DEVICE, dr_cfg=_StubDrCfg(), param_defs=PARAM_DEFS)
    h0 = sched.dist.entropy()

    with torch.inference_mode():
        for _ in range(4):
            xi, logp = sched.sample(64)
            returns = torch.ones(64, device=DEVICE)
            success = torch.ones(64, device=DEVICE)  # 100% success -> must widen
            sched.record_episodes(xi, returns, success, logp)
            metrics = sched.step()

    assert "mode" in metrics, f"optimization path never ran: {sorted(metrics)}"
    assert sched.dist.entropy() > h0, (
        f"entropy did not increase under inference_mode: {h0:.4f} -> {sched.dist.entropy():.4f}"
    )
