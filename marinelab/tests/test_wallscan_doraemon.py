# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the wallscan DORAEMON glue (Isaac-Sim-free via tests/conftest.py shim)."""
import dataclasses
import sys

import torch


def _configclass_with_factory(cls):
    # Same fix as tests/test_doraemon.py: the conftest's bare-dataclass configclass
    # rejects DoraemonCfg's mutable dict default; wrap mutables in default_factory.
    for name, value in list(vars(cls).items()):
        if name in getattr(cls, "__annotations__", {}) and isinstance(value, (dict, list, set)):
            default = value
            setattr(cls, name, dataclasses.field(default_factory=lambda d=default: type(d)(d)))
    return dataclasses.dataclass(cls)


sys.modules["isaaclab.utils"].configclass = _configclass_with_factory

from marinelab.algorithms.doraemon import DoraemonCfg  # noqa: E402
from marinelab.tasks.pkrc_wallscan.doraemon_dr import (  # noqa: E402
    NOMINAL_OVERRIDES,
    PARAM_DEFS,
    apply_xi,
    build_scheduler,
)


class _StubRandCfg:
    """Field names/values mirror WallScanTrainCfg.randomization (wallscan_env_cfg.py)."""

    added_mass_scale = (0.8, 1.2)
    linear_damping_scale = (0.8, 1.2)
    quadratic_damping_scale = (0.8, 1.2)
    volume_scale = (0.95, 1.05)
    cob_offset_x = (-0.03, 0.03)
    cob_offset_y = (-0.03, 0.03)
    cob_offset_z = (-0.03, 0.03)
    cog_offset_x = (-0.03, 0.03)
    cog_offset_y = (-0.03, 0.03)
    cog_offset_z = (-0.03, 0.03)
    inertia_scale = (0.9, 1.1)
    thrust_coefficient_scale = (0.9, 1.1)
    time_constant_scale = (0.9, 1.1)


def test_param_defs_13_dims_bounds_and_nominals():
    sched = build_scheduler(DoraemonCfg(), _StubRandCfg(), torch.device("cpu"))
    assert sched.dist.ndims == 13
    specs = {s.name: s for s in sched.dist.params}
    assert (specs["added_mass"].min_bound, specs["added_mass"].max_bound) == (0.8, 1.2)
    assert (specs["cob_offset_z"].min_bound, specs["cob_offset_z"].max_bound) == (-0.03, 0.03)
    # scale nominals explicitly 1.0, offset nominals 0.0 (spec §3)
    for name in ("added_mass", "volume", "inertia", "thrust_coefficient", "time_constant"):
        assert specs[name].nominal == 1.0
    for name in ("cob_offset_x", "cog_offset_z"):
        assert specs[name].nominal == 0.0
    xi, logp = sched.sample(8)
    assert xi.shape == (8, 13) and logp.shape == (8,)


class _FakeHydro:
    def __init__(self, n=4):
        class _Base:
            center_of_buoyancy = torch.tensor([[0.0, 0.0, 0.15]]).repeat(n, 1)
            center_of_gravity = torch.zeros(n, 3)
        self.base_parameters = _Base()
        self.scale_calls: dict = {}
        self.set_calls: dict = {}

    def scale_parameters(self, env_ids, **kw):
        self.scale_calls = kw

    def set_parameters(self, env_ids, **kw):
        self.set_calls = kw


class _FakeThruster:
    def __init__(self):
        self.calls: dict = {}

    def randomize_parameters(self, env_ids, **kw):
        self.calls = kw


def test_apply_xi_routes_columns_to_models():
    # Column i of xi must land on the model arg named by PARAM_DEFS[i].
    n = 2
    env_ids = torch.tensor([0, 1])
    xi = torch.arange(n * 13, dtype=torch.float32).reshape(n, 13) * 0.01
    col = {name: i for i, (name, _f, _lo, _hi) in enumerate(PARAM_DEFS)}
    hydro, thr = _FakeHydro(n), _FakeThruster()
    apply_xi(hydro, thr, env_ids, xi)
    assert torch.allclose(hydro.scale_calls["added_mass"], xi[:, col["added_mass"]])
    assert torch.allclose(hydro.scale_calls["volume"], xi[:, col["volume"]])
    assert torch.allclose(hydro.scale_calls["rigid_body_inertia"], xi[:, col["inertia"]])
    assert torch.allclose(thr.calls["thrust_coeff_scale"], xi[:, col["thrust_coefficient"]])
    assert torch.allclose(thr.calls["time_constant_scale"], xi[:, col["time_constant"]])
    # CoB/CoG = base + offset columns
    exp_cob = hydro.base_parameters.center_of_buoyancy[env_ids] + torch.stack(
        [xi[:, col["cob_offset_x"]], xi[:, col["cob_offset_y"]], xi[:, col["cob_offset_z"]]], dim=-1
    )
    assert torch.allclose(hydro.set_calls["center_of_buoyancy"], exp_cob)
