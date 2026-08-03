# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Wallscan DORAEMON glue: param defs, scheduler builder, per-env xi application.

Pure glue — the engine lives in marinelab.algorithms.doraemon. xi column order is
PARAM_DEFS order everywhere (spec §3: 13 dynamics params; sensor DR is NOT here).
"""
from __future__ import annotations

import torch

from marinelab.algorithms.doraemon import DoraemonCfg, DoraemonScheduler

# (name, randomization-cfg field, default lo, default hi) — bounds are read from the
# live randomization cfg by build_param_specs; the lo/hi here are documentation defaults.
PARAM_DEFS = [
    ("added_mass", "added_mass_scale", 0.8, 1.2),
    ("linear_damping", "linear_damping_scale", 0.8, 1.2),
    ("quadratic_damping", "quadratic_damping_scale", 0.8, 1.2),
    ("volume", "volume_scale", 0.95, 1.05),
    ("cob_offset_x", "cob_offset_x", -0.03, 0.03),
    ("cob_offset_y", "cob_offset_y", -0.03, 0.03),
    ("cob_offset_z", "cob_offset_z", -0.03, 0.03),
    ("cog_offset_x", "cog_offset_x", -0.03, 0.03),
    ("cog_offset_y", "cog_offset_y", -0.03, 0.03),
    ("cog_offset_z", "cog_offset_z", -0.03, 0.03),
    ("inertia", "inertia_scale", 0.9, 1.1),
    ("thrust_coefficient", "thrust_coefficient_scale", 0.9, 1.1),
    ("time_constant", "time_constant_scale", 0.9, 1.1),
]

# Scales are multiplicative -> nominal 1.0; offsets additive -> nominal 0.0 (spec §3).
NOMINAL_OVERRIDES = {
    "added_mass": 1.0, "linear_damping": 1.0, "quadratic_damping": 1.0, "volume": 1.0,
    "inertia": 1.0, "thrust_coefficient": 1.0, "time_constant": 1.0,
    "cob_offset_x": 0.0, "cob_offset_y": 0.0, "cob_offset_z": 0.0,
    "cog_offset_x": 0.0, "cog_offset_y": 0.0, "cog_offset_z": 0.0,
}

_COL = {name: i for i, (name, _f, _lo, _hi) in enumerate(PARAM_DEFS)}


def build_scheduler(doraemon_cfg: DoraemonCfg, randomization_cfg, device) -> DoraemonScheduler:
    """Scheduler over the 13 wallscan dynamics params; bounds from the live randomization cfg."""
    return DoraemonScheduler(
        doraemon_cfg,
        torch.device(device) if isinstance(device, str) else device,
        dr_cfg=randomization_cfg,
        param_defs=PARAM_DEFS,
        nominal_overrides=NOMINAL_OVERRIDES,
    )


def apply_xi(hydro, thruster, env_ids: torch.Tensor, xi: torch.Tensor) -> None:
    """Apply sampled per-env dynamics values xi[n, 13] to the physics models.

    Scales go through the tensor path of scale_parameters/randomize_parameters
    (core additive extension); CoB/CoG offsets go through set_parameters as
    base + offset (per-env value API that always existed).
    """
    c = _COL
    hydro.scale_parameters(
        env_ids,
        added_mass=xi[:, c["added_mass"]],
        linear_damping=xi[:, c["linear_damping"]],
        quadratic_damping=xi[:, c["quadratic_damping"]],
        volume=xi[:, c["volume"]],
        rigid_body_inertia=xi[:, c["inertia"]],
    )
    base = hydro.base_parameters
    cob = base.center_of_buoyancy[env_ids] + torch.stack(
        [xi[:, c["cob_offset_x"]], xi[:, c["cob_offset_y"]], xi[:, c["cob_offset_z"]]], dim=-1
    )
    cog = base.center_of_gravity[env_ids] + torch.stack(
        [xi[:, c["cog_offset_x"]], xi[:, c["cog_offset_y"]], xi[:, c["cog_offset_z"]]], dim=-1
    )
    hydro.set_parameters(env_ids, center_of_buoyancy=cob, center_of_gravity=cog)
    thruster.randomize_parameters(
        env_ids,
        thrust_coeff_scale=xi[:, c["thrust_coefficient"]],
        time_constant_scale=xi[:, c["time_constant"]],
    )
