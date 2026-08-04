# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Hydrodynamic parameter bundle and base-parameter cache.

HydroParams is a named bundle of per-env parameter tensors used by both the
read API (get_parameters, for privileged observations) and the write API
(set_parameters, for domain randomization). default_rigid_inertia centralizes
the added_mass[3:6]*0.5 fallback that was previously duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HydroParams:
    """Named bundle of per-env hydrodynamic parameter tensors.

    All fields are optional: a None field means "not provided / leave unchanged"
    on the write path, and is simply absent on the read path if not requested.
    """

    added_mass: torch.Tensor | None = None  # (N, 6)
    linear_damping: torch.Tensor | None = None  # (N, 6)
    quadratic_damping: torch.Tensor | None = None  # (N, 6)
    volume: torch.Tensor | None = None  # (N,)
    water_density: torch.Tensor | None = None  # (N,)
    center_of_buoyancy: torch.Tensor | None = None  # (N, 3)
    center_of_gravity: torch.Tensor | None = None  # (N, 3)
    rigid_body_inertia: torch.Tensor | None = None  # (N, 3)
    body_mass: torch.Tensor | None = None  # (N,)


def slender_axis(added_mass, linear_damping, quadratic_damping, inertia,
                 ref_speed: float = 0.2) -> dict[str, int]:
    """Which body axis does each parameter group claim is the vehicle's LONG one?

    An elongated body has a distinguished axis, and three independent parameter groups all
    have to agree on which one it is:

    * **Added mass** — motion ALONG the long axis displaces the least water, so
      ``argmin(added_mass[:3])`` is the long axis.
    * **Translational drag** — axial motion presents the smallest cross-section, so the long
      axis minimizes the total drag ``D_l*v + D_q*v^2`` at a reference speed. Total, not the
      linear coefficient alone: fitting ``L + Q`` to one tow curve trades the two off, so
      individual coefficients can disagree across axes while the physical drag agrees. (For
      PKRC at 0.2 m/s, surge 26.8 N vs sway 25.4 N — 5% apart despite a 4.7x difference in
      the quadratic term alone.)
    * **Rigid-body inertia** — rotation ABOUT the long axis sweeps the least mass, so
      ``argmin(inertia)`` is the long axis.

    Returns the three verdicts as axis indices (0 = x/surge, 1 = y/sway, 2 = z/heave). They
    must be equal; a disagreement means at least one group is misassigned, which is checkable
    without any external data. Shipped ``PKRCHydrodynamicsCfg`` disagreed: added mass said x
    while the USD-derived PhysX inertia said z.
    """
    am = torch.as_tensor(added_mass, dtype=torch.float64)[:3]
    dl = torch.as_tensor(linear_damping, dtype=torch.float64)[:3]
    dq = torch.as_tensor(quadratic_damping, dtype=torch.float64)[:3]
    inr = torch.as_tensor(inertia, dtype=torch.float64)[:3]
    drag = dl * ref_speed + dq * ref_speed * ref_speed
    return {
        "added_mass": int(am.argmin()),
        "drag": int(drag.argmin()),
        "inertia": int(inr.argmin()),
    }


def slender_axis_is_consistent(*args, **kwargs) -> bool:
    """True when :func:`slender_axis`'s three verdicts agree on one long axis."""
    verdicts = slender_axis(*args, **kwargs)
    return len(set(verdicts.values())) == 1


def default_rigid_inertia(cfg) -> list[float]:
    """Resolve rigid-body inertia, falling back to added_mass[3:6] * 0.5.

    Single source of the heuristic that was duplicated across the model init,
    stability validation, and randomization code.

    Args:
        cfg: A config with `added_mass` (len-6) and optional `rigid_body_inertia`.

    Returns:
        Three-element inertia diagonal [Ixx, Iyy, Izz].
    """
    if getattr(cfg, "rigid_body_inertia", None) is not None:
        return list(cfg.rigid_body_inertia)
    return [x * 0.5 for x in cfg.added_mass[3:6]]
