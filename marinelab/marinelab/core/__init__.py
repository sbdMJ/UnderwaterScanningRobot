# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""marinelab.core: general-purpose UUV simulation framework.

Stable, reusable physics + parameter + ocean-current API. Researcher code
imports from here and tunes via config + the setter API; private buffer
access is never required.
"""

from .hydrodynamics import HydrodynamicsModel
from .ocean_current import OceanCurrent
from .parameters import HydroParams, default_rigid_inertia
from .thruster import ThrusterCfg, ThrusterModel

__all__ = [
    "HydrodynamicsModel",
    "HydroParams",
    "default_rigid_inertia",
    "OceanCurrent",
    "ThrusterModel",
    "ThrusterCfg",
]
