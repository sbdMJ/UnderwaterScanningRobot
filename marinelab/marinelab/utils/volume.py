# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Deprecated: moved to marinelab.core.volume. Re-export for compatibility."""

from marinelab.core.volume import *  # noqa: F401,F403
from marinelab.core.volume import (  # noqa: F401
    VolumeInfo,
    compute_articulation_body_volumes,
    compute_collision_volume,
    compute_prim_volume,
)
