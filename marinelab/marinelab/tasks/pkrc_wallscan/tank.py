# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Cylindrical water tank spawn config for the wallscan task.

The sonar wall-distance / wall-collision termination are computed ANALYTICALLY
in geometry.py (wall_distance, radial_clearance), not via physics raycast or
contact. This tank is VISUAL ONLY (collision disabled): a solid convex
CylinderCfg with collision enabled would depenetrate the robot spawned inside
it on GPU PhysX (unfiltered dynamic-vs-kinematic), causing spurious
ejection/contacts. It exists only to give play.py a visible reference shell.

ponytail: solid CylinderCfg (not a true hollow shell/annulus) rendered
semi-transparent so the robot stays visible inside it. Upgrade to a real
annulus mesh only if a task later needs physical wall collision.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils

TANK_RADIUS = 6.0
TANK_HEIGHT = 10.0

TANK_CFG = sim_utils.CylinderCfg(
    radius=TANK_RADIUS,
    height=TANK_HEIGHT,
    axis="Z",
    # opacity 0.35: at 0.15 the wall was invisible against a dark viewport (only its shadow
    # showed as a black disc on the ground — looked like the tank was "below ground").
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.7), opacity=0.35),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),  # static, no dynamics
)
"""Spawn config for the cylindrical tank shell (visual reference only, radius=6.0m, height=10.0m)."""


def spawn_tank(prim_path: str = "/World/Tank") -> None:
    """Spawn the tank shell at ``prim_path``, centered at the origin."""
    TANK_CFG.func(prim_path, TANK_CFG, translation=(0.0, 0.0, TANK_HEIGHT / 2.0))
