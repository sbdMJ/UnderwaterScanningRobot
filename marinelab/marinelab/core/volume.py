# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for computing volumes from USD geometry.

This module provides functions to calculate the volume of USD prims,
particularly useful for buoyancy calculations in underwater simulations.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

from pxr import Usd, UsdPhysics

__all__ = [
    "compute_collision_volume",
    "compute_prim_volume",
    "compute_articulation_body_volumes",
    "VolumeInfo",
]


@dataclass
class VolumeInfo:
    """Container for articulation body volume information.

    Attributes:
        body_volumes: Dictionary mapping body name to volume in m^3.
        total_volume: Sum of all body volumes in m^3.
        body_paths: Dictionary mapping body name to USD prim path.
    """

    body_volumes: dict[str, float] = field(default_factory=dict)
    total_volume: float = 0.0
    body_paths: dict[str, str] = field(default_factory=dict)


def compute_collision_volume(prim_path: str, stage: Usd.Stage | None = None) -> float:
    """Compute the volume of collision geometry at the given prim path.

    This function searches for collision geometry under the specified prim
    and computes the total volume. It handles both primitive shapes (sphere,
    box, cylinder, capsule, cone) and mesh geometry.

    Args:
        prim_path: USD prim path to search for collision geometry.
        stage: USD stage. If None, gets the current stage from Usd.Stage.

    Returns:
        Total volume of all collision geometries in cubic meters (m^3).

    Raises:
        ValueError: If the prim path is invalid or no collision geometry found.

    Example:
        >>> volume = compute_collision_volume("/World/Robot/base_link")
        >>> print(f"Base link volume: {volume:.6f} m^3")
    """
    if stage is None:
        from omni.usd import get_context

        stage = get_context().get_stage()

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")

    total_volume = 0.0
    collision_count = 0

    # Search for collision geometry in prim and descendants
    for descendant in Usd.PrimRange(prim):
        # Check if this prim has collision API
        if UsdPhysics.CollisionAPI(descendant):
            volume = compute_prim_volume(descendant)
            if volume > 0:
                total_volume += volume
                collision_count += 1

    # If no collision API found, try to compute volume from geometry directly
    if collision_count == 0:
        for descendant in Usd.PrimRange(prim):
            prim_type = descendant.GetTypeName()
            if prim_type in _VOLUME_CALCULATORS or prim_type == "Mesh":
                volume = compute_prim_volume(descendant)
                if volume > 0:
                    total_volume += volume
                    collision_count += 1

    if collision_count == 0:
        warnings.warn(f"No collision geometry found under {prim_path}. Returning 0 volume.")

    return total_volume


def compute_prim_volume(prim: Usd.Prim) -> float:
    """Compute the volume of a single USD prim.

    Supports the following prim types:
        - Sphere: V = (4/3) * pi * r^3
        - Cube: V = size^3
        - Cylinder: V = pi * r^2 * h
        - Capsule: V = pi * r^2 * h + (4/3) * pi * r^3
        - Cone: V = (1/3) * pi * r^2 * h
        - Mesh: Uses trimesh library, falls back to convex hull for non-watertight meshes

    Args:
        prim: USD prim to compute volume for.

    Returns:
        Volume in cubic meters (m^3). Returns 0 if prim type is unsupported.
    """
    prim_type = prim.GetTypeName()

    if prim_type in _VOLUME_CALCULATORS:
        return _VOLUME_CALCULATORS[prim_type](prim)
    elif prim_type == "Mesh":
        return _compute_mesh_volume(prim)

    return 0.0


def compute_articulation_body_volumes(articulation_prim_path: str, stage: Usd.Stage | None = None) -> VolumeInfo:
    """Compute volumes for each rigid body in an articulation.

    This function traverses the articulation hierarchy and computes
    the collision volume for each rigid body link.

    Args:
        articulation_prim_path: USD path to the articulation root.
        stage: USD stage. If None, gets the current stage.

    Returns:
        VolumeInfo containing per-body volumes and total volume.

    Example:
        >>> info = compute_articulation_body_volumes("/World/Robot")
        >>> print(f"Base volume: {info.body_volumes.get('base', 0):.6f} m^3")
        >>> print(f"Total volume: {info.total_volume:.6f} m^3")
    """
    if stage is None:
        from omni.usd import get_context

        stage = get_context().get_stage()

    root_prim = stage.GetPrimAtPath(articulation_prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"Invalid articulation path: {articulation_prim_path}")

    info = VolumeInfo()

    # Find all rigid body prims
    for prim in Usd.PrimRange(root_prim):
        if UsdPhysics.RigidBodyAPI(prim):
            body_name = prim.GetName()
            body_path = str(prim.GetPath())

            # Compute volume for this body
            volume = compute_collision_volume(body_path, stage)

            info.body_volumes[body_name] = volume
            info.body_paths[body_name] = body_path
            info.total_volume += volume

    return info


# Volume calculation functions for primitive shapes


def _compute_sphere_volume(prim: Usd.Prim) -> float:
    """Compute sphere volume: V = (4/3) * pi * r^3."""
    radius_attr = prim.GetAttribute("radius")
    if not radius_attr or not radius_attr.HasValue():
        return 0.0
    radius = radius_attr.Get()
    return (4.0 / 3.0) * math.pi * radius**3


def _compute_cube_volume(prim: Usd.Prim) -> float:
    """Compute cube volume: V = size^3."""
    size_attr = prim.GetAttribute("size")
    if not size_attr or not size_attr.HasValue():
        return 0.0
    size = size_attr.Get()
    return size**3


def _compute_cylinder_volume(prim: Usd.Prim) -> float:
    """Compute cylinder volume: V = pi * r^2 * h."""
    radius_attr = prim.GetAttribute("radius")
    height_attr = prim.GetAttribute("height")
    if not radius_attr or not radius_attr.HasValue():
        return 0.0
    if not height_attr or not height_attr.HasValue():
        return 0.0
    radius = radius_attr.Get()
    height = height_attr.Get()
    return math.pi * radius**2 * height


def _compute_capsule_volume(prim: Usd.Prim) -> float:
    """Compute capsule volume: V = pi * r^2 * h + (4/3) * pi * r^3."""
    radius_attr = prim.GetAttribute("radius")
    height_attr = prim.GetAttribute("height")
    if not radius_attr or not radius_attr.HasValue():
        return 0.0
    if not height_attr or not height_attr.HasValue():
        return 0.0
    radius = radius_attr.Get()
    height = height_attr.Get()
    # Capsule = cylinder + 2 hemispheres = cylinder + 1 sphere
    cylinder_volume = math.pi * radius**2 * height
    sphere_volume = (4.0 / 3.0) * math.pi * radius**3
    return cylinder_volume + sphere_volume


def _compute_cone_volume(prim: Usd.Prim) -> float:
    """Compute cone volume: V = (1/3) * pi * r^2 * h."""
    radius_attr = prim.GetAttribute("radius")
    height_attr = prim.GetAttribute("height")
    if not radius_attr or not radius_attr.HasValue():
        return 0.0
    if not height_attr or not height_attr.HasValue():
        return 0.0
    radius = radius_attr.Get()
    height = height_attr.Get()
    return (1.0 / 3.0) * math.pi * radius**2 * height


def _compute_mesh_volume(prim: Usd.Prim) -> float:
    """Compute mesh volume using trimesh library.

    For non-watertight meshes, falls back to convex hull approximation.

    Args:
        prim: USD Mesh prim.

    Returns:
        Volume in cubic meters. Returns 0 if computation fails.
    """
    try:
        from isaaclab.utils.mesh import create_trimesh_from_geom_mesh

        mesh = create_trimesh_from_geom_mesh(prim)

        # Check if mesh is watertight
        if mesh.is_watertight:
            return abs(mesh.volume)
        else:
            # Use convex hull for non-watertight meshes
            convex_hull = mesh.convex_hull
            if convex_hull is not None:
                warnings.warn(
                    f"Mesh at {prim.GetPath()} is not watertight. Using convex hull approximation for volume."
                )
                return abs(convex_hull.volume)
            return 0.0
    except Exception as e:
        warnings.warn(f"Failed to compute mesh volume for {prim.GetPath()}: {e}")
        return 0.0


# Map prim types to volume calculation functions
_VOLUME_CALCULATORS: dict[str, callable] = {
    "Sphere": _compute_sphere_volume,
    "Cube": _compute_cube_volume,
    "Cylinder": _compute_cylinder_volume,
    "Capsule": _compute_capsule_volume,
    "Cone": _compute_cone_volume,
}
