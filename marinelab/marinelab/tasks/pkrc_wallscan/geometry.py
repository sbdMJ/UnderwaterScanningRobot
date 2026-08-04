# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch


def wall_distance(pos_xy: torch.Tensor, heading: torch.Tensor, radius: float) -> torch.Tensor:
    """Forward-ray distance to inner cylinder wall (tank centered at origin, +heading = yaw about z).
    Solves |p + t*dir|^2 = radius^2 for t>0. Robots are inside (|p| < radius)."""
    d = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)   # [N,2]
    p = pos_xy
    b = (p * d).sum(-1)
    c = (p * p).sum(-1) - radius * radius
    disc = torch.clamp(b * b - c, min=0.0)
    t = -b + torch.sqrt(disc)                                           # positive root (inside)
    return torch.clamp(t, min=0.0)


def sonar_wall_distance(
    pos_xy: torch.Tensor,
    heading: torch.Tensor,
    mount_xy: torch.Tensor,
    yaw_offset: torch.Tensor | float,
    radius: float,
    beam_half_angle: float = 0.0,
) -> torch.Tensor:
    """Wall distance measured by a sonar mounted at body-frame offset ``mount_xy`` (in-plane
    x=forward,y=left) with beam azimuth ``yaw_offset`` relative to body heading. Transforms the
    sensor to the tank frame, then rays from there (so mount pose is reflected, not body center).
    ``mount_xy`` is [..,2], ``yaw_offset`` scalar or [..]; both broadcast against ``heading`` [N]."""
    c = torch.cos(heading)
    s = torch.sin(heading)
    sx = pos_xy[..., 0] + c * mount_xy[..., 0] - s * mount_xy[..., 1]
    sy = pos_xy[..., 1] + s * mount_xy[..., 0] + c * mount_xy[..., 1]
    sensor_xy = torch.stack([sx, sy], dim=-1)
    if beam_half_angle <= 0.0:
        return wall_distance(sensor_xy, heading + yaw_offset, radius)

    # A real transducer is a CONE, not a ray: the Ping1D is 25 deg (-3 dB), a 0.66 m footprint
    # at the 1.5 m operating standoff. Seen from inside, the tank wall is CONCAVE and the range
    # along a ray grows with the angle off the outward radial, so the shortest return inside the
    # cone -- which is what an echo sounder reports -- is:
    #
    #   |phi| <= half_angle : the cone still contains the perpendicular, so the reading is the
    #                         TRUE clearance, and heading error is completely invisible in it
    #   |phi| >  half_angle : the reading is the range at (|phi| - half_angle), i.e. only the
    #                         part of the misalignment the cone cannot reach
    #
    # Both halves matter and pull opposite ways: a wide beam makes wall distance ROBUST to crab
    # (at 30 deg the one-sided over-read drops from 16.5 cm to 5.4 cm) while removing the last
    # trace of heading information from the range.
    theta = torch.atan2(sensor_xy[..., 1], sensor_xy[..., 0])
    phi = heading + yaw_offset - theta
    phi = torch.atan2(torch.sin(phi), torch.cos(phi))
    eff = (phi.abs() - beam_half_angle).clamp(min=0.0)
    return wall_distance(sensor_xy, theta + eff, radius)


def radial_clearance(pos_xy: torch.Tensor, radius: float) -> torch.Tensor:
    return radius - torch.linalg.norm(pos_xy, dim=-1)
