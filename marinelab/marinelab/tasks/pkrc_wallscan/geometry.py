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
    return wall_distance(sensor_xy, heading + yaw_offset, radius)


def radial_clearance(pos_xy: torch.Tensor, radius: float) -> torch.Tensor:
    return radius - torch.linalg.norm(pos_xy, dim=-1)
