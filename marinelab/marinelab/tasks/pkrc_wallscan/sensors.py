# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SensorCfg:
    """Noise/dropout params for ground-truth -> sensor-reading conversion.

    Defaults are unresolved/tunable per spec §13 ("초기엔 보수적으로") — picked here as
    conservative placeholders, to be revisited against real sensor datasheets:
    Ping Sonar / pressure depth / 3DM-GV7 INS / DVL-A50 / UKF-M(surface marker).
    """
    sonar_noise: float = 0.05          # m, std
    depth_noise: float = 0.02          # m, std
    ins_noise: float = 0.01            # rad (attitude/ang_vel), std
    dvl_noise: float = 0.02            # m/s, std
    ukfm_noise: float = 0.03           # m / rad (xy, yaw), std
    ukfm_valid_max_depth: float = 8.0  # m; tank is 10 m deep, marker view lost before that
    ukfm_tilt_max: float = 0.5         # rad (~28.6 deg) tilt before marker lost
    # Sonar mount extrinsics (body frame): nominal pose + per-episode DR half-ranges (spec §7).
    # Nominal from Stonefish pkrc.scn: sonar sits at camera_front's horizontal position
    # (-0.1 fwd, ~0 lateral); its mast height (15cm above camera) does NOT affect horizontal
    # wall distance on a vertical cylinder wall, so only the horizontal offset is modeled.
    sonar_mount_pos: tuple[float, float] = (0.10, 0.0)  # x=forward, y=left offset from body origin
    sonar_yaw_offset: float = 0.0                       # beam azimuth relative to body heading (rad)
    sonar_mount_dr: float = 0.0                        # uniform +-half-range on mount xy (m); 0=off
    sonar_yaw_dr: float = 0.0                          # uniform +-half-range on beam azimuth (rad); 0=off
    # Per-episode sensor BIAS DR (sim2real): each sensor gets a constant per-episode offset drawn
    # uniform in +-<field> (0 = off). Applied to the OBSERVATION only -- tracking rewards stay on
    # ground truth, so a bias the policy cannot observe never distorts the learning objective.
    sonar_bias_dr: float = 0.0     # m       (echosounder range offset)
    depth_bias_dr: float = 0.0     # m       (pressure/depth offset)
    ins_att_bias_dr: float = 0.0   # ~rad    (INS attitude: up-vector components + heading)
    ins_gyro_bias_dr: float = 0.0  # rad/s   (INS gyro / angular velocity)
    dvl_bias_dr: float = 0.0       # m/s     (DVL velocity offset)
    ukfm_bias_dr: float = 0.0      # m / rad (UKF-M position xy + yaw)


def _randn(shape, std: float, gen: torch.Generator | None, dtype, device) -> torch.Tensor:
    if std == 0.0:
        return torch.zeros(shape, dtype=dtype, device=device)
    if gen is None:
        return torch.randn(shape, dtype=dtype, device=device) * std
    return torch.randn(shape, generator=gen, dtype=dtype, device=device) * std


def _body_up(quat: torch.Tensor) -> torch.Tensor:
    """Body +z axis rotated into world frame by quat (w, x, y, z)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    qvec = torch.stack([x, y, z], dim=-1)
    v = torch.zeros_like(qvec)
    v[..., 2] = 1.0
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2.0 * (w.unsqueeze(-1) * uv + uuv)


def apply_sensors(gt: dict, cfg: SensorCfg, gen: torch.Generator | None, bias: dict | None = None) -> dict:
    """Ground truth -> noisy/dropout sensor readings (sonar/depth/INS/DVL/UKF-M).

    `bias` (optional) holds per-episode constant offsets per channel (keys: sonar, depth, up_vec,
    heading, ang_vel, lin_vel, ukfm_xy, ukfm_yaw); missing keys default to 0. Bias hits the
    observation only -- ground-truth-based rewards are computed upstream and never see it.
    """
    pos, quat = gt["pos"], gt["quat"]
    ang_vel, lin_vel, heading = gt["ang_vel"], gt["lin_vel"], gt["heading"]
    n, dtype, device = pos.shape[0], pos.dtype, pos.device
    b = bias or {}

    def noise(*shape, std):
        return _randn(shape, std, gen, dtype, device)

    # Observation sonar reads the (DR-mount) measured distance if provided, else nominal wall_dist.
    sonar = gt.get("wall_dist_meas", gt["wall_dist"]) + noise(n, std=cfg.sonar_noise) + b.get("sonar", 0.0)
    depth = pos[:, 2] + noise(n, std=cfg.depth_noise) + b.get("depth", 0.0)

    up_vec_gt = _body_up(quat)
    up_vec = up_vec_gt + noise(n, 3, std=cfg.ins_noise) + b.get("up_vec", 0.0)

    heading_noisy = heading + noise(n, std=cfg.ins_noise) + b.get("heading", 0.0)
    heading_sc = torch.stack([torch.sin(heading_noisy), torch.cos(heading_noisy)], dim=-1)

    ang_vel_out = ang_vel + noise(n, 3, std=cfg.ins_noise) + b.get("ang_vel", 0.0)
    lin_vel_out = lin_vel + noise(n, 3, std=cfg.dvl_noise) + b.get("lin_vel", 0.0)

    tilt = torch.acos(torch.clamp(up_vec_gt[:, 2], -1.0, 1.0))
    valid = ((pos[:, 2].abs() < cfg.ukfm_valid_max_depth) & (tilt < cfg.ukfm_tilt_max)).to(dtype)

    # UKF-M bias only manifests when the surface marker is visible (valid), like its noise.
    ukfm_xy = pos[:, :2] + (noise(n, 2, std=cfg.ukfm_noise) + b.get("ukfm_xy", 0.0)) * valid.unsqueeze(-1)
    ukfm_yaw = heading + (noise(n, std=cfg.ukfm_noise) + b.get("ukfm_yaw", 0.0)) * valid

    return dict(
        sonar=sonar, depth=depth, up_vec=up_vec, heading_sc=heading_sc,
        ang_vel=ang_vel_out, lin_vel=lin_vel_out,
        ukfm_xy=ukfm_xy, ukfm_yaw=ukfm_yaw, ukfm_valid=valid,
    )


def demo():
    def _gt(n=4096):
        return dict(pos=torch.zeros(n, 3), quat=torch.tensor([[1., 0, 0, 0]]).repeat(n, 1),
                    ang_vel=torch.zeros(n, 3), lin_vel=torch.zeros(n, 3),
                    heading=torch.zeros(n), wall_dist=torch.full((n,), 4.0))

    cfg = SensorCfg(sonar_noise=0.05, ukfm_valid_max_depth=8.0)
    out = apply_sensors(_gt(), cfg, torch.Generator().manual_seed(0))
    err = out["sonar"] - 4.0
    assert err.abs().mean() < 0.05 and err.std() > 0.01

    # Per-episode bias shifts the channel mean by exactly the bias; noise-only run stays ~0.
    n = 4096
    biased = apply_sensors(_gt(n), cfg, torch.Generator().manual_seed(0),
                           bias={"sonar": torch.full((n,), 0.3)})
    assert abs((biased["sonar"] - 4.0).mean().item() - 0.3) < 0.01, "sonar bias not applied"

    gt = _gt()
    gt["pos"][:, 2] = -9.0
    cfg2 = SensorCfg(ukfm_valid_max_depth=8.0)
    out2 = apply_sensors(gt, cfg2, torch.Generator().manual_seed(0))
    assert out2["ukfm_valid"].max() == 0.0

    print("sensors.py demo OK")


if __name__ == "__main__":
    demo()
