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
    # UKF-M validity gate semantics. The ArUco marker sits at the WATER SURFACE and the
    # camera looks up, so visibility is bounded by DEPTH BELOW THE SURFACE — it improves as
    # the vehicle rises. The shipped gate below instead tests |z|, and wallscan's z is HEIGHT
    # above the tank floor (spawn 9.5, z_bottom 1.0), so it makes the fix valid in the LOWER
    # 8 m and invalid in the TOP 2 m: inverted. Two other places already assume the correct
    # direction — wallscan_env.py:702 ("Spawn at the water surface ... UKF-M marker view also
    # valid there") and :485 (the operating ceiling is surface - 1 m because getting *too
    # close* to the marker degrades the fix).
    #
    #   "legacy_height"       - |z| < ukfm_valid_max_depth. Inverted, but what
    #                           checkpoints/rb_train_model_7998.pt was trained against, so it
    #                           stays the default for reproducibility.
    #   "depth_below_surface" - (ukfm_surface_z - z) < ukfm_valid_max_depth, i.e. z > 2.0 for
    #                           the shipped 8.0 m range. Physically correct.
    #
    # MEASURED impact (2026-07-31, scripts/replay_wall_frame_ekf.py, 180 s run x 5 seeds,
    # wall-frame EKF): heading-estimate RMSE 1.56 deg legacy vs 1.71 deg corrected — nearly
    # the same, because both gates cover most of the 7.5 m scan band and differ only in WHICH
    # end goes blind (legacy: the top, including the ASCEND endpoint at 8.5; corrected: the
    # bottom, including the DESCEND endpoint at 1.0). With UKF-M off entirely it is 15.0 deg,
    # so the fix is load-bearing either way: a single echo sounder cannot observe small
    # heading offsets (a 5 deg error moves the range 4.3 mm against a 100 mm per-episode
    # bias). Correcting the direction matters not for accuracy but because losing the absolute
    # fix exactly at a phase endpoint is nobody's design.
    ukfm_gate: str = "legacy_height"
    ukfm_surface_z: float = 10.0       # = tank.TANK_HEIGHT, duplicated to keep this module
    #                                   import-free (tank.py pulls in isaaclab.sim)
    ukfm_min_standoff: float = 0.0     # m below the surface the fix needs to be usable at all
    #                                   (models the "too close to the marker" degradation);
    #                                   0 = off. Only used by "depth_below_surface".
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
    # --- channel splits (2026-07-31) ---------------------------------------------------
    # The knobs above conflate channels whose real specs differ by 4-30x, so a single value
    # cannot be right for both. These optional fields override them; None keeps the legacy
    # single-knob behaviour, which is what checkpoints/rb_train_model_7998.pt was trained on.
    #
    #   ins_noise         -> attitude [rad] AND angular rate [rad/s]      (differ ~34x)
    #   ins_att_bias_dr   -> roll/pitch bias AND heading bias             (differ ~4x)
    #
    # Roll/pitch are gravity-referenced and therefore absolute; heading needs the magnetometer,
    # which is both worse by spec and questionable inside a metal tank -- exactly why this
    # project takes absolute bearing from UKF-M instead. See SensorCfgDatasheet.
    ins_att_noise: float | None = None    # rad,   attitude channels only
    ins_gyro_noise: float | None = None   # rad/s, angular-rate channel only
    ins_heading_bias_dr: float | None = None  # rad, heading only (roll/pitch keep ins_att_bias_dr)
    # --- update periods (2026-08-09) ----------------------------------------------------
    # Real sensors do not refresh at the 50 Hz control rate; the estimator condition was
    # measurably flattered by assuming they do (fresh corrections every step). Period in
    # seconds between new readings; 0.0 = a fresh reading every control step, which is the
    # legacy behaviour and stays the default everywhere. Between updates a consumer holds the
    # last value (DVL/depth: prediction inputs) or skips the correction (sonar/UKF-M).
    sonar_period: float = 0.0
    dvl_period: float = 0.0
    depth_period: float = 0.0
    ukfm_period: float = 0.0


def att_noise(cfg: SensorCfg) -> float:
    """Attitude-channel noise std [rad]; falls back to the legacy combined ``ins_noise``."""
    return cfg.ins_noise if cfg.ins_att_noise is None else cfg.ins_att_noise


def gyro_noise(cfg: SensorCfg) -> float:
    """Angular-rate noise std [rad/s]; falls back to the legacy combined ``ins_noise``."""
    return cfg.ins_noise if cfg.ins_gyro_noise is None else cfg.ins_gyro_noise


def heading_bias_dr(cfg: SensorCfg) -> float:
    """Per-episode heading bias half-range [rad]; falls back to ``ins_att_bias_dr``."""
    return cfg.ins_att_bias_dr if cfg.ins_heading_bias_dr is None else cfg.ins_heading_bias_dr


@dataclass
class SensorCfgDatasheet(SensorCfg):
    """Sensor model built from the actual datasheets instead of conservative guesses.

    ``SensorCfg``'s own docstring says its numbers are placeholders "to be revisited against
    real sensor datasheets". Two of those datasheets were read on 2026-07-31 and the guesses
    turned out to be wildly pessimistic, in ways that changed conclusions:

    ## 3DM-GV7 (Parker MicroStrain, doc 8400-0008)

    Specification table, page 2:

    | spec | value | rad |
    |:---|---:|---:|
    | Gyro turn-on to turn-on bias [note 3: repeatability, <24 h] | 0.0054 deg/s | 9.4e-5 rad/s |
    | Gyro bias error over temperature | 0.066 deg/s | 1.15e-3 rad/s |
    | Gyro bias instability | 1.5 deg/h | 7.3e-6 rad/s |
    | Gyro noise density | 12 deg/h/sqrt(Hz) | 5.8e-5 rad/s/sqrt(Hz) |
    | Roll/pitch accuracy, dynamic | 0.5 deg | 8.7e-3 rad |
    | Heading accuracy, dynamic (AHRS) | 2 deg | 3.5e-2 rad |

    ``ins_gyro_bias_dr`` models a constant per-episode offset, so the matching spec is the
    TURN-ON bias (a fresh value each power-up, fixed during a run) — not the bias instability,
    which is the in-run Allan-variance floor. The shipped 0.02 rad/s is **213x** that figure:
    over one unexcited 39 s vertical leg it integrates to 44.7 deg of heading drift, against
    0.21 deg for the real part. That single number is what made the UKF-M gate choice look
    important (measured: crab +0.899 deg for the placeholder, and the gate difference collapses
    from +0.575 to +0.028 deg once the gyro is realistic).

    Noise: 12 deg/h/sqrt(Hz) over a 50 Hz loop (~25 Hz effective bandwidth) is
    5.8e-5 * sqrt(25) = 2.9e-4 rad/s, i.e. 34x below the shipped 0.01.

    ## Ping1D (Blue Robotics)

    Published: distance resolution **0.5% of range**, min range 0.3 m, usable 100 m, beam width
    **25 deg (-3 dB)**, 115 kHz. At the 1.5 m operating standoff, 0.5% = 7.5 mm.

    Two honest caveats, because Blue Robotics publishes resolution but NOT accuracy:

    * ``sonar_bias_dr`` here is NOT from the datasheet. It is sized from the speed-of-sound
      setting: the firmware default is 1500 m/s while tank water is 1470-1520 m/s, so a
      systematic +-1.5% = +-2.2 cm at 1.5 m is expected. Any transducer delay adds to it.
    * The 25 deg beam is WIDE — a 0.66 m footprint at 1.5 m. Every sonar model in this repo
      (``geometry.sonar_wall_distance``, ``mpc_reference.sonar_range``) is a single ray, which
      on a curved wall is optimistic: a real beam returns the nearest point in its cone.
      Correcting that is a separate modelling job, not a config change.

    Effect on the observability argument: the range signature of a 5 deg heading error is
    4.3 mm, so the signal-to-bias ratio goes from 1:23 (placeholder 100 mm) to 1:5
    (22 mm) or 1:1.7 against resolution alone. Still not enough to make the echo sounder the
    primary heading sensor — UKF-M stays load-bearing — but far from the hopeless 1:23.

    ## What is NOT datasheet-backed here

    ``dvl_*``, ``depth_*`` and ``ukfm_*`` keep their placeholder values: the DVL-A50, the
    pressure sensor and the marker-localisation stack were not looked up. They are the next
    ones to check — ``dvl_bias_dr`` in particular, because the code comment on it already notes
    that a velocity bias INTEGRATES into the dead-reckoned sway (drift = bias * 180 s).

    Kept SEPARATE from ``SensorCfg`` because this changes observation difficulty, so
    ``checkpoints/rb_train_model_7998.pt`` and every measurement so far were produced under the
    placeholders. ``tests/pkrc_wallscan/test_sensors.py`` pins both against the quoted specs.
    """

    # --- Ping1D --------------------------------------------------------------------------
    sonar_noise: float = 0.0075        # 0.5% of the 1.5 m operating range
    sonar_bias_dr: float = 0.022       # NOT a datasheet figure; +-1.5% speed-of-sound error
    # --- 3DM-GV7 ------------------------------------------------------------------------
    # Exact unit conversions, not hand-rounded, so each number stays traceable to its spec.
    ins_att_noise: float | None = 1.0e-3         # ~0.06 deg; a modelling choice, see below
    ins_gyro_noise: float | None = 2.909e-4      # 12 deg/h/sqrt(Hz) * sqrt(25 Hz)
    ins_att_bias_dr: float = 8.7266e-3           # 0.5 deg  (dynamic roll/pitch accuracy)
    ins_heading_bias_dr: float | None = 3.4907e-2  # 2 deg   (dynamic heading accuracy, AHRS)
    ins_gyro_bias_dr: float = 9.4248e-5          # 0.0054 deg/s (turn-on to turn-on bias)
    # ``ins_att_noise`` is the one INVENTED number here: the datasheet quotes accuracy (a total
    # error), not a noise/bias split. The accuracy figure is assigned to the bias term because a
    # slowly-varying error is the conservative reading for a controller — a bias never averages
    # out — leaving the noise small. Replace if a real Allan-variance attitude figure turns up.


@dataclass
class SensorCfgHW2026Bag(SensorCfgDatasheet):
    """Sensor model measured from the robot itself: the 2026-08-06 hero_ws teleop rosbag.

    Source: ``docs/experiments/hw_sensor_characterization.md`` (34 s bag, depth 0.6-0.7 m,
    analysed by ``marinelab/scripts/experiments/hw_bag_characterize.py``). Noise figures are
    std(diff)/sqrt(2) over the moving robot, i.e. honest UPPER bounds; every one still came in
    ~5x below the placeholder guesses. What the bag could NOT characterize keeps the parent
    (datasheet/placeholder) value:

    * ``ukfm_noise`` / ``ukfm_bias_dr`` — the bag's /ukfm/odom was pure dead reckoning
      (0 ArUco messages, /ukfm/odom_validated never published), so its 0.5 mm "noise" is
      filter smoothness, not absolute accuracy. Placeholder 0.03 stays until a marker-visible
      bag exists.
    * gyro — measured 1.6-2.1 mrad/s bounds it, but includes real rotation; the 3DM-GV7
      datasheet figure (parent) is finer and consistent with the bound.
    * biases — a 34 s bag cannot see per-run constants; parent values stay.

    The load-bearing change is the PERIODS: DVL 9.5 Hz, depth 5.0 Hz, wall range 9.5 Hz,
    UKF-M 31.5 Hz against the sim's fresh-sample-every-step (50 Hz) legacy assumption.
    """

    # measured hf-noise (upper bounds, moving robot)
    sonar_noise: float = 0.034     # /ukfm/wall_distance; sim treats it as the wall-range channel
    depth_noise: float = 0.0042    # /bar10xt/depth
    dvl_noise: float = 0.004       # /dvl/data velocity xy
    # measured update periods (median rates over the bag)
    sonar_period: float = 0.105    # 9.5 Hz
    dvl_period: float = 0.105      # 9.5 Hz
    depth_period: float = 0.200    # 5.0 Hz
    ukfm_period: float = 0.032     # 31.5 Hz


@dataclass
class SensorCfgHW2026BagAruco(SensorCfgHW2026Bag):
    """HW2026Bag with the UKF-M channel corrected to the MARKER-VISIBLE bag (20260806_122531).

    The 122731 bag had zero ArUco fixes, so HW2026Bag inherited the placeholder UKF-M. The
    122531 bag (46 s, depth 0.6-3.8 m, marker visible from t=16.5 s) measured the absolute
    channel itself: /ukfm/odom publishes at 31.5 Hz, but that is filter output — absolute
    information (ArUco fixes) arrives at ~1.3 Hz (median gap 0.51 s, max 2.48 s), and the
    innovation |odom - aruco| at fix time is 6.5 cm median / 10.8 cm p90. The sim's ukfm
    update models exactly such an absolute fix, so it gets the fix cadence and the fix-time
    error, not the odom stream's smoothness. Correction jumps are absorbed by the node's
    low-pass (measured step 0.02 cm), so a jump model is not needed.
    """

    ukfm_noise: float = 0.065   # m/rad; median innovation at fix times (122531 bag)
    ukfm_period: float = 0.77   # s; measured ArUco fix rate ~1.3 Hz


class SensorRateGate:
    """Tracks which sensors have a FRESH reading at a given time — pure, no torch state.

    ``fresh(name, t)`` returns True when ``cfg.<name>_period`` has elapsed since the last
    fresh tick (and latches ``t``), so a 50 Hz loop over a 9.5 Hz sensor sees roughly every
    fifth tick fresh. Period 0.0 = always fresh (legacy). Consumers decide what "not fresh"
    means for each channel: hold the last value (prediction inputs like DVL velocity, depth)
    or skip the measurement update (sonar, UKF-M) so one reading never corrects twice.
    """

    NAMES = ("sonar", "dvl", "depth", "ukfm")

    def __init__(self, cfg: SensorCfg, t0: float = 0.0):
        self._period = {n: float(getattr(cfg, f"{n}_period", 0.0)) for n in self.NAMES}
        self._last = {n: None for n in self.NAMES}
        self._t0 = t0

    def reset(self, t0: float = 0.0) -> None:
        self._last = {n: None for n in self.NAMES}
        self._t0 = t0

    def fresh(self, name: str, t: float) -> bool:
        period = self._period[name]
        last = self._last[name]
        if period <= 0.0 or last is None or (t - last) >= period - 1e-9:
            self._last[name] = t
            return True
        return False


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


def ukfm_in_range(z: torch.Tensor, cfg: SensorCfg) -> torch.Tensor:
    """Is the vehicle within the surface marker's usable band? (tilt is gated separately.)

    Both gate semantics live here so the difference is one readable place rather than a
    condition buried in ``apply_sensors``; see ``SensorCfg.ukfm_gate`` for why there are two
    and what was measured. ``z`` is height above the tank floor.
    """
    if cfg.ukfm_gate == "depth_below_surface":
        depth = cfg.ukfm_surface_z - z
        in_range = depth < cfg.ukfm_valid_max_depth
        if cfg.ukfm_min_standoff > 0.0:
            in_range = in_range & (depth > cfg.ukfm_min_standoff)
        return in_range
    if cfg.ukfm_gate != "legacy_height":
        raise ValueError(f"unknown ukfm_gate {cfg.ukfm_gate!r}; "
                         "expected 'legacy_height' or 'depth_below_surface'")
    return z.abs() < cfg.ukfm_valid_max_depth


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
    up_vec = up_vec_gt + noise(n, 3, std=att_noise(cfg)) + b.get("up_vec", 0.0)

    heading_noisy = heading + noise(n, std=att_noise(cfg)) + b.get("heading", 0.0)
    heading_sc = torch.stack([torch.sin(heading_noisy), torch.cos(heading_noisy)], dim=-1)

    ang_vel_out = ang_vel + noise(n, 3, std=gyro_noise(cfg)) + b.get("ang_vel", 0.0)
    lin_vel_out = lin_vel + noise(n, 3, std=cfg.dvl_noise) + b.get("lin_vel", 0.0)

    tilt = torch.acos(torch.clamp(up_vec_gt[:, 2], -1.0, 1.0))
    valid = (ukfm_in_range(pos[:, 2], cfg) & (tilt < cfg.ukfm_tilt_max)).to(dtype)

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
