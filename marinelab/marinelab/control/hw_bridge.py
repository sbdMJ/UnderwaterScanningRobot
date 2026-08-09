# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""ROS-topic-to-SensorSample assembly — the pure half of the hardware estimator bridge.

A rclpy node (``marinelab/ros/pkrc_wallscan_bridge``) owns the subscriptions and feeds this
class plain floats; everything that can be wrong in interesting ways — freshness semantics,
frame conversions, the absolute-bearing hand-off the EKF's s-correction depends on — lives
here, numpy-only, so the native test suite covers it without ROS installed.

Freshness contract (mirrors ``SimSensorStream`` rate-and-hold, which mirrors this):

* DVL velocity / depth / IMU are PREDICTION inputs — ``assemble`` holds the latest value.
* wall range / UKF-M fix are MEASUREMENT updates — consumed once. ``assemble`` includes a
  reading only if a new message arrived since the previous ``assemble`` call, so one echo
  or one marker fix never corrects the filter twice at the 50 Hz loop rate.

Frame conventions (matches ``control.types.VehicleState`` and the sim):

* tank frame: origin on the tank axis at the FLOOR, z up. The pressure sensor reports depth
  below surface, so ``z = tank_height - depth``.
* the UKF-M node publishes in its marker-anchored world frame; ``TankCalib`` carries the
  rigid 2-D transform marker-world -> tank (surveyed at setup). The fix's absolute bearing
  ``theta = atan2(y_t, x_t)`` rides along as the third ukfm element — that bearing is what
  lets ``WallFrameEKF.update_ukfm`` correct the arc-length state (the e5_ekf s-drift fix),
  so dropping it silently re-opens the drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .estimator import SensorSample


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class TankCalib:
    """Survey constants mapping the robot's sensor frames into the tank frame."""

    tank_height: float = 10.0  # m, floor -> surface (z = tank_height - pressure depth)
    tank_radius: float = 6.0
    # Rigid 2-D transform: p_tank = R(marker_yaw) @ p_marker + (marker_x, marker_y).
    # Identity = marker world origin sits on the tank axis, axes aligned.
    marker_x: float = 0.0
    marker_y: float = 0.0
    marker_yaw: float = 0.0

    def marker_to_tank(self, x_m: float, y_m: float, yaw_m: float) -> tuple[float, float, float]:
        c, s = math.cos(self.marker_yaw), math.sin(self.marker_yaw)
        return (c * x_m - s * y_m + self.marker_x,
                c * y_m + s * x_m + self.marker_y,
                _wrap(yaw_m + self.marker_yaw))


class TopicSampleAssembler:
    """Latest-value cache with per-channel freshness; one ``assemble`` per control tick."""

    #: prediction inputs that must exist before the first sample can be assembled
    REQUIRED = ("dvl", "imu", "depth")

    def __init__(self, calib: TankCalib | None = None):
        self.calib = calib or TankCalib()
        self._dvl: tuple[float, float, float] | None = None  # v_bx, v_by, v_bz
        self._imu: tuple[float, float, float, float, float] | None = None  # gx, gy, gz, roll, pitch
        self._z: float | None = None  # tank z (already converted)
        self._wall: float | None = None  # unconsumed wall range, if any
        self._ukfm: tuple[float, float, float] | None = None  # unconsumed (r, phi, theta)
        self._age: dict[str, float] = {}  # last message stamp per channel

    # -- feeds (called from ROS callbacks; plain floats only) ---------------
    def feed_dvl(self, v_bx: float, v_by: float, v_bz: float, stamp: float) -> None:
        self._dvl = (float(v_bx), float(v_by), float(v_bz))
        self._age["dvl"] = stamp

    def feed_imu(self, gyro_x: float, gyro_y: float, gyro_z: float,
                 roll: float, pitch: float, stamp: float) -> None:
        self._imu = (float(gyro_x), float(gyro_y), float(gyro_z), float(roll), float(pitch))
        self._age["imu"] = stamp

    def feed_depth(self, depth_below_surface: float, stamp: float) -> None:
        self._z = self.calib.tank_height - float(depth_below_surface)
        self._age["depth"] = stamp

    def feed_wall_range(self, range_m: float, stamp: float) -> None:
        self._wall = float(range_m)
        self._age["wall"] = stamp

    def feed_ukfm(self, x_m: float, y_m: float, yaw_m: float, stamp: float) -> None:
        """A validated marker fix (marker-world frame) -> tank-frame (r, phi, theta).

        Feed ONLY from the validated stream (``/ukfm/odom_validated``): the plain
        ``/ukfm/odom`` keeps publishing dead-reckoned poses with no absolute content
        (measured 2026-08-06: 34 s of it with zero ArUco fixes), and treating those as
        absolute fixes would launder drift back into the filter as truth.
        """
        x_t, y_t, yaw_t = self.calib.marker_to_tank(float(x_m), float(y_m), float(yaw_m))
        r = math.hypot(x_t, y_t)
        theta = math.atan2(y_t, x_t)
        self._ukfm = (r, _wrap(yaw_t - theta), theta)
        self._age["ukfm"] = stamp

    # -- assembly ------------------------------------------------------------
    def ready(self) -> bool:
        return self._dvl is not None and self._imu is not None and self._z is not None

    def ages(self, now: float) -> dict[str, float]:
        """Seconds since each channel's last message — the node's health telemetry."""
        return {k: now - t for k, t in self._age.items()}

    def assemble(self, stamp: float) -> SensorSample | None:
        """One control tick's sample; None until every prediction input has arrived.

        Consumes the pending wall range / ukfm fix (if any) so the next tick carries
        ``sonar=None`` / ``ukfm=None`` unless a new message lands in between.
        """
        if not self.ready():
            return None
        gx, gy, gz, roll, pitch = self._imu
        v_bx, v_by, v_bz = self._dvl
        wall, self._wall = self._wall, None
        ukfm, self._ukfm = self._ukfm, None
        return SensorSample(
            v_bx=v_bx, v_by=v_by, v_bz=v_bz,
            gyro_x=gx, gyro_y=gy, gyro_z=gz, roll=roll, pitch=pitch,
            sonar=wall, depth=self._z, ukfm=ukfm, stamp=stamp,
        )
