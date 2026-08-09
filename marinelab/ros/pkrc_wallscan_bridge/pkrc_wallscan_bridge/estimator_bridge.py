# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""ROS2 node: hero_ws sensor topics -> WallFrameStateEstimator -> /wallscan/state.

Thin by design: every decision that matters (freshness semantics, frame conversions, the
absolute-bearing hand-off that the EKF's arc-length correction needs) lives in the pure,
natively-tested ``marinelab.control.hw_bridge.TopicSampleAssembler``. This file only wires
rclpy subscriptions into it and publishes the estimate.

Topics consumed (defaults match the 2026-08-06 hero_ws bags):

    /dvl/data              dvl_msgs/DVL          body velocity (prediction input, ~9.5 Hz)
    /imu/data              sensor_msgs/Imu       attitude + rates (prediction input, 100 Hz)
    /bar10xt/depth         std_msgs/Float64      depth below surface (held, ~5 Hz)
    /ukfm/wall_distance    std_msgs/Float32      wall range (consumed once, ~9.5 Hz)
    /ukfm/odom_validated   nav_msgs/Odometry     ABSOLUTE fix, only while ArUco is fresh

``/ukfm/odom_validated`` (never plain ``/ukfm/odom``): the plain stream keeps publishing
dead-reckoned poses with no absolute content — measured 2026-08-06: a 34 s bag of it with
zero ArUco fixes. The validated stream republishes at ~19 Hz while a fix is younger than
the UKFM node's marker_timeout, so between true ArUco detections (~1.3 Hz) its poses carry
a little dead reckoning; that error is bounded by the timeout and was measured at 6.5 cm
median (docs/experiments/hw_sensor_characterization.md §5).

Published:

    /wallscan/state            nav_msgs/Odometry       tank-frame pose + body twist
    /wallscan/estimator_debug  std_msgs/Float64MultiArray
        [r, phi, s, initialized, age_dvl, age_imu, age_depth, age_wall, age_ukfm]
"""
from __future__ import annotations

import math

from .marinelab_loader import load_marinelab

load_marinelab()

import rclpy  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Imu  # noqa: E402
from std_msgs.msg import Float32, Float64, Float64MultiArray  # noqa: E402

from marinelab.control.estimator import WallFrameStateEstimator  # noqa: E402
from marinelab.control.hw_bridge import TankCalib, TopicSampleAssembler  # noqa: E402
from marinelab.tasks.pkrc_wallscan.wall_frame_ekf import WallFrameEKFCfg  # noqa: E402


def _yaw_roll_pitch(q) -> tuple[float, float, float]:
    sinr = 2 * (q.w * q.x + q.y * q.z)
    cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
    sinp = max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))
    siny = 2 * (q.w * q.z + q.x * q.y)
    cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy), math.atan2(sinr, cosr), math.asin(sinp)


class EstimatorBridge(Node):
    def __init__(self):
        super().__init__("wallscan_estimator_bridge")
        p = self.declare_parameter
        # survey calibration — marker-world -> tank frame (see TankCalib)
        p("tank_height", 10.0), p("tank_radius", 6.0)
        p("marker_x", 0.0), p("marker_y", 0.0), p("marker_yaw", 0.0)
        p("rate_hz", 50.0)
        p("wall_topic", "/ukfm/wall_distance")
        p("stale_warn_s", 1.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.calib = TankCalib(
            tank_height=float(g("tank_height")), tank_radius=float(g("tank_radius")),
            marker_x=float(g("marker_x")), marker_y=float(g("marker_y")),
            marker_yaw=float(g("marker_yaw")))
        self.asm = TopicSampleAssembler(self.calib)
        self.est = WallFrameStateEstimator(WallFrameEKFCfg(tank_radius=self.calib.tank_radius))
        self.initialized = False
        self.dt = 1.0 / float(g("rate_hz"))
        self.stale_warn = float(g("stale_warn_s"))

        try:
            from dvl_msgs.msg import DVL
        except ImportError as e:  # pragma: no cover - build-environment error
            raise SystemExit("dvl_msgs not built in this workspace (colcon build dvl_msgs)") from e

        self.create_subscription(DVL, "/dvl/data", self._on_dvl, 10)
        self.create_subscription(Imu, "/imu/data", self._on_imu, 50)
        self.create_subscription(Float64, "/bar10xt/depth", self._on_depth, 10)
        self.create_subscription(Float32, str(g("wall_topic")), self._on_wall, 10)
        self.create_subscription(Odometry, "/ukfm/odom_validated", self._on_ukfm, 10)

        self.pub_state = self.create_publisher(Odometry, "/wallscan/state", 10)
        self.pub_debug = self.create_publisher(Float64MultiArray, "/wallscan/estimator_debug", 10)
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"estimator bridge up: rate {1/self.dt:.0f} Hz, tank R={self.calib.tank_radius} "
            f"H={self.calib.tank_height}, marker offset ({self.calib.marker_x}, "
            f"{self.calib.marker_y}, yaw {self.calib.marker_yaw})")

    # -- callbacks: plain floats into the pure assembler ---------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_dvl(self, m) -> None:
        if getattr(m, "velocity_valid", True):
            v = m.velocity
            self.asm.feed_dvl(v.x, v.y, v.z, self._now())

    def _on_imu(self, m: Imu) -> None:
        _, roll, pitch = _yaw_roll_pitch(m.orientation)
        w = m.angular_velocity
        self.asm.feed_imu(w.x, w.y, w.z, roll, pitch, self._now())

    def _on_depth(self, m: Float64) -> None:
        self.asm.feed_depth(m.data, self._now())

    def _on_wall(self, m: Float32) -> None:
        self.asm.feed_wall_range(float(m.data), self._now())

    def _on_ukfm(self, m: Odometry) -> None:
        pos = m.pose.pose.position
        yaw, _, _ = _yaw_roll_pitch(m.pose.pose.orientation)
        self.asm.feed_ukfm(pos.x, pos.y, yaw, self._now())

    # -- 50 Hz loop -----------------------------------------------------------
    def _tick(self) -> None:
        now = self._now()
        sample = self.asm.assemble(now)
        if sample is None:
            return
        if not self.initialized:
            if sample.ukfm is None:
                return  # wait for the first validated marker fix to anchor the filter
            r0, phi0, theta0 = sample.ukfm
            self.est.reset(r0=r0, phi0=phi0, theta0=theta0)
            self.initialized = True
            self.get_logger().info(
                f"filter anchored at r={r0:.2f} phi={phi0:.2f} theta={theta0:.2f}")

        veh = self.est.step(sample, self.dt)
        ages = self.asm.ages(now)
        for ch in ("dvl", "depth"):
            if ages.get(ch, 0.0) > self.stale_warn:
                self.get_logger().warning(f"{ch} stale: {ages[ch]:.1f} s", throttle_duration_sec=5.0)

        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "tank"
        out.child_frame_id = "base_link"
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = veh.pos_w
        w, x, y, z = veh.quat_wb
        out.pose.pose.orientation.w, out.pose.pose.orientation.x = float(w), float(x)
        out.pose.pose.orientation.y, out.pose.pose.orientation.z = float(y), float(z)
        (out.twist.twist.linear.x, out.twist.twist.linear.y,
         out.twist.twist.linear.z) = (float(v) for v in veh.lin_vel_b)
        (out.twist.twist.angular.x, out.twist.twist.angular.y,
         out.twist.twist.angular.z) = (float(v) for v in veh.ang_vel_b)
        self.pub_state.publish(out)

        dbg = Float64MultiArray()
        dbg.data = [self.est.ekf.r, self.est.ekf.phi, self.est.s_hat, 1.0,
                    ages.get("dvl", -1.0), ages.get("imu", -1.0), ages.get("depth", -1.0),
                    ages.get("wall", -1.0), ages.get("ukfm", -1.0)]
        self.pub_debug.publish(dbg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EstimatorBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
