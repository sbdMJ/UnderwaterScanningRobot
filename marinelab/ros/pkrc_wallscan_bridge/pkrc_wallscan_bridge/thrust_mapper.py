# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""ROS2 node: /wallscan/u (normalized thrust, sim order) -> /wallscan/current_cmd (A, VESC order).

Thin wrapper around the pure, natively-tested ``marinelab.control.thrust_current_map``.
The teleop node (sole CAN owner) consumes /wallscan/current_cmd only in its auto mode and
still applies wiring polarity, deadzone compensation, clamps, ramping and e-stop — see
docs/experiments/sim-to-real/thruster_mapping.md §5 for the division of labour.

Safety: if /wallscan/u goes stale (> stale_zero_s) a zero command is published so the
teleop side ramps down even if it misses its own staleness check.
"""
from __future__ import annotations

from .marinelab_loader import load_marinelab

load_marinelab()

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import Float64MultiArray  # noqa: E402

from marinelab.control.thrust_current_map import ThrustCurrentMap  # noqa: E402


class ThrustMapper(Node):
    def __init__(self):
        super().__init__("thrust_mapper")
        p = self.declare_parameter
        p("order", [0, 1, 2, 3, 4, 5])
        p("sign", [1.0, 1.0, -1.0, -1.0, 1.0, -1.0])  # bench-derived, 2026-08-09
        p("amps_at_full", [3.0, 3.0, 3.0, 3.0, 5.0, 5.0])
        p("amps_limit", [3.0, 3.0, 3.0, 3.0, 5.0, 5.0])
        p("newton_per_amp", [0.0] * 6)  # all-zero = uncalibrated fallback
        p("max_thrust", 40.0)
        p("stale_zero_s", 0.5)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        k = [float(v) for v in g("newton_per_amp")]
        self.map = ThrustCurrentMap(
            order=tuple(int(v) for v in g("order")),
            sign=tuple(float(v) for v in g("sign")),
            amps_at_full=tuple(float(v) for v in g("amps_at_full")),
            amps_limit=tuple(float(v) for v in g("amps_limit")),
            newton_per_amp=tuple(k) if any(v > 0.0 for v in k) else None,
            max_thrust=float(g("max_thrust")))
        self.stale_zero = float(g("stale_zero_s"))
        self._last_rx = None

        self.create_subscription(Float64MultiArray, "/wallscan/u", self._on_u, 10)
        self.pub = self.create_publisher(Float64MultiArray, "/wallscan/current_cmd", 10)
        self.create_timer(0.2, self._watchdog)
        tag = "CALIBRATED" if self.map.calibrated else \
            "UNCALIBRATED (teleop manual scale — low-bandwidth trials only)"
        self.get_logger().info(f"thrust mapper up [{tag}], order={self.map.order}")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_u(self, m: Float64MultiArray) -> None:
        if len(m.data) != 6:
            self.get_logger().warning(f"/wallscan/u carries {len(m.data)} values, want 6")
            return
        self._last_rx = self._now()
        out = Float64MultiArray()
        out.data = [float(v) for v in self.map.map(list(m.data))]
        self.pub.publish(out)

    def _watchdog(self) -> None:
        if self._last_rx is not None and self._now() - self._last_rx > self.stale_zero:
            out = Float64MultiArray()
            out.data = [0.0] * 6
            self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ThrustMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
