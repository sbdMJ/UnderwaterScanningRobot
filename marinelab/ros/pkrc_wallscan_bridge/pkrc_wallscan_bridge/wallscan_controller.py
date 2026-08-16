# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""ROS2 node: /wallscan/state -> WallScanControlLoop -> /wallscan/u.

The scan brain is the pure ``marinelab.control.scan_loop.WallScanControlLoop`` — the SAME
object the sim runner (`_sim_loop.run_mpc_cell`) drives, verified by a regression cell
(objective reproduced to +0.015%) and by the §8 native closed-loop tests. This node only
wires topics into it and enforces the safety envelope.

Safety envelope (the teleop node keeps the CAN bus and therefore final authority):

* ``/wallscan/enable`` (Bool, default OFF) gates output — while disabled the node still
  runs the estimator input and publishes diagnostics, but ``/wallscan/u`` stays zero.
* state staleness > ``stale_zero_s`` -> zero thrust + throttled warning.
* any controller exception -> zero thrust + error log (never crash the loop).

Inputs:
    /wallscan/state            nav_msgs/Odometry       from estimator_bridge (drives ticks)
    /wallscan/estimator_debug  std_msgs/Float64MultiArray  [r, phi, s, ...] -> s_hat
    /wallscan/enable           std_msgs/Bool
Outputs:
    /wallscan/u                std_msgs/Float64MultiArray  (6,) normalized [-1, 1]
    /wallscan/controller_debug std_msgs/Float64MultiArray
        [enabled, phase, cycles, z_ref, s_ref, solve_ms, status]

The NMPC needs plant parameters; the Jetson has no sim env, so they come from the JSON the
sim exports (``marinelab/config/pkrc_plant_fixed_tam.json``, PlantParams.to_json). acados +
casadi must be built on the Jetson (Phase D) before the nominal/bo/ssi methods construct.
"""
from __future__ import annotations

import math
import os

from .marinelab_loader import load_marinelab

load_marinelab()

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import Bool, Float64MultiArray  # noqa: E402

from marinelab.control.scan_loop import WallScanControlLoop  # noqa: E402
from marinelab.control.types import VehicleState  # noqa: E402
from marinelab.tasks.pkrc_wallscan.mpc_controller import PlantParams  # noqa: E402
from marinelab.tasks.pkrc_wallscan.mpc_reference import WallScanMPCCfg  # noqa: E402
from marinelab.tasks.pkrc_wallscan.scan_state_machine import ScanCfg  # noqa: E402


def _default_plant_json() -> str:
    root = os.environ.get("MARINELAB_ROOT", "")
    return os.path.join(root, "config", "pkrc_plant_fixed_tam.json") if root else ""


def _build_controller(node: Node, method: str, plant: PlantParams, mpc_cfg: WallScanMPCCfg,
                      horizon: int, rti_iters: int, export_root: str):
    g = lambda n: node.get_parameter(n).value  # noqa: E731
    kwargs = dict(plant=plant, mpc_cfg=mpc_cfg, horizon=horizon, rti_iters=rti_iters,
                  code_export_root=export_root)
    params_json = str(g("params_json"))
    if method in ("nominal", "bo"):
        from marinelab.control.fixed_nmpc import FixedWeightNMPC

        if method == "bo" and not params_json:
            raise SystemExit("method 'bo' needs params_json (the BO-tuned weights)")
        return FixedWeightNMPC(params_json=params_json or None, **kwargs)
    if method == "ssi":
        from marinelab.third_party.ssi_mpc_gpl.ssi_controller import SSIMPCController

        if params_json:
            kwargs["params_json"] = params_json
        ctl = SSIMPCController(
            step_dt=float(g("step_dt")), ssi_lr=float(g("ssi_lr")),
            ssi_kernel_std=float(g("ssi_kernel_std")), ssi_n_rf=int(g("ssi_n_rf")),
            ssi_seed=int(g("ssi_seed")), **kwargs)
        ctl.name = "ssi"
        return ctl
    raise SystemExit(f"unknown method {method!r} (nominal|bo|ssi)")


class WallScanControllerNode(Node):
    def __init__(self):
        super().__init__("wallscan_controller")
        p = self.declare_parameter
        p("method", "nominal")
        p("plant_json", _default_plant_json())
        p("params_json", "")            # tuned weights (bo/ssi inherit BO cost weights)
        p("ssi_lr", 0.14733286466312384)          # adopted ssi attempt-2 trial 87
        p("ssi_kernel_std", 0.18398034704266503)
        p("ssi_n_rf", 100), p("ssi_seed", 0)
        # horizon 20 / RTI 4 = the validated DEPLOY setting (E4c 2026-08-16): on the
        # Jetson the sim default h30/rti8 takes 37-38 ms vs the 20 ms tick — h20/rti4
        # runs 15.6-16.2 ms (p99 20.7, <=4.4% soft overruns) and is performance-lossless
        # in sim (e5_hwdrag_lat: cycles 2.0, nominal dobj <= +1.1% over 5 seeds).
        p("horizon", 20), p("rti_iters", 4), p("step_dt", 0.02)
        p("code_export_root", os.path.expanduser("~/.cache/wallscan_acados"))
        # Stage3 scan/reference parameters (wallscan_env_cfg) — override per tank.
        p("tank_radius", 6.0), p("d_ref", 1.5)
        p("z_top", 8.5), p("z_bottom", 1.0), p("sway_step", 1.0)
        p("reach_eps", 0.6), p("reach_hold", 10)
        p("ref_step", 0.004), p("ref_step_s", 0.002), p("dt_mpc", 0.05)
        p("stale_zero_s", 0.5)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        plant_json = str(g("plant_json"))
        if not plant_json or not os.path.isfile(plant_json):
            raise SystemExit(
                f"plant_json not found ({plant_json!r}): pass -p plant_json:=... or set "
                "MARINELAB_ROOT so config/pkrc_plant_fixed_tam.json resolves")
        plant = PlantParams.from_json(plant_json)

        scan_cfg = ScanCfg(
            z_top=float(g("z_top")), z_bottom=float(g("z_bottom")),
            sway_step=float(g("sway_step")), reach_eps=float(g("reach_eps")),
            reach_hold=int(g("reach_hold")), ref_step=float(g("ref_step")),
            ref_step_s=float(g("ref_step_s")))
        mpc_cfg = WallScanMPCCfg(
            tank_radius=float(g("tank_radius")), d_ref=float(g("d_ref")),
            z_top=float(g("z_top")), z_bottom=float(g("z_bottom")),
            sway_step=float(g("sway_step")), ref_step=float(g("ref_step")),
            ref_step_s=float(g("ref_step_s")), step_dt=float(g("step_dt")),
            dt_mpc=float(g("dt_mpc")))
        ctl = _build_controller(self, str(g("method")), plant, mpc_cfg,
                                int(g("horizon")), int(g("rti_iters")),
                                str(g("code_export_root")))
        self.loop = WallScanControlLoop(ctl, scan_cfg, mpc_cfg, horizon=int(g("horizon")))
        self.enabled = False
        self.was_enabled = False
        self.s_hat = None
        self.stale_zero = float(g("stale_zero_s"))
        self._last_state_t = None

        self.create_subscription(Odometry, "/wallscan/state", self._on_state, 10)
        self.create_subscription(Float64MultiArray, "/wallscan/estimator_debug",
                                 self._on_debug, 10)
        self.create_subscription(Bool, "/wallscan/enable", self._on_enable, 10)
        self.pub_u = self.create_publisher(Float64MultiArray, "/wallscan/u", 10)
        self.pub_dbg = self.create_publisher(Float64MultiArray, "/wallscan/controller_debug", 10)
        # Watchdog: if states stop arriving entirely, keep asserting zero thrust.
        self.create_timer(0.2, self._watchdog)
        self.get_logger().info(f"wallscan controller up: method={g('method')!r} "
                               f"plant={plant_json} (enable via /wallscan/enable)")

    # -- callbacks -----------------------------------------------------------
    def _on_enable(self, m: Bool) -> None:
        self.enabled = bool(m.data)
        self.get_logger().info(f"/wallscan/enable -> {self.enabled}")

    def _on_debug(self, m: Float64MultiArray) -> None:
        if len(m.data) >= 3:
            self.s_hat = float(m.data[2])

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _zero(self) -> None:
        msg = Float64MultiArray()
        msg.data = [0.0] * 6
        self.pub_u.publish(msg)

    def _watchdog(self) -> None:
        if self._last_state_t is not None and self._now() - self._last_state_t > self.stale_zero:
            self._zero()
            self.get_logger().warning("state stream stale — holding zero thrust",
                                      throttle_duration_sec=5.0)

    def _on_state(self, m: Odometry) -> None:
        self._last_state_t = self._now()
        if self.s_hat is None:
            return  # estimator debug (s_hat) not paired yet
        p, q, tw = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist
        veh = VehicleState(
            pos_w=np.array([p.x, p.y, p.z]),
            quat_wb=np.array([q.w, q.x, q.y, q.z]),
            lin_vel_b=np.array([tw.linear.x, tw.linear.y, tw.linear.z]),
            ang_vel_b=np.array([tw.angular.x, tw.angular.y, tw.angular.z]),
            stamp=self._last_state_t)
        theta_hat = math.atan2(p.y, p.x)

        if not self.enabled:
            self.was_enabled = False
            self._zero()
            self._debug(-1.0, 0.0)
            return
        if not self.was_enabled:  # rising edge: re-anchor the scan at the current depth
            self.loop.reset(veh)
            self.was_enabled = True
            self.get_logger().info(f"scan enabled: anchored at z={p.z:.2f}")

        try:
            out = self.loop.step(veh, s_hat=self.s_hat, theta_hat=theta_hat)
        except Exception as exc:  # never let the loop die with thrusters live
            self.get_logger().error(f"controller step failed: {exc!r} — zero thrust")
            self._zero()
            return
        msg = Float64MultiArray()
        msg.data = [float(v) for v in np.clip(out.u_cmd, -1.0, 1.0)]
        self.pub_u.publish(msg)
        self._debug(out.solve_ms, float(out.status))

    def _debug(self, solve_ms: float, status: float) -> None:
        z_ref, s_ref = self.loop.refs
        dbg = Float64MultiArray()
        dbg.data = [float(self.enabled), float(self.loop.phase), float(self.loop.cycles),
                    z_ref, s_ref, solve_ms, status]
        self.pub_dbg.publish(dbg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WallScanControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
