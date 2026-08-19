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
            ssi_seed=int(g("ssi_seed")), ssi_d_max=float(g("ssi_d_max")),
            ssi_d_tau=float(g("ssi_d_tau")), **kwargs)
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
        # |d_world| clamp (N/axis): the true residual is O(0.5 N) — anything near the
        # thrust authority is a learning artifact (bag 00_33: 10-12 N ghost) and must
        # not reach the OCP. The learner also pairs its regression with the command
        # from command_latency_s ago (same dead time the x0 predictor compensates).
        p("ssi_d_max", 5.0)
        # Injection low-pass tau (s) — the stability half of the bag-00_33 fix: the
        # learner is a parallel feedback path, and under the 0.4 s dead time its
        # tick-rate injection is unstable EVEN with aligned pairs (matched replay:
        # 20 cm limit cycle); filtering d_world well below 1/dead-time restores a
        # 1.3 cm hold while the quasi-DC residual still converges. 0 disables (sim).
        p("ssi_d_tau", 3.0)
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
        # Marker-less depth-hold (small pool): zero the horizontal error weights so the
        # MPC cannot trade depth for the FICTIONAL wall/s errors. Field finding
        # 2026-08-18: with blind-anchor drift (~0.8 m fake wall error) the optimizer
        # exploited the 21 deg roll to squeeze lateral force out of the HEAVE pair —
        # full-down u with zero depth error, robot pinned to the floor. Physically
        # neutering the horizontal thrusters (amps_limit) is not enough: the model
        # doesn't know they are neutered; the objective must match.
        p("depth_only", False)
        # Station-keeping: hold_z >= 0 pins the reference to that height and BYPASSES the
        # scan phase machine. Field finding 2026-08-18 (bag 03_41_32): with
        # z_top == z_bottom == Z_HOLD the reach conditions are trivially true, the phase
        # machine wrapped 49x in 34 s and every SWAY entry re-latched z_hold at the
        # CURRENT depth — z_ref bounced 0.13<->0.17 and the depth loop limit-cycled at
        # +-7.5 cm. For any pure depth-hold trial set hold_z (and depth_only when
        # marker-less); leave at -1 for real scanning.
        p("hold_z", -1.0)
        # Refuse enable when |current z - hold_z| exceeds this (m); 0 disables. See the
        # HOLD-Z SANITY comment at the enable edge (field 2026-08-19: stale Z_HOLD from a
        # previous session sat below the tank floor -> cap thrust into the floor).
        p("hold_z_sanity_m", 0.3)
        # Actuator-rate model overrides (mpc_controller module docstring). All-zero =
        # keep the plant JSON's values (pkrc_plant_hw2026.json ships force_rate_limit =
        # newton_per_amp * 17 A/s, the conservative end of the measured teleop ramp).
        # thrust_limits: per-thruster |F| cap in N — set to the session's realizable
        # force k*(amps_limit - I0) when thrusters are operationally clamped, e.g. the
        # marker-less depth-hold scenario: "[0.0,0.0,0.0,0.0,2.25,2.25]".
        p("force_rate_limit", [0.0] * 6)
        p("thrust_limits", [0.0] * 6)
        # Round-trip dead-time predictor (mpc_controller docstring). -1 = keep the plant
        # JSON's value (hw2026 ships 0.4 s, identified from bag 2026-08-19 23_03_29);
        # 0 disables. Over-prediction is benign — do not zero this to "simplify" a trial.
        p("command_latency_s", -1.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        plant_json = str(g("plant_json"))
        if not plant_json or not os.path.isfile(plant_json):
            raise SystemExit(
                f"plant_json not found ({plant_json!r}): pass -p plant_json:=... or set "
                "MARINELAB_ROOT so config/pkrc_plant_fixed_tam.json resolves")
        plant = PlantParams.from_json(plant_json)
        frl = [float(v) for v in g("force_rate_limit")]
        if any(v > 0.0 for v in frl):
            plant.force_rate_limit = tuple(frl)
        tl = [float(v) for v in g("thrust_limits")]
        if any(v > 0.0 for v in tl):
            plant.thrust_limits = tuple(tl)
        if float(g("command_latency_s")) >= 0.0:
            plant.command_latency_s = float(g("command_latency_s"))
        if plant.force_rate_limit is not None:
            self.get_logger().warning(
                f"ACTUATOR-RATE model: force slew {[round(v, 1) for v in plant.force_rate_limit]} N/s, "
                f"|F| limits {[round(v, 2) for v in (plant.thrust_limits or (plant.max_thrust,) * 6)]} N "
                "— nx 13+6, first start regenerates the acados C code")
        if plant.command_latency_s > 0.0:
            self.get_logger().warning(
                f"LATENCY PREDICTOR: rolling state forward {plant.command_latency_s:.2f} s "
                "through the in-flight commands before each solve (bag 23_03_29 fix)")

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
        self.get_logger().info(
            f"building acados solver: method={g('method')!r} horizon={g('horizon')} "
            f"rti={g('rti_iters')} — FIRST run generates+compiles C code into "
            f"{g('code_export_root')} (minutes on the Jetson; stuck >5 min -> delete "
            "that dir and restart)")
        ctl = _build_controller(self, str(g("method")), plant, mpc_cfg,
                                int(g("horizon")), int(g("rti_iters")),
                                str(g("code_export_root")))
        if bool(g("depth_only")):
            from marinelab.tasks.pkrc_wallscan.mpc_reference import ERROR_NAMES, NE

            w = np.asarray(ctl.weights, float).copy()
            zeroed = ("radial", "s", "v_rad", "v_tan", "head_x", "head_y")
            for i, name in enumerate(ERROR_NAMES):
                if name in zeroed:
                    w[i] = 0.0
            ctl.set_weights(w[:NE], w[NE:])
            self.get_logger().warning(
                f"DEPTH-ONLY mode: zeroed werr for {zeroed} — the controller regulates "
                "z + attitude only; horizontal commands are cost-free noise (keep the "
                "horizontal amps_limit at the deadzone!)")
        hold_z = float(g("hold_z"))
        self.hold_z_sanity = float(g("hold_z_sanity_m"))
        if hold_z >= 0.0:
            self.get_logger().warning(
                f"DEPTH-HOLD mode: phase machine bypassed, z_ref -> {hold_z:.3f} m "
                "(constant), s_ref frozen at enable — no scanning will happen")
        self.loop = WallScanControlLoop(ctl, scan_cfg, mpc_cfg, horizon=int(g("horizon")),
                                        hold_z=hold_z if hold_z >= 0.0 else None)
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
            # HOLD-Z SANITY (field 2026-08-19, bag 22_36_50): the estimator's z frame is
            # re-anchored per session (bar10xt offset drifts), so a Z_HOLD reused from a
            # previous session can sit BELOW THE TANK FLOOR — the target was 0.85 m off,
            # unreachable, and the controller held cap thrust into the floor. Refuse to
            # enable when the hold target is implausibly far from the CURRENT depth;
            # hold_z_sanity_m:=0 disables the guard (big-tank long descents).
            if (self.loop.hold_z is not None and self.hold_z_sanity > 0.0
                    and abs(float(p.z) - self.loop.hold_z) > self.hold_z_sanity):
                self.enabled = False
                self._zero()
                self.get_logger().error(
                    f"HOLD-Z SANITY: refusing enable — hold_z {self.loop.hold_z:.3f} is "
                    f"{abs(float(p.z) - self.loop.hold_z):.2f} m from current z {p.z:.3f} "
                    f"(> {self.hold_z_sanity:.2f}). Re-read Z_HOLD from /wallscan/state "
                    "THIS session (the z frame moves with the bar10xt offset), or raise "
                    "hold_z_sanity_m if the jump is intentional.")
                return
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
        try:
            self._debug(out.solve_ms, float(out.status), aux=out.aux)
        except Exception as exc:  # diagnostics must never take the control loop down
            self.get_logger().error(f"debug publish failed: {exc!r}",
                                    throttle_duration_sec=5.0)

    def _debug(self, solve_ms: float, status: float, aux: dict | None = None) -> None:
        z_ref, s_ref = self.loop.refs
        data = [float(self.enabled), float(self.loop.phase), float(self.loop.cycles),
                float(z_ref), float(s_ref), float(solve_ms), float(status)]
        # SSI diagnostics, appended (existing indices unchanged): the bag must be able to
        # judge the learner — d_world (N, world) the residual injects, and the one-step
        # prediction error norm. Absent for nominal/bo (7-element message as before).
        # NB: build the plain list FIRST — Float64MultiArray.data is an array.array and
        # rejects `+= list` (field-crashed 2026-08-19 23:59, first ssi enable tick).
        if aux and "ssi_residual_b" in aux:
            d = getattr(self.loop.ctl, "_d_world", None)
            data += [float(v) for v in d] if d is not None else [0.0, 0.0, 0.0]
            pe = aux.get("ssi_pred_err")
            data.append(float(pe) if pe is not None and np.isfinite(pe) else 0.0)
        dbg = Float64MultiArray()
        dbg.data = data
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
