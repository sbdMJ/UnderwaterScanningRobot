#!/usr/bin/env python3
# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Replay a hero_ws rosbag through TopicSampleAssembler + WallFrameStateEstimator.

Plumbing validation for the hardware estimator bridge, runnable on any machine with the
``rosbags`` package (no ROS install): the exact code path the rclpy node drives —
callbacks -> assembler -> 50 Hz assemble -> estimator.step — fed from recorded topics.

HONEST SCOPE: the 2026-08-06 bags were recorded in a test pool, not the R = 6 m cylindrical
tank the wall-frame EKF models, so (r, phi) against the sonar model are not meaningful
accuracy numbers here. What this validates is the plumbing the e5_ekf campaign showed is
load-bearing: every channel arrives at its measured rate, wall/ukfm are consumed once, the
fix carries its absolute bearing, and the filter steps 50 Hz without blowing up.

    python hw_bag_replay_estimator.py <bag_dir> [--tank-height 10] [--tank-radius 6]
"""
from __future__ import annotations

import argparse
import math
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _install_marinelab_shim() -> None:
    """Import marinelab.control without the heavy package __init__ (no isaaclab here).

    Same trick as tests/conftest.py and the pkrc_wallscan_bridge ROS node: register bare
    package modules with a real __path__ so submodule imports resolve from disk while the
    env-registering __init__ never runs.
    """
    for name, sub in (("marinelab", ""), ("marinelab.tasks", "tasks")):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(REPO / "marinelab" / sub)] if sub else [str(REPO / "marinelab")]
            sys.modules[name] = pkg


_install_marinelab_shim()

from marinelab.control.estimator import WallFrameStateEstimator  # noqa: E402
from marinelab.control.hw_bridge import TankCalib, TopicSampleAssembler  # noqa: E402
from marinelab.tasks.pkrc_wallscan.wall_frame_ekf import WallFrameEKFCfg  # noqa: E402


def _yaw_rp(q) -> tuple[float, float, float]:
    """(yaw, roll, pitch) from a geometry_msgs quaternion (x, y, z, w fields)."""
    sinr = 2 * (q.w * q.x + q.y * q.z)
    cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
    sinp = max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))
    siny = 2 * (q.w * q.z + q.x * q.y)
    cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy), math.atan2(sinr, cosr), math.asin(sinp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="rosbag2 directory (with metadata.yaml)")
    ap.add_argument("--tank-height", type=float, default=10.0)
    ap.add_argument("--tank-radius", type=float, default=6.0)
    ap.add_argument("--wall-topic", default="/ukfm/wall_distance")
    args = ap.parse_args()

    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    msgdir = Path.home() / "PKRC로봇_코드_및_데이터/hero_ws/src/dvl_msgs/msg"
    if msgdir.is_dir():
        custom = {}
        for f in msgdir.glob("*.msg"):
            custom.update(get_types_from_msg(f.read_text(), f"dvl_msgs/msg/{f.stem}"))
        ts.register(custom)

    calib = TankCalib(tank_height=args.tank_height, tank_radius=args.tank_radius)
    asm = TopicSampleAssembler(calib)
    est = WallFrameStateEstimator(WallFrameEKFCfg(tank_radius=args.tank_radius))

    topics = {"/dvl/data", "/imu/data", "/bar10xt/depth", "/ukfm/odom_validated",
              args.wall_topic}
    events: list[tuple[float, str, object]] = []
    with Reader(args.bag) as r:
        conns = [c for c in r.connections if c.topic in topics]
        for conn, t_ns, raw in r.messages(connections=conns):
            events.append((t_ns * 1e-9, conn.topic, ts.deserialize_cdr(raw, conn.msgtype)))
    events.sort(key=lambda e: e[0])
    if not events:
        raise SystemExit("no matching topics in this bag")

    t0, t_end = events[0][0], events[-1][0]
    dt, tick = 0.02, events[0][0]
    n_samples = n_sonar = n_fix = 0
    trace = {"r": [], "phi": [], "s": [], "s_bearing": []}
    i = 0
    initialized = False
    while tick <= t_end:
        while i < len(events) and events[i][0] <= tick:
            t, topic, m = events[i]
            i += 1
            if topic == "/dvl/data":
                v = m.velocity
                asm.feed_dvl(v.x, v.y, v.z, t)
            elif topic == "/imu/data":
                yaw, roll, pitch = _yaw_rp(m.orientation)
                w = m.angular_velocity
                asm.feed_imu(w.x, w.y, w.z, roll, pitch, t)
            elif topic == "/bar10xt/depth":
                asm.feed_depth(m.data, t)
            elif topic == "/ukfm/odom_validated":
                p = m.pose.pose.position
                yaw, _, _ = _yaw_rp(m.pose.pose.orientation)
                asm.feed_ukfm(p.x, p.y, yaw, t)
            elif topic == args.wall_topic:
                asm.feed_wall_range(float(m.data), t)

        sample = asm.assemble(tick)
        if sample is not None:
            if not initialized and sample.ukfm is not None:
                r0, phi0, theta0 = sample.ukfm
                est.reset(r0=r0, phi0=phi0, theta0=theta0)
                initialized = True
            if initialized:
                n_sonar += sample.sonar is not None
                if sample.ukfm is not None:
                    n_fix += 1
                    trace["s_bearing"].append(
                        est.s_hat + args.tank_radius * math.atan2(
                            math.sin(sample.ukfm[2] - est.theta_hat),
                            math.cos(sample.ukfm[2] - est.theta_hat)))
                est.step(sample, dt)
                n_samples += 1
                trace["r"].append(est.ekf.r)
                trace["phi"].append(est.ekf.phi)
                trace["s"].append(est.s_hat)
        tick += dt

    span = t_end - t0
    print(f"bag span {span:.1f} s | estimator ticks {n_samples} "
          f"({n_samples / max(span, 1e-9):.1f} Hz effective)")
    print(f"sonar corrections {n_sonar} ({n_sonar / max(span, 1e-9):.1f} Hz), "
          f"marker fixes {n_fix} ({n_fix / max(span, 1e-9):.2f} Hz)")
    for k in ("r", "phi", "s"):
        v = np.asarray(trace[k])
        print(f"  {k:>3}: min {v.min():+.3f}  max {v.max():+.3f}  final {v[-1]:+.3f}")
    if trace["s_bearing"]:
        gap = np.abs(np.asarray(trace["s_bearing"]) - np.interp(
            np.linspace(0, 1, len(trace["s_bearing"])),
            np.linspace(0, 1, len(trace["s"])), np.asarray(trace["s"])))
        print(f"  |s - bearing-implied s| at fixes: median {np.median(gap):.3f} m, "
              f"p90 {np.percentile(gap, 90):.3f} m")
    finite = all(np.isfinite(np.asarray(trace[k])).all() for k in ("r", "phi", "s"))
    print("PLUMBING", "OK" if (initialized and finite and n_fix > 0 and n_sonar > 0) else "FAIL",
          f"(initialized={initialized}, finite={finite})")


if __name__ == "__main__":
    main()
