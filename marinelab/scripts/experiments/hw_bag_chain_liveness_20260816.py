#!/usr/bin/env python3
"""Scenario-2 chain-liveness bags (2026-08-16, small acrylic pool, marker-less) — PASS.

Two bags: stationary (112 s) and hand-moved (45 s), full node chain up with
enable OFF (blind anchor r0=4.5, wall source = ping1d, no marker -> no ukfm).

- End-to-end 50 Hz: /wallscan/state 49.9-50.0 Hz, u/current_cmd/debug all in
  lockstep; u == 0 on every tick while disabled; controller status 0 for 100%
  of ticks. (solve_ms is -1 while disabled — the node skips the solver; timing
  evidence stays with bench_inference E4c.)
- Depth/attitude estimation is REAL and tracks: state z == 0.85 - bar depth to
  the cm through hand-lifts (0.11..0.31 m), yaw follows ~160 deg rotations.
- Robustness observed: the DVL driver stalled up to 17 s (acrylic, valid only
  41-47%, |v| garbage to 0.19 m/s) and the estimator rate-gate coasted through
  it without dropping a tick.
- As predicted for marker-less blind anchor: x/y/s_hat are dead-reckoned
  fiction (~1-2 cm/s wander while physically stationary) — scenario 3 requires
  the marker fix.
- Ping1D in the small pool multipaths (readings 0.79..5.26 m) — feeding it to
  the EKF wall channel is fine for liveness but supports scenario 3's choice
  to run wall-less (dead default topic).
"""
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path
from rosbags.typesys import get_types_from_msg

ts = get_typestore(Stores.ROS2_HUMBLE)
MSGDIR = Path("/home/mjkim/PKRC로봇_코드_및_데이터/hero_ws/src/dvl_msgs/msg")
types = {}
for f in MSGDIR.glob("*.msg"):
    types.update(get_types_from_msg(f.read_text(), f"dvl_msgs/msg/{f.stem}"))
ts.register(types)


def load(bag):
    o = {"st": [], "sx": [], "sy": [], "sz": [], "syaw": [], "svz": [],
         "et": [], "edbg": [], "ct": [], "cdbg": [], "ut": [], "u": [],
         "it": [], "cur": [], "dt": [], "dep": [], "vt": [], "dvl": [], "rt": [], "rng": []}
    with Reader(bag) as r:
        t0 = r.start_time * 1e-9
        for c, tn, raw in r.messages():
            t = tn * 1e-9 - t0
            m = ts.deserialize_cdr(raw, c.msgtype)
            if c.topic == "/wallscan/state":
                p, q = m.pose.pose.position, m.pose.pose.orientation
                yaw = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                o["st"].append(t); o["sx"].append(p.x); o["sy"].append(p.y)
                o["sz"].append(p.z); o["syaw"].append(yaw)
                o["svz"].append(m.twist.twist.linear.z)
            elif c.topic == "/wallscan/estimator_debug":
                o["et"].append(t); o["edbg"].append(list(m.data))
            elif c.topic == "/wallscan/controller_debug":
                o["ct"].append(t); o["cdbg"].append(list(m.data))
            elif c.topic == "/wallscan/u":
                o["ut"].append(t); o["u"].append(list(m.data))
            elif c.topic == "/teleop/thruster_currents":
                o["it"].append(t); o["cur"].append(list(m.data))
            elif c.topic == "/bar10xt/depth":
                o["dt"].append(t); o["dep"].append(m.data)
            elif c.topic == "/dvl/data":
                o["vt"].append(t)
                o["dvl"].append((m.velocity.x, m.velocity.y, m.velocity.z,
                                 getattr(m, "velocity_valid", True), m.altitude))
            elif c.topic == "/sensor/sonar/ping1d/range":
                o["rt"].append(t); o["rng"].append(m.range)
    return {k: np.array(v) for k, v in o.items()}


for bag, name in (("/home/mjkim/Downloads/rosbag2_2026_08_16-18_5/rosbag2_2026_08_16-18_57_09", "정지"),
                  ("/home/mjkim/Downloads/rosbag2_2026_08_16-18_5/rosbag2_2026_08_16-18_59_54", "손이동")):
    d = load(bag)
    print(f"\n######## {name}  dur={d['st'][-1]:.0f}s  state rate={len(d['st'])/d['st'][-1]:.1f} Hz")
    e = d["edbg"]
    print(f"est: r {e[:,0].min():.2f}..{e[:,0].max():.2f}  phi {e[:,1].min():.3f}..{e[:,1].max():.3f}  "
          f"s_hat {e[:,2].min():.2f}..{e[:,2].max():.2f}")
    print(f"ages max: dvl {e[:,4].max():.2f} imu {e[:,5].max():.2f} depth {e[:,6].max():.2f} "
          f"wall {e[:,7].max():.2f} ukfm {e[:,8].max():.2f}")
    c = d["cdbg"]  # [enabled, phase, cycles, z_ref, s_ref, solve_ms, status]
    print(f"ctl: enabled max={c[:,0].max():.0f}  solve_ms mean={c[:,5].mean():.1f} "
          f"p99={np.percentile(c[:,5],99):.1f} max={c[:,5].max():.1f}  "
          f"status!=0: {(c[:,6]!=0).mean()*100:.1f}%")
    u = d["u"]
    print(f"u: |u|max per ch = " + " ".join(f"{np.abs(u[:,i]).max():.2f}" for i in range(6)))
    cur = d["cur"]
    print(f"teleop currents |max| = {np.abs(cur).max():.2f} A")
    print(f"state z {min(d['sz']):.2f}..{max(d['sz']):.2f}  yaw {np.degrees(min(d['syaw'])):.0f}..{np.degrees(max(d['syaw'])):.0f} deg")
    print(f"x {min(d['sx']):.2f}..{max(d['sx']):.2f}  y {min(d['sy']):.2f}..{max(d['sy']):.2f}")
    dv = d["dvl"]
    print(f"dvl: valid {np.mean([float(v[3]) for v in dv])*100:.0f}%  |vx|max {max(abs(v[0]) for v in dv):.3f} "
          f"|vy|max {max(abs(v[1]) for v in dv):.3f}  alt {min(v[4] for v in dv):.2f}..{max(v[4] for v in dv):.2f}")
    print(f"ping1d: {d['rng'].min():.2f}..{d['rng'].max():.2f} m")
    # depth vs state z consistency
    print(f"bar depth {d['dep'].min():.2f}..{d['dep'].max():.2f} (tank z = 0.85 - depth 기대)")
