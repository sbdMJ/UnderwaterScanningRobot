#!/usr/bin/env python3
"""§4f drag-ID session, 2026-08-11 mj_ws bags (~0.85 m pool) — extraction + verdict.

Auto-mode current chain (patch 0001): VESC current = cmd x gain x polarity, and the
operator launched with default gains (all 1.0), heave polarity +1 -> heave VESC current
== commanded current exactly. /teleop/thruster_currents is the MANUAL-path mix output
and is meaningless in auto mode — segmentation is done from dynamics + the operator's
stated command order (equilibrium [-3.1,+3.1]; descents [-3.2],[-4],[-5] x2 each;
ascent [+2,-2] hand-pushed to bottom and released).

Findings (see thruster_mapping.md §4f 결과):
- 12 dips = 6 command runs + surface-bob rebounds. Descent terminals (plateaus, two
  runs each): 3.2 A -> ~0.078, 4 A -> ~0.115, 5 A -> ~0.150 m/s. Monotone in current
  -> [-I,+I] = DOWN re-confirmed by dynamics.
- Free rise after cut: ~0.26 m/s -> strongly positive buoyancy; equilibrium at 3.1 A
  -> B ~ 7.9 N (0.81 kgf) on the assumed heave k.
- DECISION: 5 A descent terminal 0.15 < 0.2 m/s scan reference. Quantitative curve
  fitting is unreliable in a 0.3 m column (surface/bottom effects, the free-rise point
  is inconsistent with the descent curve), but the decision number is a direct
  measurement, repeatable across both 5 A runs.
- surge/sway bag unusable: no heading hold in auto mode -> the robot yawed through
  every run (up to 107 deg/s); DVL vy pinned at ~0 throughout (channel or frame
  suspect). Translation drag needs a bigger pool + yaw-balanced pairs.
"""
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

ts = get_typestore(Stores.ROS2_HUMBLE)
from pathlib import Path
from rosbags.typesys import get_types_from_msg
MSGDIR = Path("/home/mjkim/PKRC로봇_코드_및_데이터/hero_ws/src/dvl_msgs/msg")
types = {}
for f in MSGDIR.glob("*.msg"):
    types.update(get_types_from_msg(f.read_text(), f"dvl_msgs/msg/{f.stem}"))
ts.register(types)
BASE = "/home/mjkim/Downloads/mj_ws_bags"


def load(bag):
    out = {"dt": [], "d": [], "vt": [], "vx": [], "vy": [], "vz": [], "valid": []}
    with Reader(f"{BASE}/{bag}") as r:
        t0 = r.start_time * 1e-9
        conns = [c for c in r.connections if c.topic in ("/bar10xt/depth", "/dvl/data")]
        for c, tn, raw in r.messages(connections=conns):
            t = tn * 1e-9 - t0
            m = ts.deserialize_cdr(raw, c.msgtype)
            if c.topic == "/bar10xt/depth":
                out["dt"].append(t); out["d"].append(m.data)
            else:
                out["vt"].append(t)
                out["vx"].append(m.velocity.x); out["vy"].append(m.velocity.y)
                out["vz"].append(m.velocity.z)
                out["valid"].append(getattr(m, "velocity_valid", True))
    return {k: np.array(v) for k, v in out.items()}


def win_slope(t, y, a, b):
    m = (t >= a) & (t < b)
    if m.sum() < 3:
        return np.nan
    A = np.vstack([t[m], np.ones(m.sum())]).T
    return float(np.linalg.lstsq(A, y[m], rcond=None)[0][0])


print("=========== 하강bag: dip-by-dip v(t) trend (0.8 s windows)")
h = load("하강bag")
t, d = h["dt"], h["d"]
v = np.gradient(d, t)
# find dips: contiguous regions where depth > rest+0.03
rest = np.median(d)
above = d > rest + 0.03
edges = np.flatnonzero(np.diff(above.astype(int)))
starts = [e for e in edges if above[e + 1]]
ends = [e for e in edges if not above[e + 1]]
for i, (s, e) in enumerate(zip(starts, ends)):
    ta, tb = t[s] - 1.5, t[e] + 0.5
    prof = []
    x = ta
    while x < tb:
        prof.append((x - t[s], win_slope(t, d, x, x + 0.8)))
        x += 0.4
    peak_v = np.nanmax([p[1] for p in prof])
    peak_d = d[s:e + 2].max()
    print(f"dip{i + 1}: t={t[s]:6.1f}s depth {rest:.2f}->{peak_d:.2f} "
          f"peak v_down={peak_v:+.3f}")
    print("   v(t): " + " ".join(f"{p[1]:+.3f}" if np.isfinite(p[1]) else "  ---" for p in prof))
# free-rise after cut: slope right after each dip peak
print("\nfree-rise (after cut, buoyancy-only): per-dip max rise speed")
for i, (s, e) in enumerate(zip(starts, ends)):
    idx = np.argmax(d[s:e + 2]) + s
    vr = [win_slope(t, d, t[idx] + dt0, t[idx] + dt0 + 0.8) for dt0 in (0.2, 0.8, 1.4)]
    print(f"dip{i + 1}: rise v = " + " ".join(f"{x:+.3f}" for x in vr if np.isfinite(x)))

print("\n=========== 상승bag: depth profile")
u = load("상승bag")
t, d = u["dt"], u["d"]
for a in np.arange(0, t[-1], 1.0):
    m = (t >= a) & (t < a + 1)
    if m.any():
        print(f"{a:5.1f}s depth={d[m].mean():6.3f} v={win_slope(t, d, a, a + 1):+7.4f}")

print("\n=========== surge_sway_bag: DVL body velocities (1 s means)")
g = load("surge_sway_bag")
vt = g["vt"]
for a in np.arange(0, vt[-1], 2.0):
    m = (vt >= a) & (vt < a + 2)
    if m.sum() >= 2:
        vx, vy, vz = g["vx"][m].mean(), g["vy"][m].mean(), g["vz"][m].mean()
        if abs(vx) > 0.02 or abs(vy) > 0.02:
            print(f"{a:6.1f}s vx={vx:+.3f} vy={vy:+.3f} vz={vz:+.3f} (n={m.sum()})")
