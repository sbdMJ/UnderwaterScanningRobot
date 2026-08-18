#!/usr/bin/env python3
"""Rigorous heave-drag UPPER BOUND from the 122531 bag, tether-robust.

Axioms (operator-verified facts, 2026-08-11): the vehicle is positively buoyant
(B >= 0 up) and the (I_T5<0, I_T6>0) current pattern is DOWN thrust. The session
had tether handling, but a surface tether can only pull UP (tension R >= 0), so
during any steady DESCENT second the vertical balance

    T_down(I) = D(v) + B + R        (all of D, B, R >= 0)

gives  D(v) <= T_down(I) = 2*k_h*(|I| - I0)  regardless of the unknown R and B.
That inequality is enough to test the sim's heave damping (119.44 N/(m/s) linear
+ 38.51 quadratic, itself a sway-approx): if sim drag at the measured speed
exceeds the bound, the sim coefficient is PROVABLY too high by at least that
ratio. Caveats: k_h, I0 are the T1-T4 bollard average (heave unmeasured, ~+-20%
scale on the bound); the bound only speaks at the measured speeds (~0.10-0.15
m/s) — it does NOT bound drag at 0.2 m/s if the true curve is strongly quadratic.
"""
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

BAG = "/home/mjkim/Downloads/20260806_122531"
K_H, I0 = 1.674, 0.729
SIM_D1, SIM_D2 = 119.44, 38.51

ts = get_typestore(Stores.ROS2_HUMBLE)
depth_t, depth, cur_t, cur = [], [], [], []
with Reader(BAG) as r:
    conns = [c for c in r.connections if c.topic in ("/bar10xt/depth", "/teleop/thruster_currents")]
    t0 = r.start_time * 1e-9
    for conn, t_ns, raw in r.messages(connections=conns):
        t = t_ns * 1e-9 - t0
        m = ts.deserialize_cdr(raw, conn.msgtype)
        if conn.topic == "/bar10xt/depth":
            depth_t.append(t); depth.append(m.data)
        else:
            cur_t.append(t); cur.append(abs(m.data[4]))

depth_t, depth = np.array(depth_t), np.array(depth)
cur_t, cur = np.array(cur_t), np.array(cur)
vz = np.gradient(depth, depth_t)  # + = down

print(f"{'t[s]':>5} {'v_down':>7} {'|I|':>5} {'T_down=bound':>13} {'sim D(v)':>9} {'sim/bound >=':>12}")
worst = 0.0
for sec in range(5, 28):
    md = (depth_t >= sec) & (depth_t < sec + 1)
    mc = (cur_t >= sec) & (cur_t < sec + 1)
    if not md.any() or not mc.any():
        continue
    v = float(vz[md].mean())
    dv = float(vz[md].max() - vz[md].min())
    if v < 0.09 or dv > 0.06:  # steady, clearly-descending seconds only
        continue
    i = float(cur[mc].mean())
    bound = 2 * K_H * (i - I0)
    sim = SIM_D1 * v + SIM_D2 * v * v
    ratio = sim / bound
    worst = max(worst, ratio)
    print(f"{sec:5d} {v:7.3f} {i:5.2f} {bound:13.2f} {sim:9.2f} {ratio:12.2f}")

print(f"\nguaranteed: sim heave drag is >= {worst:.1f}x too high at ~0.10-0.15 m/s")
for dk in (0.8, 1.2):  # k_h uncertainty band
    print(f"  with k_h x{dk:.1f}: factor scales to >= {worst / dk:.1f}x")
