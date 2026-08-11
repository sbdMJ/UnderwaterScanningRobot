#!/usr/bin/env python3
"""[RETRACTED 2026-08-11 — do not trust the numbers] Heave drag + buoyancy ID attempt.

The analysis below assumes free swimming, but the session had tether handling (the
operator confirmed the vehicle is positively buoyant and that (-,+) drives it DOWN,
which contradicts the free-swimming reading of the hold/ascend segments). Kept only as
a record of the method and the failure mode: a teleop bag cannot anchor a force balance
unless tether slack is known. Use the dedicated protocol in thruster_mapping.md §4c/§4e.

The bag's /teleop/thruster_currents are actual VESC currents (mix_thrusters
applies TAM, gain, max_current AND polarity before publishing). The session
contains three quasi-steady vertical regimes under depth control:

  descend  ~23 s at |I| = 3.1-3.4 A,  v = +0.11..0.15 m/s (depth-rate)
  hold      ~4 s at |I| = 4.4 A,      v = 0
  ascend    ~6 s at |I| = 5.0 A,      v = -0.10..-0.18 m/s (transient, excluded)

Only ONE sign interpretation is dynamically consistent (more current -> more
upward velocity, monotonic): the (I_T5<0, I_T6>0) pattern is UP thrust and the
vehicle is NEGATIVELY buoyant. The hold regime is then a force balance
W_net = T_up(I_hold), and each steady descend point gives drag directly:
D(v) = W_net - T_up(I) = 2k[(I_hold - I) ] (I0 cancels).

Thrust model: T_up(I) = 2 * k_h * (|I| - I0) with the 2026-08-11 bollard-pull
constants (k_h, I0 are the T1-T4 average — heave itself is unmeasured, so
absolute forces carry that ~+-20% scale; the sim-vs-real drag RATIO scales the
same way and stays >~3x regardless).
"""
import numpy as np
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

BAG = "/home/mjkim/Downloads/20260806_122531"
K_H, I0 = 1.674, 0.729          # assumed = T1-T4 bollard average (§4d)
SIM_D1, SIM_D2 = 119.44, 38.51  # sim heave damping (sway-approx, pkrc.py)

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
            cur_t.append(t); cur.append(abs(m.data[4]))  # |I| on the heave pair

depth_t, depth = np.array(depth_t), np.array(depth)
cur_t, cur = np.array(cur_t), np.array(cur)
vz = np.gradient(depth, depth_t)  # + = down (bar10xt depth is positive-down)


def regime(t_a, t_b):
    md = (depth_t >= t_a) & (depth_t <= t_b)
    mc = (cur_t >= t_a) & (cur_t <= t_b)
    return float(vz[md].mean()), float(cur[mc].mean())


def t_up(i):
    return 2.0 * K_H * (i - I0)


v_hold, i_hold = regime(45.0, 48.5)
w_net = t_up(i_hold)
print(f"hold:    |I| = {i_hold:.2f} A, v = {v_hold:+.3f} m/s  ->  W_net = {w_net:.2f} N "
      f"({w_net / 9.81:.2f} kgf NEGATIVE buoyancy)")

pts = []
for (a, b) in [(8, 18), (22, 27)]:
    v, i = regime(a, b)
    d = t_up(i_hold) - t_up(i)   # = 2*K_H*(i_hold - i); I0 cancels
    pts.append((v, d))
    sim = SIM_D1 * v + SIM_D2 * v * v
    print(f"descend t={a}-{b}s: |I| = {i:.2f} A, v = {v:+.3f} m/s  ->  D = {d:.2f} N "
          f"(d_eff {d / v:.1f} N/(m/s); sim would be {sim:.1f} N -> {sim / d:.1f}x)")

v_arr = np.array([p[0] for p in pts]); d_arr = np.array([p[1] for p in pts])
d1 = float(np.dot(d_arr, v_arr) / np.dot(v_arr, v_arr))
print(f"\nlinear fit: d1_heave ≈ {d1:.1f} N/(m/s)   (sim uses {SIM_D1} -> {SIM_D1 / d1:.1f}x too high)")
print(f"ascend-at-0.2 m/s check: needs T_up = W + D = {w_net + d1 * 0.2:.1f} N pair "
      f"-> {(w_net + d1 * 0.2) / (2 * K_H) + I0:.1f} A/thruster (limit 5 A)")
print(f"with neutral ballast:    needs {d1 * 0.2:.1f} N pair "
      f"-> {(d1 * 0.2) / (2 * K_H) + I0:.1f} A/thruster")
