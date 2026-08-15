#!/usr/bin/env python3
"""§4g trim session bags, 2026-08-15 — heave k confirmation + 5 A descent bound.

Two bags after fitting the 0.5 kg lead (1 kg over-sank; §4g adopted plan revised):

`rosbag2_2026_08_15-17_14_05=` — INVALID for descent. /wallscan/current_cmd held
[-5,+5] for 42 s yet depth oscillated +-0.1 m around the 0.6 m manual depth-hold
target (PID error visible in /teleop/depth_debug): teleop was in MANUAL mode the
whole bag, the command never reached the VESCs.

`rosbag2_2026_08_15-18_00_17` — the decisive one:
- k CONFIRMED: with auto engaged at [-0.85,+0.85] the robot free-floated at
  0.72-0.73 m for 27 s (the same bag later reaches 0.801 m -> not resting on the
  floor). Equilibrium shift 3.1 -> 0.85 A under a 0.5 kg lead (4.47 N in water)
  gives k_T5/T6 = 4.47 / (2 x 2.25) ~ 0.99 N/A by the deadzone-cancelling
  difference — 59% of the T1-T4-average assumption (1.674).
- AUTO-DROP ROOT CAUSE: a 21 s gap in /wallscan/current_cmd (t=38.2 -> 59.2,
  operator restarting the pub with the new value) trips patch 0001's 0.5 s stale
  fallback; the wound-up manual depth-hold integrator then yanks the robot up at
  0.1 m/s. Procedure fix: never let the pub die, or re-press 'g' after restart.
- 5 A DESCENT (trimmed): only ~1.7 s of true auto (t~62.5-64.2) before another
  drop; v reached 0.137 m/s and was still rising -> terminal LOWER BOUND ~0.14.
  Initial acceleration ~0.2 m/s^2 with F_net ~ 8.2 N gives m_eff ~ 30-40 kg,
  tau ~ 1.2-1.6 s, extrapolated terminal 0.22-0.33 m/s (matches prediction).
- A 0.85 m pool CANNOT directly demonstrate >=0.2 m/s (needs ~0.6 m of travel to
  reach terminal); certification moves to the main tank or the model verdict.

In-auto /teleop/thruster_currents is the manual-path would-be output (integrator
wound to ~2 A while actual was 0.85 A) — cross-evidence that auto was engaged.
"""
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

ts = get_typestore(Stores.ROS2_HUMBLE)
BASE = "/home/mjkim/Downloads"


def load(bag):
    out = {"dt": [], "d": [], "ct": [], "cmd": [], "tt": [], "tc": [], "gt": [], "tgt": []}
    with Reader(f"{BASE}/{bag}") as r:
        t0 = r.start_time * 1e-9
        for c, tn, raw in r.messages():
            t = tn * 1e-9 - t0
            if c.topic == "/bar10xt/depth":
                m = ts.deserialize_cdr(raw, c.msgtype)
                out["dt"].append(t); out["d"].append(m.data)
            elif c.topic == "/wallscan/current_cmd":
                m = ts.deserialize_cdr(raw, c.msgtype)
                out["ct"].append(t); out["cmd"].append(m.data[4])
            elif c.topic == "/teleop/thruster_currents":
                m = ts.deserialize_cdr(raw, c.msgtype)
                out["tt"].append(t); out["tc"].append(m.data[4])
            elif c.topic == "/teleop/depth_debug":
                m = ts.deserialize_cdr(raw, c.msgtype)
                out["gt"].append(t); out["tgt"].append(m.data[0])
    return {k: np.array(v) for k, v in out.items()}


def win_slope(t, y, a, b):
    m = (t >= a) & (t < b)
    if m.sum() < 3:
        return np.nan
    A = np.vstack([t[m], np.ones(m.sum())]).T
    return float(np.linalg.lstsq(A, y[m], rcond=None)[0][0])


for bag in ("rosbag2_2026_08_15-17_14_05=", "rosbag2_2026_08_15-18_00_17"):
    b = load(bag)
    t, d = b["dt"], b["d"]
    print(f"\n######## {bag}  dur={t[-1]:.0f}s  depth {d.min():.3f}..{d.max():.3f}")
    if b["ct"].size:
        gaps = np.diff(b["ct"])
        for i in np.flatnonzero(gaps > 0.4):
            print(f"cmd GAP {gaps[i]:.2f}s at t={b['ct'][i]:.1f} -> stale fallback (patch 0001)")
        sw = np.flatnonzero(np.abs(np.diff(b["cmd"])) > 0.1)
        print("cmd T5 levels:", sorted(set(np.round(b["cmd"], 2))))
    if b["gt"].size:
        print(f"manual depth-hold target: {np.median(b['tgt']):.3f} m (constant={np.ptp(b['tgt']) < 0.02})")
    print("v(t) 0.8 s windows (down = +):")
    x = 0.0
    rows = []
    while x < t[-1]:
        rows.append((x, win_slope(t, d, x, x + 0.8)))
        x += 0.4
    print(" ".join(f"{v:+.2f}" if np.isfinite(v) else "----" for _, v in rows))
