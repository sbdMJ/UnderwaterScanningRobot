#!/usr/bin/env python3
"""Depth-hold retry bag 2026-08-19 22:36:50 — first field run of the ACTUATOR-RATE model.
VERDICT: aborted on floor contact, but NOT a model failure — a stale Z_HOLD put the
target below the tank floor. The new stack itself behaved exactly as designed.

What the rate model delivered, field-verified in this bag:
- slew signature: |u4| per-tick delta p99 0.0011, max 0.0915 = EXACTLY the modelled
  bound (16.83 N/s * 0.02 s / 3.68 N) — the nx-19 model is live and the published
  force ramps instead of jumping. ZERO bang-bang (04_15: sign flip every ~0.85 s).
- thrust_limits live: u4/u5 saturate at -0.61 = -2.25/3.68 (the session heave cap),
  u0..u3 identically 0.00 the whole bag (horizontal caps at 0 — no recruitment,
  the 04_15 cost-free-noise channel is gone).
- phase pinned 0, z_ref one clean ref_step slew 0.982 -> 0.130, solver status 0 100%.

ROOT CAUSE of the floor contact — STALE Z_HOLD (operator trap, 3rd frame-related
incident): the estimator re-anchors its z frame each session and the bar10xt offset
drifts, so state z = 0.85 - bar TODAY (float 0.98, floor ~0.73) vs the previous
session's frame (float ~0.25, floor 0.016). Z_HOLD=0.13 was reused from 08-18 —
in today's frame that is ~0.6 m BELOW THE FLOOR. The controller did the only thing
consistent with its objective: slewed z_ref to an unreachable target and held the
heave cap (2.25 N) into the floor for the remaining ~15 s. The runbook's "read
Z_HOLD from /wallscan/state THIS session" step was skipped.

Fix committed with this script: wallscan_controller now REFUSES the enable rising
edge when |current z - hold_z| > hold_z_sanity_m (default 0.3 m; 0 disables) with a
HOLD-Z SANITY error log — a stale target can no longer reach the thrusters.

Secondary finding to track: solve time 25.8 ms mean on the Jetson (nx 13->19 cost;
was 15.6-16.2 ms at h20/rti4) — over the 20 ms tick, i.e. the loop effectively runs
~39 Hz. Harmless against the 0.5 s teleop stale threshold and the rate model is
robust to tick stretch (the f_act bookkeeping under-integrates, which is the
conservative direction), but if it grows, bench h15/rti4 or rti_iters:=3 on the
Jetson (E4(c) protocol) before the next scenario.
"""
import numpy as np
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

ts = get_typestore(Stores.ROS2_HUMBLE)
MSGDIR = Path("/home/mjkim/PKRC로봇_코드_및_데이터/hero_ws/src/dvl_msgs/msg")
types = {}
for f in MSGDIR.glob("*.msg"):
    types.update(get_types_from_msg(f.read_text(), f"dvl_msgs/msg/{f.stem}"))
ts.register(types)
BAG = "/home/mjkim/Downloads/rosbag2_2026_08_19-22_36_50"

o = {"st": [], "sz": [], "ct": [], "c": [], "ut": [], "u": [], "dt": [], "dep": []}
with Reader(BAG) as r:
    t0 = r.start_time * 1e-9
    for c, tn, raw in r.messages():
        t = tn * 1e-9 - t0
        m = ts.deserialize_cdr(raw, c.msgtype)
        if c.topic == "/wallscan/state":
            o["st"].append(t); o["sz"].append(m.pose.pose.position.z)
        elif c.topic == "/wallscan/controller_debug":
            o["ct"].append(t); o["c"].append(list(m.data))
        elif c.topic == "/wallscan/u":
            o["ut"].append(t); o["u"].append(list(m.data))
        elif c.topic == "/bar10xt/depth":
            o["dt"].append(t); o["dep"].append(m.data)
o = {k: np.array(v) for k, v in o.items()}

c, u = o["c"], o["u"]
en = c[:, 0] > 0
print(f"dur {o['st'][-1]:.0f}s  enabled from t={o['ct'][en][0]:.1f}s  "
      f"solve {c[:,5][en].mean():.1f} ms (deploy tick 20 ms!)  "
      f"status!=0 {(c[:,6][en]!=0).mean()*100:.1f}%")
print(f"phase uniq {sorted(set(c[en,1]))}  z_ref {c[en,3][0]:.3f} -> {c[en,3][-1]:.3f} (one slew)")

# the z-frame drift that caused the incident: state z vs bar10xt offset
zi = np.interp(o["dt"], o["st"], o["sz"])
off = zi + o["dep"]
print(f"state z = C - bar10xt with C = {off.mean():.3f} +- {off.std():.3f} "
      f"(prev session C ~ 0.5: float z 0.25 / floor 0.016 -> today float 0.98 / floor 0.73)")
print(f"Z_HOLD used: {c[en,3][-1]:.3f} -> {0.73 - c[en,3][-1]:.2f} m BELOW today's floor")

# rate-model signatures
du = np.abs(np.diff(u[:, 4]))
print(f"u4 per-tick |delta| p99 {np.percentile(du,99):.4f} max {du.max():.4f} "
      f"(rate bound 16.83*0.02/3.68 = 0.0915)")
print(f"u caps: u4/u5 min {u[:,4].min():.2f}/{u[:,5].min():.2f} (thrust_limits -2.25/3.68 = -0.61)  "
      f"|u0..u3| max {np.abs(u[:,:4]).max():.2f} (horizontal caps 0)")
flips = int(np.sum(np.diff(np.sign(u[:, 4] + 1e-12)) != 0))
print(f"u4 sign flips: {flips} (04_15 bang-bang: ~26 in 32 s)")
