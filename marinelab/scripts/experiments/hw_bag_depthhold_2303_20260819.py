#!/usr/bin/env python3
"""Depth-hold retry bag 2026-08-19 23:03:29 — rate model + retuned weights, correct
Z_HOLD (0.81, HOLD-Z guard passed). VERDICT: FAIL — 15-17 cm p-p / 4.2 s cap-to-cap
limit cycle. ROOT CAUSE 9: ~0.4 s of unmodeled round-trip COMMAND LATENCY.

Everything previously fixed stayed fixed (verified in this bag):
- rate model live (u4 tick-delta capped at 0.0915), thrust_limits live (u4 at -0.61 cap,
  u0..u3 = 0), phase 0, z_ref one slew 0.972 -> 0.810, solver status 0 100%.
- vz_from_depth healthy: corr 0.99 vs true vz, amp 0.78, lag 0.54 s (the earlier
  "corr 0.16" readout was an artifact of an over-tight differentiation window).
- state z = bar10xt ZOH exactly (no EKF smoothing lag), frame C = 0.850.
- solve 24.2 ms (nx-19 cost, over the 20 ms tick — same soft overrun as 22_36).

How the cause was pinned (isaaclab/logs/_probe_field_2303.py, matched replay):
1. Spectrum peak 0.24 Hz (T=4.2 s); F_cmd leads v by 0.98 s = T/4 with corr 0.93 —
   thrust drives a mass-dominated plant. Transfer amplitude gives m_eff ~ 24 kg = the
   model's 23.3 (added mass is NOT in this plant's mass matrix) — dynamics match.
2. Amplitude check: cap force 4.5 N at 0.24 Hz on 23.3 kg -> z amp = F/(m w^2) = 8.5 cm
   = the observed +-8 cm. Pure cap-to-cap relay, no external force, no sensor artifact.
3. Matched replay with 5 Hz depth + vz LPF + buoy error +0.25 N reproduces the field
   ONLY when ~0.4 s input dead time is injected: with it, ALL weight sets limit-cycle
   (retuned: 16.3 cm / 4.1 s — the field signature; default z=40: 19.9 / 4.5;
   BO: 16.4 / 3.7). Without it every set is far smaller/faster than the field.
   => the operator's config was CORRECT; the chain's dead time (solve overrun 24 ms +
   mapper hop + teleop 20 ms grid + CAN + T200 spin-up + sensor pipeline) is the driver.
4. Fix: Smith-style predictor — roll the measured x13 forward through the in-flight
   command history before each solve. In the matched replay it collapses the limit
   cycle to 0.6-2.2 cm p-p / 0% saturation and is robust to misestimating the delay
   (actual 0.0-0.6 s vs predicted 0.4 s; over-prediction is benign). The stiff default
   weights are NOT rescued by the predictor alone (11.7 cm) — keep the retuned set.

Committed with this script: PlantParams.command_latency_s (hw2026 JSON ships 0.4),
predictor inside WallScanMPC.solve (history reset on enable), node override param
command_latency_s + LATENCY PREDICTOR warn, runbook WARN checklist now 4 lines.
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
BAG = "/home/mjkim/Downloads/rosbag2_2026_08_19-23_03_29"

o = {"st": [], "sz": [], "ct": [], "c": [], "ut": [], "u": []}
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
o = {k: np.array(v) for k, v in o.items()}

c, u = o["c"], o["u"]
en = c[:, 0] > 0
t_en = o["ct"][en][0]
print(f"dur {o['st'][-1]:.0f}s  enabled t={t_en:.1f}s  solve {c[:,5][en].mean():.1f} ms  "
      f"status!=0 {(c[:,6][en]!=0).mean()*100:.1f}%  z_ref {c[en,3][0]:.3f}->{c[en,3][-1]:.3f}")

tg = np.arange(t_en + 6.0, o["st"][-1] - 0.3, 0.02)
z = np.interp(tg, o["st"], o["sz"]); zref = np.interp(tg, o["ct"], c[:, 3])
print(f"hold: z p-p {(z.max()-z.min())*100:.1f} cm  |err| mean {np.abs(z-zref).mean()*100:.1f} cm  "
      f"mean err {(z-zref).mean()*100:+.1f} cm")
zc = z - z.mean()
sp = np.abs(np.fft.rfft(zc * np.hanning(len(zc)))) ** 2
fr = np.fft.rfftfreq(len(zc), 0.02)
pk = fr[np.argmax(sp[1:]) + 1]
print(f"spectrum peak {pk:.2f} Hz (T={1/pk:.1f} s)")

u4 = np.interp(tg, o["ut"], u[:, 4])
F = (u4 + np.interp(tg, o["ut"], u[:, 5])) * 3.68
F = np.convolve(F, np.ones(11)/11, mode="same")
v = np.gradient(np.convolve(z, np.ones(11)/11, mode="same"), tg)
core = slice(100, -100)  # edge effects of the smoothing windows corrupt the xcorr
a1, b1 = F[core], v[core]
a1 = (a1 - a1.mean()) / a1.std(); b1 = (b1 - b1.mean()) / b1.std()
best, bl = -2, 0
for L in range(-150, 151):
    cc_ = np.mean(a1[:len(a1)-L or None]*b1[L:]) if L >= 0 else np.mean(a1[-L:]*b1[:L])
    if cc_ > best:
        best, bl = cc_, L
w = 2 * np.pi * pk
print(f"F_cmd leads v by {bl*0.02:+.2f} s (T/4 = {1/pk/4:.2f} s) corr {best:.2f} — mass-driven")
print(f"m_eff from |z/F|: {F.std()/ (zc.std() * w**2):.1f} kg (model rigid mass 23.3)")
print(f"cap occupancy |u4| > 0.95*0.61: {np.mean(np.abs(u4) > 0.95*2.25/3.68)*100:.0f}%")
print(f"amplitude sanity: cap 4.5 N / (23.3 * w^2) = {4.5/(23.3*w**2)*100:.1f} cm amp "
      f"vs observed {zc.std()*np.sqrt(2)*100:.1f} cm")
