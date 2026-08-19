#!/usr/bin/env python3
"""Depth-hold bag 2026-08-19 23:47:09 — rate model + retuned weights + latency predictor,
all four startup WARNs live. VERDICT: PASS on the oscillation criteria — the limit-cycle
campaign (root causes 1-9) is closed. Two items remain below.

Measured against the acceptance criteria (NEXT_SESSION/E-2, deadzone-floor expectation):
  1. PASS  residual oscillation: settled 5 s-window p-p 1.6-2.8 cm (04_15: 13-18 cm,
     23_03: 15-17 cm). No periodic mode left — the z spectrum is flat (peak is the DC
     wander bin), vs the sharp 0.24 Hz line in 23_03. This IS the predicted deadzone-
     relay floor (needed static force < min realizable 0.37 N).
  2. PASS  heave unsaturated: 0% of ticks near the cap, u4 in [-0.19, +0.28], 33 gentle
     sign flips in 44 s (the deadzone dither), tick-delta at the modelled slew bound.
  3. PASS  phase pinned 0, z_ref one slew 0.975 -> 0.810, u0..u3 identically 0,
     status 0 100%, solve 24.7 ms (known nx-19 soft overrun, unchanged).
  4. PARTIAL  disturbance recovery: a ~7 cm excursion at t=11.7 s recovered in 1.2 s
     (exactly the matched-replay prediction), but the DELIBERATE 10 cm push-and-release
     of the protocol was not performed in this bag -> do it in the next session tick.

Remaining imperfection, documented not fixed: a +3..+6 cm one-sided offset ABOVE z_ref,
slowly wandering. Mechanism: the true equilibrium down-force sits INSIDE the thruster
friction deadzone (needed ~0.2-0.3 N < 0.37 N realizable floor), so the vehicle drifts
buoyant-up until the error is large enough for the MPC to command a super-deadzone nudge
— an asymmetric slow relay around a positive offset, the exact structural limit the
03_41 postmortem predicted. Mitigations, in order of preference: (a) method:=ssi — the
SSI learner estimates the residual disturbance online and should absorb the bias (this
is the next runbook step anyway, and the nominal-vs-adaptive gap here is precisely the
paper's story); (b) operational: set hold_z 4 cm below the true target; (c) accept — the
scan phase machine's reach bands (reach_eps) are wider than this bias in the real task.

E-2 status after this bag: depth-hold closed loop DEMONSTRATED in the small tank.
Root-cause ledger of the campaign: 1 fictional horizontal error -> depth_only,
2 IMU mount, 3 mapper calibration, 4 reach_eps latch, 5 DVL death -> vz_from_depth,
6 phase-machine churn -> hold_z, 7 stale flags -> runbook checklists, 8 instant-force
model -> actuator-rate OCP + retuned weights + thrust caps, 9 0.4 s command latency ->
in-flight predictor. Every fix is field-verified in this bag simultaneously.
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
BAG = "/home/mjkim/Downloads/rosbag2_2026_08_19-23_47_09"

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
      f"status!=0 {(c[:,6][en]!=0).mean()*100:.1f}%  z_ref {c[en,3][0]:.3f}->{c[en,3][-1]:.3f}  "
      f"phase uniq {sorted(set(c[en,1]))}")

tg = np.arange(t_en + 6.0, o["st"][-1] - 0.3, 0.02)
z = np.interp(tg, o["st"], o["sz"]); zref = np.interp(tg, o["ct"], c[:, 3])
err = z - zref
print(f"hold ({tg[-1]-tg[0]:.0f}s): |err| mean {np.abs(err).mean()*100:.1f} cm  "
      f"offset {err.mean()*100:+.1f} cm  z p-p {(z.max()-z.min())*100:.1f} cm")
pp5 = [(z[(tg >= a) & (tg < a + 5)].max() - z[(tg >= a) & (tg < a + 5)].min()) * 100
       for a in np.arange(tg[0] + 5, tg[-1] - 5, 5.0)]
print(f"settled 5s-window p-p: {min(pp5):.1f}-{max(pp5):.1f} cm (accept +-2-3 cm ripple)")
zc = z - z.mean()
sp = np.abs(np.fft.rfft(zc * np.hanning(len(zc)))) ** 2
fr = np.fft.rfftfreq(len(zc), 0.02)
band = (fr > 0.1) & (fr < 1.0)
print(f"periodic mode check: peak in 0.1-1 Hz band is "
      f"{sp[band].max()/sp[1:].max()*100:.0f}% of total peak (23_03: dominant 0.24 Hz line)")
ue = u[np.searchsorted(o["ut"], tg[0]):]
cap = 2.25 / 3.68
print(f"heave: sat {np.mean(np.abs(ue[:,4]) > 0.95*cap)*100:.0f}%  "
      f"u4 [{ue[:,4].min():+.2f},{ue[:,4].max():+.2f}]  "
      f"flips {int(np.sum(np.diff(np.sign(ue[:,4]+1e-12)) != 0))}  "
      f"|u0..u3| max {np.abs(ue[:,:4]).max():.3f}")
exc = np.abs(err) > 0.05
if exc.any():
    i0 = np.where(exc)[0][0]
    rec = np.where(~exc[i0:])[0]
    print(f"excursion >5 cm at t={tg[i0]:.1f}s, recovered in "
          f"{rec[0]*0.02:.1f}s" if len(rec) else "excursion not recovered")
