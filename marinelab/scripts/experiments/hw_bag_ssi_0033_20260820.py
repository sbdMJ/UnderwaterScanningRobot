#!/usr/bin/env python3
"""SSI depth-hold bag 2026-08-20 00:33:31 — first ssi run (rate model + latency predictor
+ retuned weights, new debug channels live). VERDICT: FAIL, and the debug channels caught
the mechanism directly. ROOT CAUSE 10: the LEARNER's feedback path is latency-corrupted.

What the bag shows (this script prints it):
- hold quality collapsed vs nominal: 5 s-window p-p 12.5-14.7 cm (nominal 23_47: 1.6-2.5),
  heave sat 44%, offset +7.0 cm. Solve 30.9 ms (learner overhead on top of nx-19).
- the new controller_debug[7:10] shows d_world OSCILLATING with |d| spikes of 10-12.6 N —
  2-3x the total heave authority (4.5 N). A real residual cannot be that large; the
  learner was injecting a ghost disturbance. pred_err flat at ~0.21 (no convergence).

Mechanism (matched replay _probe_field_2303.py, run_ssi cases): the RFF learner regresses
transitions x_k -> x_{k+1} against the command issued NOW, but under the chain's 0.4 s
dead time the transition is driven by the command from 0.4 s AGO — phase-shifted pairs
teach a phase-shifted residual, which is injected as d_world and re-excites the loop:
the replay reproduces the field signature (fixOFF: 16-17 cm p-p, |d| max ~10 N).
The MPC's own x0 latency predictor (root cause 9 fix) cannot help here — the learner is
a SECOND, parallel feedback path with its own latency problem.

Fix (committed with this script), validated in the same replay:
1. latency-aligned regression pairs — record_control receives the command from
   command_latency_s ago (in-flight FIFO, zeros on enable). Improves the ESTIMATE:
   d_z mean +0.32 N vs +0.69 N unaligned (true injected residual: +0.25 N).
2. injection low-pass ssi_d_tau=3 s — the STABILITY half: even aligned, a tick-rate
   learner injection under 0.4 s dead time is gain*delay unstable (replay: 20 cm limit
   cycle); the residuals this axis exists for are quasi-DC (buoyancy/deadzone bias,
   slow currents), so filtering d_world well below 1/dead-time restores a 1.3 cm hold
   with d_z converging to the true value.
3. |d_world| clamp ssi_d_max=5 N — physical sanity bound; a learning artifact must
   never outweigh the thrust authority inside the OCP.
Class defaults keep legacy sim behavior (no clamp/no tau/latency 0) — E1-E4 and the ssi
hyperparameter tuning are untouched; the deployed values are wallscan_controller node
parameter defaults. Replay end state: SSI @ 0.4 s dead time = 1.3 cm p-p, 0% sat,
d_z -> true residual. Field retry pending (rsync + T2 restart).
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
BAG = "/home/mjkim/Downloads/rosbag2_2026_08_20-00_33_31"

o = {"st": [], "sz": [], "ct": [], "c": [], "ut": [], "u": []}
with Reader(BAG) as r:
    t0 = r.start_time * 1e-9
    for c, tn, raw in r.messages():
        m = ts.deserialize_cdr(raw, c.msgtype)
        if c.topic == "/wallscan/state":
            o["st"].append(tn * 1e-9 - t0); o["sz"].append(m.pose.pose.position.z)
        elif c.topic == "/wallscan/controller_debug":
            o["ct"].append(tn * 1e-9 - t0); o["c"].append(list(m.data))
        elif c.topic == "/wallscan/u":
            o["ut"].append(tn * 1e-9 - t0); o["u"].append(list(m.data))
st = np.array(o["st"]); sz = np.array(o["sz"]); ct = np.array(o["ct"])
u = np.array(o["u"])
c7 = np.array([r[:7] for r in o["c"]])
en = c7[:, 0] > 0
t_en = ct[en][0]
D = np.array([r[7:10] for r in o["c"] if len(r) >= 11])
PE = np.array([r[10] for r in o["c"] if len(r) >= 11])
print(f"dur {st[-1]:.0f}s  solve {c7[en,5].mean():.1f} ms  status!=0 "
      f"{(c7[en,6]!=0).mean()*100:.1f}%  phase {sorted(set(c7[en,1]))}  ssi rows {len(D)}")

tg = np.arange(t_en + 6.0, st[-1] - 0.3, 0.02)
z = np.interp(tg, st, sz); zref = np.interp(tg, ct, c7[:, 3])
err = z - zref
pp5 = [(z[(tg >= a) & (tg < a + 5)].max() - z[(tg >= a) & (tg < a + 5)].min()) * 100
       for a in np.arange(tg[0], tg[-1] - 5, 5.0)]
print(f"hold: offset {err.mean()*100:+.1f} cm  5s p-p {min(pp5):.1f}-{max(pp5):.1f} cm "
      f"(nominal 23_47: 1.6-2.5)")
ue = u[np.searchsorted(o["ut"], tg[0]):]
cap = 2.25 / 3.68
print(f"heave sat {np.mean(np.abs(ue[:,4]) > 0.95*cap)*100:.0f}%")
print(f"d_world_z: mean {D[:,2].mean():+.2f} N  |d| max {np.abs(D).max():.1f} N "
      f"(heave authority 4.5 N — ghost)  pred_err mean {np.nanmean(PE):.3f} (flat = no convergence)")
