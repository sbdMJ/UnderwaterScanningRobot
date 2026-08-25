#!/usr/bin/env python3
"""SSI depth-hold bag 2026-08-20 01:11:43 — retry with the root-cause-10 fixes
(latency-aligned pairs + d_world low-pass 3 s + clamp 5 N). VERDICT: PASS —
and SSI now BEATS nominal, which is the paper's point.

              nominal (23_47)    ssi broken (00_33)    ssi fixed (THIS BAG)
offset        +4.2 cm            +7.0 cm               +1.0 cm
ripple p-p    1.6-2.5 cm         11-15 cm              1.2-1.9 cm (settled, t>16 s)
heave sat     0%                 44%                    0%
d_world_z     —                  ghost, |d| to 12.7 N   +0.6..+0.9 N, |d| max 1.4 N
pred_err      —                  0.21 flat              0.07-0.16, decaying

The learner converged to a steady +0.6-0.9 N upward residual — the real unmodeled
buoyancy/deadzone bias (same order as the bag-23_03 static estimate of ~+0.5 N) — and
the OCP, told about it, plans commands that sit OUTSIDE the friction deadzone, which
removes the one-sided offset nominal structurally cannot fix (needed force < 0.37 N
realizable floor). First ~15 s show the convergence transient (p-p 5-6 cm) before the
3 s injection filter and OGD settle; from t=16 s every error sample is within
[-0.5, +2.3] cm. Matches the matched-replay prediction (1.3 cm hold, d_z -> true).

E-2 SCENARIO-3 CAMPAIGN FULLY CLOSED: nominal 4/4 criteria (bags 23_47 + 23_59) and
ssi adaptive pass (this bag), root causes 1-10 all identified, fixed, field-verified.
Nominal-vs-SSI on hardware now mirrors the paper's sim ordering: the adaptive method
absorbs the structural residual the tuned static method cannot.
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
BAG = "/home/mjkim/Downloads/rosbag2_2026_08_20-01_11_43"

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
st = np.array(o["st"]); ct = np.array(o["ct"]); u = np.array(o["u"])
sz = np.array(o["sz"])
c7 = np.array([r[:7] for r in o["c"]])
en = c7[:, 0] > 0
rows = [r for r in o["c"] if len(r) >= 11]
D = np.array([r[7:10] for r in rows]); PE = np.array([r[10] for r in rows])
print(f"dur {st[-1]:.0f}s  solve {c7[en,5].mean():.1f} ms  "
      f"status!=0 {(c7[en,6]!=0).mean()*100:.1f}%  phase {sorted(set(c7[en,1]))}")

t_en = ct[en][0]
tg = np.arange(t_en + 6.0, st[-1] - 0.3, 0.02)
z = np.interp(tg, st, sz); zref = np.interp(tg, ct, c7[:, 3])
err = z - zref
settled = tg > 16.0
pp5 = [(z[(tg >= a) & (tg < a + 5)].max() - z[(tg >= a) & (tg < a + 5)].min()) * 100
       for a in np.arange(16.0, tg[-1] - 5, 5.0)]
print(f"offset {err.mean()*100:+.1f} cm (nominal +4.2)  settled ripple p-p "
      f"{min(pp5):.1f}-{max(pp5):.1f} cm  settled err range "
      f"[{err[settled].min()*100:+.1f},{err[settled].max()*100:+.1f}] cm")
ue = u[np.searchsorted(o["ut"], tg[0]):]
print(f"heave sat {np.mean(np.abs(ue[:,4]) > 0.95*2.25/3.68)*100:.0f}%  "
      f"u4 [{ue[:,4].min():+.2f},{ue[:,4].max():+.2f}]")
print(f"learner: d_z mean {D[:,2].mean():+.2f} N (range {D[:,2].min():+.2f}..{D[:,2].max():+.2f})  "
      f"|d| max {np.abs(D).max():.2f} N (00_33 ghost: 12.7)  "
      f"pred_err mean {np.nanmean(PE):.3f} (00_33: 0.21 flat)")
