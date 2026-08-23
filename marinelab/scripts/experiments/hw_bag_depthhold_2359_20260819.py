#!/usr/bin/env python3
"""Depth-hold bag 2026-08-19 23:59:17 — deliberate push-and-release test (nominal, same
config as the passing 23_47 bag). VERDICT: PASS — criterion 4 (the last open one) closed.

Two hand pushes down, ~8 cm each (t~11 s: err -7.9 cm, t~24 s: err -7.6 cm). Both times
the controller answered with |u4| up to ~0.5-0.6 (briefly near the 0.61 cap — the right
response), crossed back through z_ref within 1-2 s, overshot into the familiar buoyant
offset band (+4..+6 cm, the deadzone equilibrium side), and settled. No oscillation
re-ignition, solver status 0 throughout, phase pinned 0. The residual wander outside the
push events is the known +0..+7 cm deadzone-offset band (wider here than the hands-off
23_47 bag because hands were in the water between events).

With this bag, ALL FOUR scenario-3 acceptance criteria are field-demonstrated on the
nominal method (ripple floor + unsaturated heave + phase pinned in 23_47; push recovery
here). Scenario-3 nominal = DONE.

Same-session note: the first method:=ssi attempt (23:59 session end) crashed the T2 node
at the first enabled tick — Float64MultiArray.data is an array.array and rejects
`+= list` in the new SSI debug append. Fixed with this commit (build the list first,
assign once) and the debug/publish tail is now exception-guarded so diagnostics can
never take the control loop down again. The mapper watchdog zeroed the currents on the
crash (stale-u path) — the safety envelope held. SSI retry is re-armed after rsync.
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
BAG = "/home/mjkim/Downloads/rosbag2_2026_08_19-23_59_17"

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
o = {k: np.array(v) for k, v in o.items()}

c, u = o["c"], o["u"]
en = c[:, 0] > 0
print(f"dur {o['st'][-1]:.0f}s  solve {c[:,5][en].mean():.1f} ms  "
      f"status!=0 {(c[:,6][en]!=0).mean()*100:.1f}%  phase {sorted(set(c[en,1]))}")

tg = np.arange(2.0, o["st"][-1] - 0.3, 0.02)
z = np.interp(tg, o["st"], o["sz"]); zref = np.interp(tg, o["ct"], c[:, 3])
u4 = np.interp(tg, o["ut"], u[:, 4])
err = z - zref

# push events: err below -5 cm (pushes were downward; the buoyant band is positive)
push = err < -0.05
i = 0
while i < len(push):
    if push[i]:
        j = i
        while j < len(push) and push[j]:
            j += 1
        ip = i + err[i:j].argmin()
        # recovery = first return to err >= -3 cm after the peak
        k = ip + np.argmax(err[ip:] >= -0.03)
        print(f"PUSH t={tg[ip]:.1f}s depth {err[ip]*100:+.1f} cm  "
              f"u4 response max {u4[ip:ip+150].max():+.2f} (cap 0.61)  "
              f"back to -3 cm in {(k-ip)*0.02:.1f}s  "
              f"overshoot then {err[k:k+400].max()*100:+.1f} cm (buoyant offset band)")
        i = j + 250
    else:
        i += 1
