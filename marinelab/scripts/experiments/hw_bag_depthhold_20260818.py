#!/usr/bin/env python3
"""First closed-loop depth-hold attempt (2026-08-18 bag) — FAILED, root cause found.

Symptom: enable at t=7.2 s, robot dove past the target and sat on the floor for
60 s (z_est 0.016 vs z_ref 0.044) with u4=u5=-1.0 (full down, saturated) and
actual /wallscan/current_cmd = [-3,+3] (descend).

Root cause — FICTIONAL horizontal errors dominated the MPC cost: with the
marker-less blind anchor the estimator fiction had drifted to r=5.34 (wall
distance 0.66 vs d_ref 1.5 -> 0.84 m fake error) and s_hat=+1.0 vs s_ref~0 by
the time enable arrived. At the first solve the depth error was ~ZERO yet the
optimizer saturated all four horizontal channels AND commanded full-down heave:
at the estimated +21 deg roll, heave thrust has a lateral world component, so
the MPC exploited the tilt to squeeze extra lateral force out of the heave pair
— trading (cheap) depth error for the (heavily weighted, fictional) wall/s
errors. Physically the horizontal thrusters were deadzone-clamped to zero
thrust, but THE MODEL DOES NOT KNOW THAT — the objective, not the actuator
clamp, had to change.

Fix: wallscan_controller depth_only:=true zeroes werr for
(radial, s, v_rad, v_tan, head_x, head_y), keeping z/v_z/roll/pitch/w — the
channels whose estimates are real (pressure + IMU) in the marker-less pool.
Also measured here: bar10xt carries a ~+0.5 m offset in this pool (state z:
floor 0.016, surface float ~0.25), so Z_HOLD must be read live (~0.10-0.15
mid-column), and the 'stationary' pre-enable fiction drift (r 4.5 -> 5.34)
shows the blind anchor decays within minutes — depth_only is mandatory, not
optional, without a marker.
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
BAG = "/home/mjkim/Downloads/rosbag2_2026_08_18-01_03_31"

o = {"st": [], "sz": [], "svz": [], "ct": [], "c": [], "ut": [], "u": [],
     "it": [], "cur": [], "dt": [], "dep": [], "et": [], "en": []}
with Reader(BAG) as r:
    t0 = r.start_time * 1e-9
    for c, tn, raw in r.messages():
        t = tn * 1e-9 - t0
        m = ts.deserialize_cdr(raw, c.msgtype)
        if c.topic == "/wallscan/state":
            o["st"].append(t); o["sz"].append(m.pose.pose.position.z)
            o["svz"].append(m.twist.twist.linear.z)
        elif c.topic == "/wallscan/controller_debug":
            o["ct"].append(t); o["c"].append(list(m.data))
        elif c.topic == "/wallscan/u":
            o["ut"].append(t); o["u"].append(list(m.data))
        elif c.topic == "/teleop/thruster_currents":
            o["it"].append(t); o["cur"].append(list(m.data))
        elif c.topic == "/bar10xt/depth":
            o["dt"].append(t); o["dep"].append(m.data)
        elif c.topic == "/wallscan/enable":
            o["et"].append(t); o["en"].append(m.data)
o = {k: np.array(v) for k, v in o.items()}
print(f"dur {o['st'][-1]:.0f}s  enable events: " +
      " ".join(f"{t:.1f}s:{'ON' if e else 'OFF'}" for t, e in zip(o["et"], o["en"])))
c = o["c"]  # [enabled, phase, cycles, z_ref, s_ref, solve_ms, status]
en = c[:, 0]
print(f"enabled fraction {en.mean()*100:.0f}%  status!=0: {(c[:,6][en>0]!=0).mean()*100:.1f}% (while on)")
# timeline every 2 s
print("\n t   en ph  z_ref  z_est  depth  u4     u5     I_T5   I_T6   vz")
for a in np.arange(0, o["st"][-1], 2.0):
    def w(tt, vv, fn=np.mean):
        m_ = (tt >= a) & (tt < a + 2)
        return fn(vv[m_]) if m_.any() else np.nan
    zref = w(o["ct"], c[:, 3]); zest = w(o["st"], o["sz"]); dep = w(o["dt"], o["dep"])
    u = o["u"]
    u4 = w(o["ut"], u[:, 4]); u5 = w(o["ut"], u[:, 5])
    i5 = w(o["it"], o["cur"][:, 4]); i6 = w(o["it"], o["cur"][:, 5])
    enn = w(o["ct"], en, np.max); ph = w(o["ct"], c[:, 1], np.max)
    vz = w(o["st"], o["svz"])
    print(f"{a:4.0f}  {enn:.0f} {ph:2.0f} {zref:6.3f} {zest:6.3f} {dep:6.3f} "
          f"{u4:+.3f} {u5:+.3f} {i5:+6.2f} {i6:+6.2f} {vz:+.3f}")
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
BAG = "/home/mjkim/Downloads/rosbag2_2026_08_18-01_03_31"

o = {"ct": [], "c": [], "et": [], "e": [], "ut": [], "u": [], "cc": [], "cct": [], "st": [], "sz": []}
with Reader(BAG) as r:
    t0 = r.start_time * 1e-9
    for c, tn, raw in r.messages():
        t = tn * 1e-9 - t0
        m = ts.deserialize_cdr(raw, c.msgtype)
        if c.topic == "/wallscan/controller_debug":
            o["ct"].append(t); o["c"].append(list(m.data))
        elif c.topic == "/wallscan/estimator_debug":
            o["et"].append(t); o["e"].append(list(m.data))
        elif c.topic == "/wallscan/u":
            o["ut"].append(t); o["u"].append(list(m.data))
        elif c.topic == "/wallscan/current_cmd":
            o["cct"].append(t); o["cc"].append(list(m.data))
        elif c.topic == "/wallscan/state":
            o["st"].append(t); o["sz"].append(m.pose.pose.position.z)
o = {k: np.array(v) for k, v in o.items()}

print("fine timeline t=6..18s (0.5 s):")
print("  t  ph  z_ref  s_ref |  r_est  phi   s_hat |  z_est |  u0..u5 | cc4 cc5")
for a in np.arange(6.0, 18.0, 0.5):
    def w(tt, vv, fn=np.mean):
        m_ = (tt >= a) & (tt < a + 0.5)
        return fn(vv[m_], axis=0) if m_.any() else None
    c = w(o["ct"], o["c"]); e = w(o["et"], o["e"]); u = w(o["ut"], o["u"])
    cc = w(o["cct"], o["cc"]); z = w(o["st"], o["sz"])
    if c is None: continue
    print(f"{a:5.1f} {c[1]:2.0f} {c[3]:6.3f} {c[4]:+6.2f} | {e[0]:5.2f} {e[1]:+5.2f} {e[2]:+6.2f} | {z:5.3f} | "
          + " ".join(f"{x:+.2f}" for x in u) + f" | {cc[4]:+5.2f} {cc[5]:+5.2f}")
print("\nlate (t=40): ", end="")
m_ = (o["ct"] >= 40) & (o["ct"] < 41)
c = o["c"][m_].mean(axis=0)
m_ = (o["et"] >= 40) & (o["et"] < 41)
e = o["e"][m_].mean(axis=0)
m_ = (o["ut"] >= 40) & (o["ut"] < 41)
u = o["u"][m_].mean(axis=0)
print(f"ph={c[1]:.0f} z_ref={c[3]:.3f} s_ref={c[4]:+.2f} | r={e[0]:.2f} phi={e[1]:+.2f} s_hat={e[2]:+.2f} | u=" +
      " ".join(f"{x:+.2f}" for x in u))
