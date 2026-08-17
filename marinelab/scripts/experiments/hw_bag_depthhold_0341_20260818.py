#!/usr/bin/env python3
"""Depth-hold retry bag 2026-08-18 03:41:32 (first run WITH vz_from_depth) — still a
+-7.5 cm limit cycle. THIRD root cause found: the scan phase machine itself.

What the earlier fixes DID deliver (all verified in this bag):
- attitude correct: roll ~0.1 deg, pitch ~3.2 deg static trim (imu_mount_rpy_deg works)
- solver at deploy setting: solve 16.1 ms mean, status==0 100%
- vz_from_depth alive and useful: vz_est tracks true vz (0.6 s-smoothed depth gradient)
  with corr 0.88, lag 0.50 s (= the LPF tau, as designed), amplitude ratio 0.72.
  DVL was 0% valid the whole bag — depth-derived vz carried ALL vertical damping.
- z_ref anchors at enable depth and ramps (reach_eps 0.05 in effect).

Root cause of the remaining oscillation — PHASE MACHINE CHURN: with
z_top == z_bottom == Z_HOLD and sway_step == 0, every phase's reach condition is
trivially satisfied, so the machine wrapped DESCEND->SWAY_A->ASCEND->SWAY_B
49 times in 34 s ("cycles" hit 12). Every SWAY entry re-latches z_hold at the
CURRENT depth, so z_ref bounced 0.130 <-> 0.169 <-> 0.121 — a +-4 cm square wave
injected into the reference. The MPC (werr z=40) answered each bounce with
saturated heave (87% of ticks at |u|>0.95, /wallscan/current_cmd sign-flipping
every ~0.85 s); the teleop ramp (~17-30 A/s) turned that bang-bang into a lagged
triangle (actual T5 range [-0.6,+1.8] A vs commanded +-3 A, amp ratio 0.21,
lag 0.4 s), and the loop settled into a 3.8 s / 15 cm p-p limit cycle
(z p-p per 7 s window: 18.4 / 15.0 / 13.4 / 15.0 cm — flat, self-sustained).

Compounding operator miss: depth_only was NOT active this run — u0..u3 sat at
their amps_limit clamps ([0.69,0.69,0.76,0.76] A commanded) driven by the
fictional blind-anchor radial error (r_est 3.70, fake wall error ~0.8 m).
Harmless to heave at pitch 3 deg (5% coupling) but must be on. Detection rule
for the runbook: |u0..u3| should be ~0 under depth_only; here mean |u0|=0.52.

Fixes (committed with this script):
1) WallScanControlLoop(hold_z=...) + wallscan_controller -p hold_z:=Z — pure
   station-keeping mode that BYPASSES the phase machine: z_ref slews once to the
   constant target (same ref_step slew + horizon preview), s_ref frozen at the
   enable-time s_hat, phase pinned to 0. Unit test:
   tests/control/test_scan_loop.py::test_hold_z_pins_the_reference...
2) Runbook scenario-3 T2 commands now carry hold_z:=0.13 and a hard checklist
   line: after enable, `ros2 topic echo /wallscan/u` — first four channels ~0,
   else depth_only was dropped.

Expectation for the next retry: no reference excitation left; residual motion
should be the deadzone-relay ripple (needed static force 0.24 N < minimum
realizable 0.37 N) — +-2-3 cm at a long period, u4/u5 NOT saturated.
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
BAG = "/home/mjkim/Downloads/rosbag2_2026_08_18-03_41_32"

o = {"st": [], "sz": [], "svz": [], "ct": [], "c": [], "ut": [], "u": [],
     "it": [], "cur": [], "dt": [], "dep": [], "cct": [], "cc": [],
     "et": [], "e": [], "dvt": [], "dvv": []}
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
        elif c.topic == "/wallscan/current_cmd":
            o["cct"].append(t); o["cc"].append(list(m.data))
        elif c.topic == "/wallscan/estimator_debug":
            o["et"].append(t); o["e"].append(list(m.data))
        elif c.topic == "/dvl/data":
            o["dvt"].append(t); o["dvv"].append(1.0 if m.velocity_valid else 0.0)
o = {k: np.array(v) for k, v in o.items()}

c, u = o["c"], o["u"]
ph = c[:, 1]
tr = np.where(np.diff(ph) != 0)[0]
print(f"dur {o['st'][-1]:.0f}s  solve {c[:,5][c[:,0]>0].mean():.1f} ms  "
      f"status!=0 {(c[:,6][c[:,0]>0]!=0).mean()*100:.1f}%  DVL valid {o['dvv'].mean()*100:.0f}%")
print(f"PHASE CHURN: {len(tr)} transitions, cycles={c[-1,2]:.0f} in {o['ct'][-1]:.0f}s")
print(f"z_ref bounce: uniq refs visited {sorted(set(np.round(c[:,3][c[:,0]>0], 2)))}")
print(f"depth_only check |u0..u3| mean: {[round(float(np.abs(u[:,i]).mean()),2) for i in range(4)]} (want ~0)")
print(f"heave saturation |u4|>0.95: {np.mean(np.abs(u[:,4])>0.95)*100:.0f}% of ticks")

tg = np.arange(5.0, 33.0, 0.05)
z = np.interp(tg, o["st"], o["sz"]); vz = np.interp(tg, o["st"], o["svz"])
k = 13
vzt = np.gradient(np.convolve(z, np.ones(k) / k, mode="same"), tg)
a0 = (z - z.mean()); print(f"z p-p {z.max()-z.min():.3f} m  std {z.std()*100:.1f} cm")


def lag(a, b, maxlag=3.0):
    n = int(maxlag / 0.05); best, bl = -2, 0
    a0 = (a - a.mean()) / a.std(); b0 = (b - b.mean()) / b.std()
    for L in range(-n, n + 1):
        cc = np.mean(a0[:len(a0) - L or None] * b0[L:]) if L >= 0 else np.mean(a0[-L:] * b0[:L])
        if cc > best:
            best, bl = cc, L
    return bl * 0.05, best


l1, c1 = lag(vzt, vz)
print(f"vz_est vs true vz: lag {l1:+.2f}s corr {c1:.2f} amp {vz.std()/vzt.std():.2f}")
i5c = np.interp(tg, o["cct"], o["cc"][:, 4]); i5a = np.interp(tg, o["it"], o["cur"][:, 4])
l4, c4 = lag(i5c, i5a)
print(f"teleop ramp: I_act vs I_cmd lag {l4:+.2f}s corr {c4:.2f} amp {i5a.std()/i5c.std():.2f} "
      f"(cmd +-3 A bang-bang, act [{o['cur'][:,4].min():+.1f},{o['cur'][:,4].max():+.1f}] A)")
