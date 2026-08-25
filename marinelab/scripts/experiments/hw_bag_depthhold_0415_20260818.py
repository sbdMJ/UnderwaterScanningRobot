#!/usr/bin/env python3
"""Depth-hold retry bag 2026-08-18 04:15:35 — first field run with hold_z + depth_only
both active. VERDICT: FAIL (criteria 4-5), but hold_z itself is field-verified.

Measured against the 03:41 postmortem criteria (hw_bag_depthhold_0341_20260818.py):
  1. PASS  phase pinned at 0 the whole bag, 0 transitions, cycles 0 — hold_z bypasses
     the phase machine in the field exactly as in the unit test.
  2. PASS  z_ref constant 0.130 m end to end, zero re-latching bounce.
  3. PASS (revised evidence) depth_only WAS active. |u0..u3| mean 0.10-0.33 is NOT the
     03:41 dropped-flag signature (quasi-DC clamp saturation at |u0| mean 0.52 tracking
     the ~1 m fictional wall error — here r_est 5.97 would make that error even LARGER,
     ~1.5 m, so OFF would look worse than 03:41, not milder). Instead u0..u3 flip sign
     in sync with the heave bang-bang (corr vs u4: -0.71/-0.75/-0.64/-0.48; ~25-34 sign
     flips each vs u4's 26): the cost-free recruitment the DEPTH-ONLY warning in
     wallscan_controller.py predicts — with heave saturated, the optimizer squeezes
     vertical force out of the horizontal pairs through the 3.2 deg pitch trim.
     => The runbook rule "u[0..3] ~ 0 else depth_only dropped" is only valid while
     heave is UNSATURATED; the runbook now says so.
  4. FAIL  heave still bang-bang: |u4|>0.95 on 81% of ticks (03:41: 87%), cmd +-3 A,
     u4/u5 identical (corr +1.00), 66% down / 34% up duty (positive-buoyancy trim).
  5. FAIL  z oscillation p-p 11.9 cm, std 3.1 cm, |z - z_ref| mean 4.5 cm max 11 cm
     around a +4.5 cm mean offset ABOVE the reference; dominant half-period ~1.5 s.
     Barely better than 03:41's 15 cm p-p even though the reference excitation is
     gone — the phase-machine churn (root cause 6) was an amplifier, not the driver.
  6. N/A   no push-and-release event in this 32 s bag (>5 cm excursions are the limit
     cycle itself; the t=7.0 s one "recovers" in 1.8 s = half a cycle).

ROOT CAUSE 8 — heave relay limit cycle (the residual driver, isolated now that the
reference is clean): werr z=40 makes the MPC a near-relay (a 5 cm error already
commands full scale), the plant answer is force-quantized (needed static force 0.24 N
< minimum realizable 0.37 N: deadzone relay, oscillation structurally unavoidable)
and LAGGED (teleop current ramp 17-30 A/s: actual T5 modulation +-0.7 A vs commanded
+-3 A, amp ratio ~0.23 at this period — same 0.21 measured in 03:41), plus vz comes
through a 0.5 s LPF on 5 Hz depth. Relay + lag => describing-function limit cycle at
+-5 cm instead of the +-2-3 cm deadzone floor the 03:41 postmortem hoped for. The MPC
model assumes instantaneous current tracking — the actuator mismatch is unmodeled.
Candidate fixes (decide before next tank session, in order of principle): (a) model
the current ramp/lag as a first-order actuator state in the OCP, (b) leave the relay
regime: lower werr z / add u-rate cost so 5 cm errors do not saturate, (c) raise the
heave amps_limit headroom so the static trim sits inside the proportional band.
The +4.5 cm mean offset is the asymmetric side of the same relay (down-duty 66%
against buoyancy, min force 0.37 N > needed 0.24 N -> it overshoots down, drifts up).

Solver health this bag: solve 15.9 ms mean, status==0 100%, DVL valid 0% (acrylic
tank, expected — vz_from_depth carried damping).
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
BAG = "/home/mjkim/Downloads/rosbag2_2026_08_18-04_15_35"

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
en = c[:, 0] > 0
t_en = o["ct"][en][0] if en.any() else None
print(f"dur {o['st'][-1]:.0f}s  enabled {en.mean()*100:.0f}% (from t={t_en:.1f}s)"
      if t_en is not None else "NEVER ENABLED — bag is not a closed-loop run")
print(f"solve {c[:,5][en].mean():.1f} ms  status!=0 {(c[:,6][en]!=0).mean()*100:.1f}%  "
      f"DVL valid {o['dvv'].mean()*100:.0f}%")

# 1-2. phase machine bypassed? z_ref constant after slew?
ph = c[en, 1]
tr = np.where(np.diff(ph) != 0)[0]
zr = c[en, 3]
print(f"[1] phase transitions {len(tr)} (want 0)  phase uniq {sorted(set(ph))}  "
      f"cycles end {c[-1,2]:.0f} (want 0)")
zr_late = zr[len(zr) // 3:]
print(f"[2] z_ref: start {zr[0]:.3f} -> end {zr[-1]:.3f}, last-2/3 p-p "
      f"{zr_late.max()-zr_late.min()*1:.3f} m (want ~0, no bounce)  "
      f"uniq(2cm) {sorted(set(np.round(zr, 2)))}")

# 3-4. depth_only + heave saturation (enabled ticks only)
ue = u[np.searchsorted(o["ut"], o["ct"][en][0]):]
print(f"[3] depth_only |u0..u3| mean {[round(float(np.abs(ue[:,i]).mean()),3) for i in range(4)]} (want ~0)")
print(f"[4] heave sat |u4|>0.95: {np.mean(np.abs(ue[:,4])>0.95)*100:.0f}% of ticks  "
      f"|u4| mean {np.abs(ue[:,4]).mean():.2f} max {np.abs(ue[:,4]).max():.2f}")

# 5. residual depth oscillation, enabled window, after the z_ref slew settles
t_hold = t_en + 5.0
tg = np.arange(t_hold, o["st"][-1] - 0.5, 0.05)
z = np.interp(tg, o["st"], o["sz"])
zref_g = np.interp(tg, o["ct"], c[:, 3])
err = z - zref_g
print(f"[5] hold window {tg[0]:.1f}-{tg[-1]:.1f}s: z p-p {(z.max()-z.min())*100:.1f} cm  "
      f"std {z.std()*100:.1f} cm  |z-z_ref| mean {np.abs(err).mean()*100:.1f} cm "
      f"max {np.abs(err).max()*100:.1f} cm (accept: +-2-3 cm)")
# dominant period via zero crossings of demeaned z
zd = z - np.convolve(z, np.ones(101) / 101, mode="same")
zc = np.where(np.diff(np.sign(zd[50:-50])) != 0)[0]
if len(zc) > 2:
    print(f"    dominant half-period ~{np.diff(zc).mean()*0.05:.1f} s "
          f"(03:41 limit cycle was 1.9 s half-period)")

# 6. any push-release event? (large |z - z_ref| excursion then recovery)
exc = np.abs(err) > 0.05
if exc.any():
    i0 = np.where(exc)[0][0]
    rec = np.where(~exc[i0:])[0]
    print(f"[6] disturbance >5 cm at t={tg[i0]:.1f}s, "
          + (f"recovered in {rec[0]*0.05:.1f}s" if len(rec) else "NOT recovered in bag"))
else:
    print("[6] no >5 cm excursion — no push-release event in this bag")

# supporting: vz_from_depth still alive, thruster currents vs cmd
tgv = tg
vz = np.interp(tgv, o["st"], o["svz"])
k = 13
vzt = np.gradient(np.convolve(np.interp(tgv, o["st"], o["sz"]), np.ones(k) / k, mode="same"), tgv)
if vz.std() > 1e-6 and vzt.std() > 1e-6:
    a0 = (vzt - vzt.mean()) / vzt.std(); b0 = (vz - vz.mean()) / vz.std()
    print(f"vz_from_depth: corr(vz_est, depth-gradient) {np.mean(a0*b0):.2f}  "
          f"vz std {vz.std():.3f} m/s")
i5c = np.interp(tg, o["cct"], o["cc"][:, 4]); i5a = np.interp(tg, o["it"], o["cur"][:, 4])
print(f"T5 current: cmd [{i5c.min():+.2f},{i5c.max():+.2f}] A  "
      f"act [{i5a.min():+.2f},{i5a.max():+.2f}] A  (03:41 was +-3 A bang-bang)")
