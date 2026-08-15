#!/usr/bin/env python3
"""§4g surge/sway drag-ID session, 2026-08-15 bags — extraction + verdict.

Setup (thruster_mapping.md §4g): MANUAL keyboard mode (heading hold active), robot
~0.5 m deep. /teleop/thruster_currents is the actual per-thruster VESC current in
manual mode (mix_thrusters publishes post gain x polarity). Surge speed from the
Ping1D driver topic /sensor/sonar/ping1d/range (forward wall range rate; the DVL is
dead in the acrylic pool); sway speed from the operator's tape-mark timing
(93 cm / ~7 s), the Ping range staying ~flat during sway is the cross-check.

Findings (see thruster_mapping.md §4g 결과):
- Drive current +-1.48 A on the pair -> deadzone-affine thrust ~2.5 N net for both
  axes (surge 2x1.594x(1.48-0.694), sway 2x1.754x(1.48-0.764)).
- Heading-hold contamination negligible: dYaw <= 7 deg on the clean runs and the
  correction currents on the other pair stay under the ~0.7 A deadzone (zero force).
- Surge terminals (trend-end of clean windows): forward ~0.10-0.11 m/s (seg1/seg4
  tails), reverse ~0.135-0.14 m/s plateau (seg7, conf=100 throughout, the cleanest
  run of the session). Effective drag 18-24 N/(m/s) at ~0.12 m/s.
- Sway: 0.133 m/s average over 93 cm INCLUDING the from-rest transient -> terminal
  >= 0.133; effective drag <= 18.9 N/(m/s) at 0.133 m/s.
- Sim comparison at the same speeds (pkrc_plant_fixed_tam.json d1+d2|v|):
  surge sim ~117-123 vs measured 18-24 -> ~5-6.5x too high;
  sway sim ~125 vs measured <=18.9 -> >=6.6x too high.
  (Heave was 2.3-3.5x, §4f.) Sim translational drag is 3-7x overestimated on ALL axes.
- At the 3 A horizontal cap the measured drag predicts surge 0.2-0.35 m/s and sway
  0.23-0.41 m/s (quad-only vs linear-only bracket) — both clear the scan ramps
  (sway ref 0.1 m/s already exceeded at HALF current). Heave descent (§4f: 0.15 m/s
  at 5 A < 0.2) remains the only binding axis -> ballast trim still required.
- Known limits: a single force level per axis (the 1.5 A manual scale), so d1/d2
  cannot be separated (the -0.89 A surge point sits too close to the deadzone to
  constrain a fit); Ping glitches (conf 0 spikes) excluded by the trend rule.
"""
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

ts = get_typestore(Stores.ROS2_HUMBLE)
BASE = "/home/mjkim/Downloads/8_15_실험"

K = {"surge": (1.594, 0.694), "sway": (1.754, 0.764)}  # newton_per_amp, amps_offset
SIM_D = {"surge": (97.79, 180.85), "sway": (119.44, 38.51)}  # d1, d2


def load(bag):
    out = {"it": [], "cur": [], "rt": [], "rng": [], "ct": [], "conf": [],
           "yt": [], "quat": []}
    with Reader(f"{BASE}/{bag}") as r:
        t0 = r.start_time * 1e-9
        for c, tn, raw in r.messages():
            t = tn * 1e-9 - t0
            m = ts.deserialize_cdr(raw, c.msgtype)
            if c.topic == "/teleop/thruster_currents":
                out["it"].append(t); out["cur"].append(list(m.data))
            elif c.topic == "/sensor/sonar/ping1d/range":
                out["rt"].append(t); out["rng"].append(m.range)
            elif c.topic == "/sensor/sonar/ping1d/confidence":
                out["ct"].append(t); out["conf"].append(m.data)
            elif c.topic == "/imu/data":
                out["yt"].append(t)
                q = m.orientation
                out["quat"].append((q.w, q.x, q.y, q.z))
    return {k: np.array(v) for k, v in out.items()}


def yaw_of(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def win_slope(t, y, a, b):
    m = (t >= a) & (t < b)
    if m.sum() < 3:
        return np.nan
    A = np.vstack([t[m], np.ones(m.sum())]).T
    return float(np.linalg.lstsq(A, y[m], rcond=None)[0][0])


def segments(it, cur, idx, thr=0.5):
    mag = np.abs(cur[:, idx]).max(axis=1)
    on = mag > thr
    e = np.flatnonzero(np.diff(on.astype(int)))
    starts = ([0] if on[0] else []) + [i for i in e if on[i + 1]]
    ends = [i for i in e if not on[i + 1]] + ([len(on) - 1] if on[-1] else [])
    return [(it[s], it[q]) for s, q in zip(starts, ends) if it[q] - it[s] > 1.0]


def pair_force(axis, amps):
    k, i0 = K[axis]
    return 2.0 * k * max(0.0, abs(amps) - i0)


def report(bag, axis, drive_idx):
    d = load(bag)
    it, cur = d["it"], d["cur"]
    yaw = yaw_of(d["quat"])
    print(f"\n######## {bag} ({axis})  dur={it[-1]:.0f}s  "
          f"conf<80: {(d['conf'] < 80).mean() * 100:.0f}%")
    for k, (a, b) in enumerate(segments(it, cur, drive_idx)):
        m = (it >= a) & (it < b)
        mean_c = cur[m].mean(axis=0)
        amps = np.abs(mean_c[drive_idx]).mean()
        my = (d["yt"] >= a) & (d["yt"] < b)
        dyaw = np.degrees(yaw[my][-1] - yaw[my][0]) if my.sum() > 2 else np.nan
        print(f"\nseg{k + 1}: t={a:6.1f}..{b:6.1f} ({b - a:4.1f}s)  "
              f"I=[{', '.join(f'{c:+.2f}' for c in mean_c[:4])}]  "
              f"F_net={pair_force(axis, amps):.2f} N  dYaw={dyaw:+.0f} deg")
        prof, x = [], a
        while x < b:
            prof.append(win_slope(d["rt"], d["rng"], x, x + 1.0))
            x += 0.5
        print("   dr/dt: " + " ".join(f"{p:+.2f}" if np.isfinite(p) else " ---" for p in prof))


report("surge_실험", "surge", [0, 1])
report("sway_실험", "sway", [2, 3])

print("\n######## verdict")
for axis, v, note in (("surge", 0.105, "fwd trend-end (seg1/seg4)"),
                      ("surge", 0.138, "rev plateau (seg7, conf 100)"),
                      ("sway", 0.133, "tape 93 cm / 7 s, incl. transient (lower bound)")):
    f = pair_force(axis, 1.48)
    d1, d2 = SIM_D[axis]
    sim_eff = d1 + d2 * v
    meas_eff = f / v
    print(f"{axis:5s} v={v:.3f} m/s  F={f:.2f} N  d_eff={meas_eff:5.1f}  "
          f"sim_eff={sim_eff:5.1f}  ratio={sim_eff / meas_eff:.1f}x  ({note})")
for axis in ("surge", "sway"):
    f3 = pair_force(axis, 3.0)
    f, v = pair_force(axis, 1.48), {"surge": 0.12, "sway": 0.133}[axis]
    v_quad = (f3 * v * v / f) ** 0.5      # pure-quadratic extrapolation
    v_lin = f3 * v / f                    # pure-linear extrapolation
    print(f"{axis:5s} @3 A: F={f3:.2f} N -> v in [{v_quad:.2f}, {v_lin:.2f}] m/s")
