#!/usr/bin/env python3
"""Phase A: characterize real PKRC sensors from the 2026-08-06 teleop rosbag.

Rates/gaps per topic, UKFM availability, and high-frequency noise floors
(detrended first-difference estimate: std(diff(x))/sqrt(2), which removes the
slow teleop motion and leaves the sensor noise — an upper bound, since real
motion also contributes to the diff).
"""
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

BAG = "/home/mjkim/PKRC로봇_코드_및_데이터/20260806_122731"
ts = get_typestore(Stores.ROS2_HUMBLE)

# register the wayfinder custom types from hero_ws
from pathlib import Path
from rosbags.typesys import get_types_from_msg
MSGDIR = Path("/home/mjkim/PKRC로봇_코드_및_데이터/hero_ws/src/dvl_msgs/msg")
types = {}
for f in MSGDIR.glob("*.msg"):
    types.update(get_types_from_msg(f.read_text(), f"dvl_msgs/msg/{f.stem}"))
ts.register(types)

series: dict[str, list] = {k: [] for k in (
    "ukfm_t", "ukfm_xy", "ukfm_yaw", "ukfm_z",
    "wall_t", "wall_d", "dvl_t", "dvl_v", "depth_t", "depth",
    "imu_t", "imu_rpy_rate", "ekf_t", "alt_t")}

def quat_to_yaw(q):
    return np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

with Reader(BAG) as r:
    conns = [c for c in r.connections if c.topic in (
        "/ukfm/odom", "/ukfm/wall_distance", "/dvl/data", "/bar10xt/depth",
        "/imu/data", "/ekf/odometry_earth", "/dvl/altitude")]
    for conn, t_ns, raw in r.messages(connections=conns):
        t = t_ns * 1e-9
        m = ts.deserialize_cdr(raw, conn.msgtype)
        if conn.topic == "/ukfm/odom":
            p = m.pose.pose.position
            series["ukfm_t"].append(t)
            series["ukfm_xy"].append((p.x, p.y))
            series["ukfm_z"].append(p.z)
            series["ukfm_yaw"].append(quat_to_yaw(m.pose.pose.orientation))
        elif conn.topic == "/ukfm/wall_distance":
            series["wall_t"].append(t); series["wall_d"].append(m.data)
        elif conn.topic == "/dvl/data":
            v = m.velocity
            series["dvl_t"].append(t); series["dvl_v"].append((v.x, v.y, v.z))
        elif conn.topic == "/bar10xt/depth":
            series["depth_t"].append(t); series["depth"].append(m.data)
        elif conn.topic == "/imu/data":
            w = m.angular_velocity
            series["imu_t"].append(t); series["imu_rpy_rate"].append((w.x, w.y, w.z))
        elif conn.topic == "/ekf/odometry_earth":
            series["ekf_t"].append(t)
        elif conn.topic == "/dvl/altitude":
            series["alt_t"].append(t)

def stats(name, t, x=None, unit=""):
    t = np.asarray(t)
    if len(t) < 3:
        print(f"{name:22s} n={len(t)} (too few)"); return
    dt = np.diff(t)
    line = f"{name:22s} n={len(t):5d}  rate={1/np.median(dt):5.1f} Hz  max_gap={dt.max()*1e3:6.0f} ms"
    if x is not None:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        noise = np.std(np.diff(x, axis=0), axis=0) / np.sqrt(2)
        line += f"  hf-noise={np.array2string(noise, precision=4)} {unit}"
    print(line)

span = max(series["imu_t"]) - min(series["imu_t"])
print(f"bag span used: {span:.1f} s\n")
stats("/ukfm/odom xy", series["ukfm_t"], series["ukfm_xy"], "m")
stats("/ukfm/odom yaw", series["ukfm_t"], np.unwrap(series["ukfm_yaw"]), "rad")
stats("/ukfm/odom z", series["ukfm_t"], series["ukfm_z"], "m")
stats("/ukfm/wall_distance", series["wall_t"], series["wall_d"], "m")
stats("/dvl/data vel", series["dvl_t"], series["dvl_v"], "m/s")
stats("/bar10xt/depth", series["depth_t"], series["depth"], "m")
stats("/imu/data gyro", series["imu_t"], series["imu_rpy_rate"], "rad/s")
stats("/ekf/odometry_earth", series["ekf_t"])
stats("/dvl/altitude", series["alt_t"])

# UKFM availability: fraction of the bag with a fix younger than 0.5 s,
# and dropout episodes (gap > 0.5 s) — the sim's ukfm_valid analogue.
t = np.asarray(series["ukfm_t"]); gaps = np.diff(t)
drop = gaps[gaps > 0.5]
print(f"\nUKFM availability: {100*(1 - drop.sum()/span):.1f}%  "
      f"dropouts>0.5s: {len(drop)}  longest={drop.max() if len(drop) else 0:.2f} s")
print(f"UKFM z vs depth sensor range: ukfm_z [{min(series['ukfm_z']):.2f}, {max(series['ukfm_z']):.2f}]"
      f"  bar10xt [{min(series['depth']):.2f}, {max(series['depth']):.2f}] m")
print(f"wall_distance range: [{min(series['wall_d']):.2f}, {max(series['wall_d']):.2f}] m")
