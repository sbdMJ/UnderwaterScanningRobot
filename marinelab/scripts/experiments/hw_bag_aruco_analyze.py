#!/usr/bin/env python3
"""Characterize UKFM absolute correction from the marker-visible 122531 bag."""
import numpy as np
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

BAG = "/home/mjkim/PKRC로봇_코드_및_데이터/20260806_122531"
ts = get_typestore(Stores.ROS2_HUMBLE)
MSGDIR = Path("/home/mjkim/PKRC로봇_코드_및_데이터/hero_ws/src/dvl_msgs/msg")
types = {}
for f in MSGDIR.glob("*.msg"):
    types.update(get_types_from_msg(f.read_text(), f"dvl_msgs/msg/{f.stem}"))
ts.register(types)

aruco_t, aruco_xy = [], []
odom_t, odom_xy, odom_yaw = [], [], []
val_t = []
depth_t, depth = [], []

def yaw_of(q):
    return np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

with Reader(BAG) as r:
    conns = [c for c in r.connections if c.topic in (
        "/aruco/pose_6dof", "/ukfm/odom", "/ukfm/odom_validated", "/bar10xt/depth")]
    for conn, t_ns, raw in r.messages(connections=conns):
        t = t_ns * 1e-9
        m = ts.deserialize_cdr(raw, conn.msgtype)
        if conn.topic == "/aruco/pose_6dof":
            p = m.pose.position
            aruco_t.append(t); aruco_xy.append((p.x, p.y))
        elif conn.topic == "/ukfm/odom":
            p = m.pose.pose.position
            odom_t.append(t); odom_xy.append((p.x, p.y))
            odom_yaw.append(yaw_of(m.pose.pose.orientation))
        elif conn.topic == "/ukfm/odom_validated":
            val_t.append(t)
        elif conn.topic == "/bar10xt/depth":
            depth_t.append(t); depth.append(m.data)

t0 = min(odom_t)
aruco_t = np.array(aruco_t) - t0
odom_t = np.array(odom_t) - t0
val_t = np.array(val_t) - t0
odom_xy = np.array(odom_xy)
aruco_xy = np.array(aruco_xy)
span = odom_t.max()

print(f"span {span:.1f} s | aruco fixes {len(aruco_t)} | validated msgs {len(val_t)}")
print(f"depth range: [{min(depth):.2f}, {max(depth):.2f}] m")

# ArUco fix cadence
gaps = np.diff(aruco_t)
print(f"\naruco fix gaps: median {np.median(gaps):.2f} s  p90 {np.percentile(gaps,90):.2f} s"
      f"  max {gaps.max():.2f} s  (rate {len(aruco_t)/ (aruco_t.max()-aruco_t.min()):.2f} Hz)")

# validated coverage: fraction of odom ticks with a validated msg within 0.2 s
cov = np.mean([np.min(np.abs(val_t - t)) < 0.2 for t in odom_t]) if len(val_t) else 0.0
print(f"validated coverage of odom ticks: {100*cov:.0f}%")
print(f"validated active window: [{val_t.min():.1f}, {val_t.max():.1f}] s of [0, {span:.1f}]")

# innovation proxy: odom position at fix time vs aruco position (before correction pulls it)
inn = []
for ta, (ax, ay) in zip(aruco_t, aruco_xy):
    i = np.searchsorted(odom_t, ta) - 1
    if 0 <= i < len(odom_xy):
        inn.append(np.hypot(odom_xy[i][0] - ax, odom_xy[i][1] - ay))
inn = np.array(inn)
print(f"\n|odom - aruco| at fix times (innovation proxy):"
      f" median {np.median(inn)*100:.1f} cm  p90 {np.percentile(inn,90)*100:.1f} cm  max {inn.max()*100:.1f} cm")

# correction jumps: odom position step size right after a fix vs typical step
step = np.hypot(*np.diff(odom_xy, axis=0).T)
typical = np.median(step)
jump = []
for ta in aruco_t:
    i = np.searchsorted(odom_t, ta)
    if 0 < i < len(step):
        jump.append(step[i])
jump = np.array(jump)
print(f"odom step right after fix: median {np.median(jump)*100:.2f} cm vs typical step {typical*100:.2f} cm")

# dead-reckoning drift rate proxy: innovation vs time since previous fix
if len(inn) > 2:
    dt_since = np.concatenate([[np.nan], np.diff(aruco_t)])
    ok = ~np.isnan(dt_since)
    rate = inn[ok] / np.maximum(dt_since[ok], 1e-6)
    print(f"drift rate between fixes: median {np.median(rate)*100:.1f} cm/s  p90 {np.percentile(rate,90)*100:.1f} cm/s")
