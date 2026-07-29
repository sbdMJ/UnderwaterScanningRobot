"""Convert PKRC_robot.obj -> PKRC.usd (rigid body + convex collision + mass 22.8kg).
Run: ./isaaclab.sh -p /root/home/rl_ws/marinelab/convert_pkrc_usd.py
"""
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
sim_app = app_launcher.app

import os
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

D = "/root/home/rl_ws/marinelab/marinelab/assets/pkrc/meshes"
cfg = MeshConverterCfg(
    asset_path=os.path.join(D, "PKRC_robot.obj"),
    usd_dir=D,
    usd_file_name="PKRC.usd",
    force_usd_conversion=True,
    make_instanceable=False,
    scale=(0.01, 0.01, 0.01),                     # obj is in cm (matches .scn scale)
    # obj long axis is Y (Y-up); rotate +90 deg about X so long axis -> world Z
    # (vehicle stands VERTICAL, foam on top). Verified via screenshot_pkrc.py.
    # + 180 deg yaw (07-25): the obj's camera/sonar face landed on body -x, but the
    # whole frame convention (sonar mount +0.10 on +x, TAM surge=+x, heading control)
    # assumes front=+x — in the viewer the robot visibly faced AWAY from the wall.
    # Visual/collision-hull alignment only; body-frame physics and training unaffected.
    # R = Rz(180) * Rx(90) as a quaternion (w, x, y, z):
    rotation=(0.0, 0.0, 0.7071068, 0.7071068),
    mass_props=sim_utils.MassPropertiesCfg(mass=22.8),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    mesh_collision_props=sim_utils.ConvexHullPropertiesCfg(),
)
conv = MeshConverter(cfg)

# Single rigid body -> apply ArticulationRootAPI so it loads as a 1-link (floating base)
# articulation. The env (BlueROVEnv) uses the Isaac Articulation API; without this the
# spawn fails: "Failed to find an articulation ... no USD ArticulationRootAPI".
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open(conv.usd_path)
applied = False
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        print("ArticulationRoot applied to:", prim.GetPath())
        applied = True
        break
stage.GetRootLayer().Save()
assert applied, "no RigidBodyAPI prim found to make articulation root"
print("USD_OK:", conv.usd_path)
sim_app.close()
