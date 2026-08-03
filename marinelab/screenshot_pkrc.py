"""Render one screenshot of PKRC.usd to see its orientation.
Run: ./isaaclab.sh -p screenshot_pkrc.py [quat_w quat_x quat_y quat_z]
Saves /root/home/rl_ws/marinelab/pkrc_view.png
"""
import sys
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
sim_app = app_launcher.app

import numpy as np
import torch
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sensors import Camera, CameraCfg

USD = "/root/home/rl_ws/marinelab/marinelab/assets/pkrc/meshes/PKRC.usd"
q = tuple(float(x) for x in sys.argv[1:5]) if len(sys.argv) >= 5 else (1.0, 0.0, 0.0, 0.0)

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cpu"))
sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

spawn = sim_utils.UsdFileCfg(usd_path=USD)
spawn.func("/World/PKRC", spawn, translation=(0.0, 0.0, 1.0), orientation=q)   # above ground

cam = Camera(CameraCfg(
    prim_path="/World/cam",
    height=600, width=800,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(),
    offset=CameraCfg.OffsetCfg(pos=(3.0, 0.0, 1.2), convention="world"),
))
sim.reset()
cam.set_world_poses_from_view(
    eyes=torch.tensor([[3.0, 0.0, 1.2]]), targets=torch.tensor([[0.0, 0.0, 1.0]])
)
for _ in range(20):
    sim.step()
cam.update(sim.get_physics_dt())
rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
Image.fromarray(rgb).save("/root/home/rl_ws/marinelab/pkrc_view.png")
print("SHOT_OK bbox-nonblack:", int((rgb.sum(-1) > 30).sum()))
sim_app.close()
