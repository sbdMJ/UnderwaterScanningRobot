# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""PKRC underwater vehicle configuration for Isaac Lab.

PKRC: 6-thruster UUV (surge x2, sway x2, heave x2), mass 22.8 kg, ~+0.5 kgf buoyant.
Hydrodynamic coefficients identified from the Stonefish PKRC model (see pkrc_sysid.py):
    - surge/sway damping: MEASURED (steady-state tow in Stonefish)
    - heave damping: sway approximation (symmetric ~0.42x0.42 cross-section)
    - added_mass: analytic ellipsoid estimate (steady tests cannot isolate it)
    - roll/pitch/yaw: placeholder (BlueROV-scale) -> tune after training
Thruster allocation matrix (TAM) copied verbatim from stonefish PKRC config/pkrc_tam.yaml.
"""
from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from ..uuv_cfg import HydrodynamicsCfg, ThrusterCfg

_PKRC_MESHES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")
PKRC_USD_PATH = os.path.join(_PKRC_MESHES_DIR, "PKRC.usd")


@configclass
class PKRCHydrodynamicsCfg(HydrodynamicsCfg):
    """Hydrodynamic parameters for PKRC. [surge, sway, heave, roll, pitch, yaw]."""

    # Analytic ellipsoid added mass (prolate spheroid a/b~2.4). roll/pitch/yaw: placeholder.
    added_mass: tuple[float, ...] = (19.40, 64.65, 64.65, 0.5, 0.5, 0.5)

    # Measured (surge/sway) + sway-approx (heave). roll/pitch/yaw: placeholder.
    # Rotational damping raised (07-25): at (1,1,1)/(5,5,5) the spin search wobbled the hull
    # 19-26 deg in sim while the real frame vehicle (top foam, flat frames) barely rolls or
    # pitches. Roll/pitch stiffened ~15x; yaw kept low so the 0.63 rad/s search sweep stays
    # inside the ~24 N*m Mz authority. ponytail: hand-calibrated to "barely tilts" — replace
    # with measured drag coefficients if the real vehicle gets characterized.
    linear_damping: tuple[float, ...] = (97.79, 119.44, 119.44, 15.0, 15.0, 4.0)
    quadratic_damping: tuple[float, ...] = (180.85, 38.51, 38.51, 30.0, 30.0, 8.0)

    # 22.8 kg mass + ~0.5 kgf positive buoyancy -> displaced volume for (22.8+0.5)/997.
    volume: float | None = 0.02337
    body_name: str = "Robot"          # single articulation body name (matches spawn prim)

    # Foam is mounted on TOP -> CoB well above CoG => strong self-righting (never lies down).
    # 0.15 m metacentric separation gives restoring moment ~33*sin(theta) N*m.
    # ponytail: 0.15 is a sensible foam-on-top estimate; set from CAD/CoM measurement for exactness.
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.15)
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    water_density: float = 997.0
    body_mass: float | None = 22.8
    use_full_coriolis: bool = True


@configclass
class PKRCThrusterCfg(ThrusterCfg):
    """PKRC 6-thruster config. Order: [Surge_L, Surge_R, Sway_L, Sway_R, Heave_L, Heave_R]."""

    num_thrusters: int = 6
    # BlueRobotics T200 @ 16V nominal: forward 5.25 kgf (51.5 N), reverse 4.1 kgf (40 N).
    # ponytail: assumes 16V operation; use 12V (36 N) or 20V (66 N) peak if the PKRC bus differs.
    max_thrust: float = 51.5          # T200 16V forward peak
    thrust_coefficient: float = 40.0  # effective full-command thrust (BlueROV/T200 convention)

    # Thruster Allocation Matrix (from stonefish PKRC config/pkrc_tam.yaml, rows = [Fx,Fy,Fz,Mx,My,Mz])
    allocation_matrix: tuple[tuple[float, ...], ...] = (
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0),        # Fx surge
        (0.0, 0.0, 1.0, 1.0, 0.0, 0.0),        # Fy sway
        (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),        # Fz heave
        (0.0, 0.0, 0.0, 0.0, 0.16, -0.16),     # Mx roll (heave differential)
        (0.0, 0.0, 0.09, 0.09, 0.0, 0.0),      # My pitch (sway at z=-0.09)
        (-0.15, 0.15, 0.15, -0.15, 0.0, 0.0),  # Mz yaw
    )


PKRC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=PKRC_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,            # PhysX gravity on; hydrodynamics applies buoyancy only
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=False,   # use_full_coriolis handles gyroscopic term
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={},                         # no joints: override default {".*": 0.0} (would match nothing)
        joint_vel={},
    ),
    actuators={},                             # single rigid body: no rotor joints (unlike BlueROV)
)
"""Configuration of the PKRC underwater vehicle."""
