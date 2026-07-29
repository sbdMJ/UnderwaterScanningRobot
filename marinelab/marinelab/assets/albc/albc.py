# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ALBC underwater vehicle configuration for Isaac Lab.

The following configuration parameters are available:

* :obj:`ALBC_CFG`: ALBC articulation configuration
* :obj:`ALBCHydrodynamicsCfg`: Main body hydrodynamic parameters
* :obj:`ALBCBuoyHydrodynamicsCfg`: Buoy (link3) hydrodynamic parameters
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from ..uuv_cfg import HydrodynamicsCfg

# Path to ALBC USD file
_ALBC_MESHES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")
ALBC_USD_PATH = os.path.join(_ALBC_MESHES_DIR, "Agent.usd")


##
# Configuration - Hydrodynamics.
##


@configclass
class ALBCHydrodynamicsCfg(HydrodynamicsCfg):
    """Hydrodynamic parameters for ALBC main body.

    Cylinder formulas (Fossen, 2021) with URDF dimensions:
        Main body: a=0.18m (diameter), b=0.325m (length), m=9.18kg, rho=998 kg/m^3

    Physics Model:
        - PhysX gravity ENABLED; this model applies ONLY buoyancy as external force
        - Buoyancy-weight difference determines net vertical force
        - Appendage corrections (+15% translational drag) for gripper asymmetry
    """

    # Added mass [surge, sway, heave, roll, pitch, yaw]
    # Capped for explicit integration stability: M_a[i] < I_rigid[i]
    # surge/sway=8.0 (theory 8.25, capped), heave=1.0, roll/pitch=0.09, yaw=0.035
    added_mass: tuple[float, ...] = (8.0, 8.0, 1.0, 0.09, 0.09, 0.035)

    # Linear damping [surge, sway, heave, roll, pitch, yaw] (skin friction + appendage)
    # ITTC-1957 with x2.2 roughness+appendage correction
    linear_damping: tuple[float, ...] = (2.0, 2.0, 1.5, 0.3, 0.3, 0.15)

    # Quadratic damping [surge, sway, heave, roll, pitch, yaw] (form drag + appendage)
    # Cd_cross=1.17, Cd_axial=1.0, with appendage correction
    quadratic_damping: tuple[float, ...] = (39.0, 39.0, 15.0, 1.0, 1.0, 0.5)

    # Volume (m^3): pure cylinder 0.00827 + 8.8% appendage = 0.009
    # Buoyancy = 998 * 0.009 * 9.81 = 88.1 N
    volume: float = 0.009

    body_name: str = "base"

    # Center of buoyancy at geometric center of cylinder (body frame)
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Center of gravity below CoB for passive stability (restoring moment only)
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, -0.05)

    water_density: float = 998.0  # Freshwater (kg/m^3)

    # Rigid body inertia [Ixx, Iyy, Izz] from URDF: R=0.09m, L=0.325m, m=9.18kg
    rigid_body_inertia: tuple[float, float, float] | None = (0.0994, 0.0994, 0.0372)

    # Body mass from URDF (kg) - for CoG correction torque during domain randomization
    body_mass: float = 9.18

    # C_A only (not C_RB) -- PhysX handles rigid body Coriolis via enable_gyroscopic_forces
    use_full_coriolis: bool = False

    # Added mass force (M_A * v_dot) via explicit integration
    # Worst axis: yaw M_A=0.035 / I_zz=0.0372 = 0.94
    apply_added_mass_force: bool = True
    added_mass_stability_factor: float = 0.5

    # Semi-implicit damping clamp: max D_eff per axis = factor * I / dt
    damping_stability_factor: float = 0.8


@configclass
class ALBCBuoyHydrodynamicsCfg(HydrodynamicsCfg):
    """Hydrodynamic parameters for ALBC buoy (link3 / ABPC).

    Cylinder formulas (Fossen, 2021) with URDF dimensions:
        Buoy: a=0.17m (diameter), b=0.118m (length), m=0.93kg, rho=998 kg/m^3
        Short cylinder (b/a=0.69): added mass C_a reduced to ~0.75

    Physics Model:
        - PhysX gravity ENABLED; this model applies ONLY buoyancy as external force
        - Buoyancy = 26.2 N, Weight = 9.1 N -> Net = +17.1 N
    """

    # Added mass [surge, sway, heave, roll, pitch, yaw]
    # Capped for stability: M_a[i] < I_rigid[i] (m=0.93, Ixx=Iyy=0.00278, Izz=0.00336)
    # surge/sway=0.7 (theory 2.67, capped), heave=0.2, roll/pitch=0.002, yaw=0.002
    added_mass: tuple[float, ...] = (0.7, 0.7, 0.2, 0.002, 0.002, 0.002)

    # Linear damping (skin friction + roughness)
    linear_damping: tuple[float, ...] = (0.8, 0.8, 0.6, 0.02, 0.02, 0.01)

    # Quadratic damping (form drag, short-cylinder correction)
    quadratic_damping: tuple[float, ...] = (10.0, 10.0, 8.0, 0.05, 0.05, 0.02)

    # Volume (m^3): pi * 0.085^2 * 0.118 = 0.00268
    # Buoyancy = 26.2 N; system net: ~+10 N (main 88.1 + buoy 26.2 - weight 104.1)
    volume: float = 0.00268

    body_name: str = "link3"

    # Center of buoyancy in link3 frame (URDF collision origin z-offset)
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.059)

    # Center of gravity matching URDF inertial origin (symmetric cylinder, CoG=CoB)
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, 0.059)

    water_density: float = 998.0  # Freshwater (kg/m^3)

    # Rigid body inertia [Ixx, Iyy, Izz] from URDF: R=0.085m, H=0.118m, m=0.93kg
    rigid_body_inertia: tuple[float, float, float] | None = (0.00278, 0.00278, 0.00336)

    # Body mass from URDF (kg) - for CoG correction torque during domain randomization
    body_mass: float = 0.93

    # C_A only -- PhysX handles rigid body Coriolis via enable_gyroscopic_forces
    use_full_coriolis: bool = False

    # Added mass force: worst axis surge/sway 0.7/0.93 = 0.75
    apply_added_mass_force: bool = True
    added_mass_stability_factor: float = 0.4

    # Semi-implicit damping clamp
    damping_stability_factor: float = 0.8


##
# Configuration - ALBC Constants.
##

# ALBC (Active Linear Buoyancy Controller) joint names
ALBC_JOINT_NAMES: list[str] = ["joint1", "joint2"]

# ALBC Arm Geometry (from agent.urdf - keep in sync!)
# These values are extracted from URDF joint origins:
#   joint1: xyz="0 0 0.1625"
#   joint2: xyz="0.233 0 0.01"  (link1 length)
#   buoy_fixer: xyz="0.233 0 0.01"  (link2 length)
ALBC_LINK1_LENGTH: float = 0.233  # meters (joint2 x-offset from joint1)
ALBC_LINK2_LENGTH: float = 0.233  # meters (buoy_fixer x-offset from joint2)
ALBC_HEIGHT_OFFSET: float = 0.1625  # meters (joint1 z-offset from base)


##
# Configuration - Articulation.
##

ALBC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ALBC_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # PhysX handles gravity; HydrodynamicsModel applies buoyancy only
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
            # PhysX numerical damping (not hydrodynamic); 0.2 matches MarineGym
            linear_damping=0.2,
            angular_damping=0.2,
            max_angular_velocity=720.0,  # deg/s (= 4*pi rad/s); arm links accumulate parent joint velocities
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),  # 2m above ground plane (scene is underwater)
        rot=(1.0, 0.0, 0.0, 0.0),  # Upright orientation
        joint_pos={"joint.*": 0.0},
        joint_vel={"joint.*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            stiffness=100.0,  # Kp: w_n=57.7 rad/s with J~0.15 kg*m^2
            damping=3.0,  # Kd: damping ratio ~0.7 (near critically damped)
            effort_limit_sim=13.0,  # Nm, PhysX hard cap (above motor stall torque 9.5 Nm)
            velocity_limit_sim=3.1,  # rad/s, PhysX hard cap = measured XW540-T260 no-load plateau (2026-07-06)
        ),
    },
)
"""Configuration of ALBC underwater vehicle."""
