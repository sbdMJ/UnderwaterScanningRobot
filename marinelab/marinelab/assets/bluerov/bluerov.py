# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BlueROV2 underwater vehicle configuration for Isaac Lab.

The BlueROV2 is a popular open-source underwater ROV manufactured by Blue Robotics.
This configuration includes:
    - Articulation settings (USD model, actuators)
    - Hydrodynamic parameters (from MarineGym, IROS 2025)
    - Thruster configuration (6-thruster layout)

Reference:
    MarineGym: A High-Fidelity Underwater Robot Simulator (IROS 2025)
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from ..uuv_cfg import HydrodynamicsCfg, ThrusterCfg

# Path to BlueROV USD file (relative to this module)
_BLUEROV_MESHES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")
BLUEROV_USD_PATH = os.path.join(_BLUEROV_MESHES_DIR, "BlueROV.usd")


@configclass
class BlueROVHydrodynamicsCfg(HydrodynamicsCfg):
    """Hydrodynamic parameters for BlueROV2.

    Parameters are from experimental identification (MarineGym, IROS 2025).

    BlueROV2 Heavy specifications:
        - Air weight: ~11.5 kg
        - Displacement volume: 11.35 L (0.0113459 m^3)
        - Designed for neutral buoyancy in freshwater

    Physics Model:
        - PhysX gravity is ENABLED (disable_gravity=False)
        - Weight is handled by PhysX naturally for all bodies
        - This model applies ONLY buoyancy as external force
        - For neutral buoyancy: buoyancy force ~= total weight
    """

    # Added mass coefficients [surge, sway, heave, roll, pitch, yaw]
    added_mass: tuple[float, ...] = (5.5, 12.7, 14.57, 0.12, 0.12, 0.12)

    # Linear damping coefficients
    linear_damping: tuple[float, ...] = (4.03, 6.22, 5.18, 0.07, 0.07, 0.07)

    # Quadratic damping coefficients
    quadratic_damping: tuple[float, ...] = (18.18, 21.66, 36.99, 1.55, 1.55, 1.55)

    # Vehicle volume (m^3). Set explicitly because the BlueROV USD collision
    # geometry auto-calculates to 1.0 m^3 (unit cube), which would make buoyancy
    # ~9780 N (88x weight). 0.0113459 m^3 = 11.35 L displacement (MarineGym ref,
    # IROS 2025) gives neutral buoyancy against the ~11.26 kg PhysX mass.
    volume: float | None = 0.0113459

    # Body name for volume auto-calculation (if volume is None)
    body_name: str = "base_link"

    # Center of buoyancy position in body frame (m)
    # Positive z means CoB above CoG (provides passive stability)
    center_of_buoyancy: tuple[float, float, float] = (0.0, 0.0, 0.01)

    # Center of gravity at body frame origin
    # Note: This is for restoring moment calculation. Actual CoG is from PhysX.
    center_of_gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Freshwater density
    water_density: float = 997.0

    # Enable full Coriolis matrix
    use_full_coriolis: bool = True


@configclass
class BlueROVThrusterCfg(ThrusterCfg):
    """Thruster configuration for BlueROV2 Heavy.

    BlueROV2 Heavy uses 6 T200 thrusters:
        - 4 horizontal (vectored) for surge, sway, yaw
        - 2 vertical for heave, pitch, roll
    """

    num_thrusters: int = 6
    max_thrust: float = 50.0
    thrust_coefficient: float = 40.0
    time_constant_up: float = 0.1
    time_constant_down: float = 0.05

    # BlueROV2 Heavy allocation matrix
    # Thrusters: [FL, FR, RL, RR, VL, VR] (Front-Left, Front-Right, Rear-Left, Rear-Right, Vert-Left, Vert-Right)
    allocation_matrix: tuple[tuple[float, ...], ...] = (
        # Fx: forward surge force (horizontal thrusters at 45 degrees)
        (0.707, 0.707, -0.707, -0.707, 0.0, 0.0),
        # Fy: lateral sway force
        (-0.707, 0.707, 0.707, -0.707, 0.0, 0.0),
        # Fz: vertical heave force
        (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        # Mx: roll torque (the two vertical thrusters are y-separated, opposite
        # signs -> they produce roll)
        (0.0, 0.0, 0.0, 0.0, 0.1, -0.1),
        # My: pitch torque. NOTE (A2): the two vertical thrusters are only
        # y-separated (not fore/aft / x-separated), so they cannot produce an
        # independent pitch moment. With both signs equal this row is effectively
        # a second heave contribution, not a controllable pitch input. This
        # matches the real BlueROV frame (ArduSub holds pitch/roll neutral on the
        # 6-thruster vectored config). Kept as-is (no geometric evidence in the
        # repo to justify flipping a sign); pitch is not independently actuated.
        (0.0, 0.0, 0.0, 0.0, 0.12, 0.12),
        # Mz: yaw torque
        (0.19, -0.19, 0.19, -0.19, 0.0, 0.0),
    )


BLUEROV_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=BLUEROV_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # PhysX handles gravity; HydrodynamicsModel applies buoyancy only
            max_depenetration_velocity=10.0,
            # use_full_coriolis=True in HydrodynamicsCfg already applies the
            # rigid-body gyroscopic term omega x (I omega). Disable PhysX's own
            # to avoid double-counting (see hydrodynamics.py warning).
            enable_gyroscopic_forces=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),  # Start 2m above ground (underwater)
        rot=(1.0, 0.0, 0.0, 0.0),  # Upright orientation
        joint_pos={"rotor_.*": 0.0},
        joint_vel={"rotor_.*": 0.0},
    ),
    actuators={
        "thrusters": ImplicitActuatorCfg(
            joint_names_expr=["rotor_.*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration of BlueROV2 Heavy underwater vehicle."""
