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
class PKRCHydrodynamicsCfgZSlender(PKRCHydrodynamicsCfg):
    """PKRC hydrodynamics with the axes assigned to the hull the USD actually describes.

    2026-07-31: ``PKRCHydrodynamicsCfg`` and the USD mesh disagree about which body axis the
    vehicle is long along, and the disagreement is checkable without any external data
    (``core.parameters.slender_axis``):

    * ``added_mass = (19.40, 64.65, 64.65, ...)`` puts the SMALL entry on surge, i.e. claims
      the long axis is **x** (axial motion displaces the least water).
    * The PhysX inertia derived from ``PKRC.usd`` is ``diag(1.412, 1.406, 0.393)``, so the
      minimum is ``Izz`` and the long axis is **z**. Magnitudes match a 1.0 m x 0.42 m prolate
      spheroid almost exactly — analytic (0.402, 1.341, 1.341) vs measured
      (1.412, 1.406, 0.393) — so it is the same body with the axes permuted, not a different
      body. (Probe: ``isaaclab/logs/_probe_plant.py``.)

    The vehicle is a **z-slender (vertical) hull**, confirmed by the user, so the mesh is right
    and the coefficients are misassigned. Two consequences, and only two — the measured axes
    survive:

    1. **The measured surge/sway tow data is already consistent with z-slenderness.** Total
       drag at operating speed is nearly equal on the two axes (0.123 m/s: 14.8 vs 15.3 N;
       0.2 m/s: 26.8 vs 25.4 N), which is what a symmetric 0.42 x 0.42 x-y cross-section
       predicts. The 4.7x spread in the quadratic coefficients alone is the usual L-vs-Q fit
       ambiguity, not a physical asymmetry. So ``surge`` and ``sway`` are left untouched.
    2. **Heave is the axis that was wrong.** The shipped comment sets heave = sway as a "sway
       approximation (symmetric ~0.42x0.42 cross-section)" — correct reasoning applied to the
       wrong axis. On a z-slender hull heave is the AXIAL direction: it presents the small
       0.42 x 0.42 face (0.14 m^2) while surge/sway present 0.42 x 1.0 (0.33 m^2), a factor
       2.4 = a/b. So heave damping is scaled by 1/2.4 here, and the added-mass axis assignment
       is permuted so its small entry lands on heave.

    ``ref_step``-paced heave is the wallscan's PRIMARY motion axis, so the shipped numbers
    overstate the thrust a real descent needs by roughly 2x: 25.4 N of drag + 4.9 N net
    buoyancy = 30.3 N (cmd -0.379/thruster) becomes ~10.6 + 4.9 = 15.5 N (cmd -0.19).

    LIMITS of this config, stated plainly:

    * 1/2.4 is a **cross-sectional-area approximation**, chosen deliberately over waiting for
      data. Axial motion on a streamlined hull is also less separated than transverse motion,
      so the true reduction is probably LARGER than 2.4. A Stonefish steady-state heave tow —
      the same method that produced the surge/sway numbers — would replace the estimate.
    * The added-mass MAGNITUDES are inherited unverified. They are labelled an "analytic
      ellipsoid estimate" upstream and only their axis assignment is corrected here.
    * ``apply_added_mass_force`` stays False, so added mass acts only through the ``C_A``
      Coriolis coupling; the permutation's effect on the plant is therefore modest compared to
      the heave-damping change.

    Kept SEPARATE from ``PKRCHydrodynamicsCfg`` because ``checkpoints/rb_train_model_7998.pt``
    and every measurement in this repo so far were produced on the shipped coefficients.
    ``tests/test_hydro_axis_consistency.py`` pins both.
    """

    # small entry moved from surge to heave (axis permutation only; magnitudes inherited)
    added_mass: tuple[float, ...] = (64.65, 64.65, 19.40, 0.5, 0.5, 0.5)
    # surge/sway measured and unchanged; heave = sway / 2.4 (cross-sectional area ratio)
    linear_damping: tuple[float, ...] = (97.79, 119.44, 49.77, 15.0, 15.0, 4.0)
    quadratic_damping: tuple[float, ...] = (180.85, 38.51, 16.05, 30.0, 30.0, 8.0)
    # The plant's rotational dynamics use the PhysX tensor regardless; setting it explicitly
    # here makes the Coriolis term consistent with it instead of the added_mass[3:6]*0.5
    # fallback (0.25), which is 5.6x too small in roll/pitch.
    rigid_body_inertia: tuple[float, float, float] = (1.412, 1.406, 0.393)


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


@configclass
class PKRCThrusterCfgFixedTAM(PKRCThrusterCfg):
    """PKRC thrusters with the sway moment arm on the axis physics actually puts it.

    2026-07-30: ``PKRCThrusterCfg``'s TAM is not self-consistent. Its ``Fy`` row gives the
    sway thrusters a force of exactly ``(0, 1, 0)`` (no cant -- ``Fx``/``Fz`` are 0), and the
    moment of a force is always perpendicular to it (``tau = r x F``), so those columns have
    ``tau_y == 0`` for ANY mounting position. The 0.09 arm therefore cannot live in ``My``;
    with the sway thrusters at ``z = -0.09`` (the value the shipped comment names) it is
    ``Mx = -z*Fy = +0.09*Fy``. Every other row checks out against the geometry its own
    entries imply: heave differential at ``y = +-0.16`` -> ``Mx``, surge at ``y = +-0.15``
    and sway at ``x = +-0.15`` -> ``Mz``, both heave thrusters at ``x = 0`` -> no ``My``.
    ``core.thruster.allocation_moment_residual`` is the permanent guard for this class of
    typo; ``tests/test_thruster_allocation.py`` pins both matrices.

    MEASURED consequences (``isaaclab/logs/_probe_tam_tilt.py``, 4 envs x 18 s steady state,
    constant sway command sized for the 0.123 m/s leg -> ``F_y = 15.27 N``):

    | case                          | roll     | pitch    | \\|tilt\\| |
    |:------------------------------|---------:|---------:|---------:|
    | shipped TAM, sway             | +0.002   | +2.298   | 2.298    |
    | fixed TAM, sway               | +2.292   | +0.000   | 2.292    |
    | fixed TAM, sway + heave diff  | -0.006   | +0.000   | **0.000**|
    | shipped TAM, sway + heave diff| -2.297   | +2.300   | 3.250    |

    The closed-form prediction ``asin(0.09*F_y / (rho*g*V*z_cb)) = 2.298 deg`` matches the
    shipped-TAM measurement to three decimals, so the mechanism is understood, not fitted.
    The last row is the decisive one: under the shipped TAM a heave differential makes tilt
    WORSE (it adds a second tilt axis, ``sqrt(2.297^2 + 2.300^2) = 3.250``) because nothing
    but the sway thrusters can produce ``My``. Under the fixed TAM the same parasitic moment
    lands on roll, where the heave differential has authority, and
    ``dF_heave = -(0.09/0.16)*F_y`` cancels it exactly -- 8.6 N at 0.123 m/s, 14.3 N at
    0.2 m/s, against 40 N per thruster.

    Knock-on: the 07-28 tilt mitigation halved the sway leg to 0.1 m/s
    (``ScanCfg.ref_step_s``) precisely because the pitch was uncancellable. Measured here at
    the ORIGINAL 0.2 m/s: 3.821 deg uncompensated, **0.000 deg compensated** (heave commands
    -0.240/+0.118, a quarter of authority). So on this TAM full scan speed and zero tilt are
    available together -- revisit ``ref_step_s`` and ``tilt_scale`` before reusing the
    07-28 conclusions.

    Kept as a SEPARATE class rather than a fix in place: ``checkpoints/rb_train_model_7998.pt``
    was trained where sway heels the pitch axis, so its published numbers (sway tilt 2.20 deg)
    are only reproducible under ``PKRCThrusterCfg``. Still unconfirmed against the Stonefish
    ``pkrc_tam.yaml`` original, which is not present on this host -- that file is now only
    needed to confirm the ARM VALUES (0.09 / 0.16 / 0.15); the axis is settled by algebra and
    by the measurement above.
    """

    allocation_matrix: tuple[tuple[float, ...], ...] = (
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0),        # Fx surge
        (0.0, 0.0, 1.0, 1.0, 0.0, 0.0),        # Fy sway
        (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),        # Fz heave
        (0.0, 0.0, 0.09, 0.09, 0.16, -0.16),   # Mx roll: sway at z=-0.09 AND heave differential
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),        # My pitch: unactuated (heave pair sits at x=0)
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
