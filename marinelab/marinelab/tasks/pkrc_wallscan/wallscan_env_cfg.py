# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""WallScan env cfg — PKRC cylindrical-tank wall scan (station-keep -> vertical -> sway curriculum).

Not a BlueROVEnvCfg subclass: obs/reward shape are wallscan-specific (sonar/scan-state
blocks, no goal_pos), so this mirrors bluerov_env_cfg.py's scene/sim/physx/DR structure
directly rather than inheriting its hover-specific fields (per plan Task 5 interfaces).
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from marinelab.assets.pkrc import PKRC_CFG, PKRCHydrodynamicsCfg, PKRCThrusterCfg

from ..bluerov.bluerov_env_cfg import DomainRandomizationCfg
from .scan_state_machine import ScanCfg
from .sensors import SensorCfg
from .tank import TANK_HEIGHT, TANK_RADIUS


@configclass
class WallScanDoraemonCfg:
    """Adaptive dynamics-DR via DORAEMON (spec §6). Dynamics params only — sensor-bias
    DR stays fixed-uniform. MVP limitation: scheduler state is NOT restored across
    train.py --resume; the DR distribution restarts at init_concentration."""

    enable: bool = False
    success_cycles: int = 1     # episode success = completed scan cycles >= this
    alpha: float = 0.5          # target IS-estimated success rate
    kl_ub: float = 0.5          # trust-region KL bound per update
    init_concentration: float = 30.0
    # 250 (07-25 s_train diag): measured call rate is ~1 step() per RL iteration
    # (2971 calls / 3000 iters), NOT per reset event as originally assumed — at 2000
    # the optimizer fired only once per run. 250 iters x ~17 episodes/iter ~ 4000
    # episodes ~ exactly one buffer turnover per optimization (reference cadence).
    step_interval: int = 250
    min_episodes: int = 500
    buffer_size: int = 4000


@configclass
class WallScanEnvCfg(DirectRLEnvCfg):
    """Base config for the PKRC wall-scan task. See spec .sp/specs/2026-07-22-pkrc-wall-scan-rl-design.md."""

    # Environment settings
    # 120 s: a full scan cycle (descend 8 m + sway + ascend 8 m + sway + 4 reach-holds) needs
    # ~60-120 s; the original 40 s made cycles>=1 unreachable, so both the task success
    # termination and the DORAEMON success signal were permanently 0 (2026-07-22 train run).
    # 180 s: 120 s left the last phase advance ~20-40 s short (curriculum3: avg ~2/4 advances,
    # episodes timing out at 5592/6000 steps) — search ~10 s + descend + 2 sways + ascend + holds
    # needs ~140-160 s.
    episode_length_s: float = 180.0
    decimation: int = 2
    action_space: int = 6  # 6 thruster commands
    # up(3)+heading_sc(2) + ang_vel(3) + lin_vel(3) + wall_dist(1) + depth(1)
    # + ukfm_xy(2)+ukfm_yaw(1)+ukfm_valid(1) + cmd_err(3) + yaw_err_sc(2)+search_flag(1)
    # + phase_sc(2) + prev_action(6) = 31
    observation_space: int = 31
    state_space: int = 0
    debug_vis: bool = True

    # Simulation configuration (mirrors bluerov_env_cfg.py)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            enable_external_forces_every_iteration=True,
        ),
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=20.0,  # tank diameter is 12 m, needs more clearance than bluerov's open water
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # Robot configuration (PKRC)
    robot: ArticulationCfg = PKRC_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    body_link_name: str = "Robot"

    # Hydrodynamics / thrusters (PKRC-specific)
    hydrodynamics: PKRCHydrodynamicsCfg = PKRCHydrodynamicsCfg()
    thrusters: PKRCThrusterCfg = PKRCThrusterCfg()

    # Task-specific config
    sensors: SensorCfg = SensorCfg()
    # reach_eps 0.45: rollout traces (stage3_ext) show the depth-hold equilibrium parks at
    # 1.35-1.45 — 5 cm outside the 0.3 band, so DESCEND completion never triggered for most
    # envs. 0.45 covers the observed parking; +-45 cm settle precision is acceptable for a
    # 12 m tank scan.
    # z_top 8.5: at 9.0 the ascend band top (9.45) sat right under the FREE-FLOAT equilibrium
    # (positive buoyancy parks the robot at ~9.5+), so "just float" landed 5-25 cm outside the
    # band and the 3rd advance never triggered (stage3_fix2: waypoint saturated at exactly 400).
    # 8.5 puts 0.55 m between float equilibrium and the band -> settling requires a clear active
    # descent. Coverage loss is nominal: the ~0.9 m robot body reaches ~9 m when centered at 8.5.
    # reach_eps 0.6 (07-24 f_stage3c trace): a sway-tracking equilibrium parked 0.51-0.53 m
    # from the target — 6 cm outside the 0.45 band, under the OLD single-scale-0.5 tracking.
    # reach_eps 0.3 (07-26 m_train2 trace): with 2-scale tracking + the marker frame the
    # equilibria are tight (sway depth-hold 4 cm), but the 0.6 band let SWAY clear at its
    # edge — every descend STARTED -34 cm off s_ref and repaid it at 2.3 cm/s mid-descend
    # (96% same-sign, the whole residual "drift"). 0.3 forces settling before the gate.
    # Kill-boundary recheck: sway has no kill; z_bottom band floor 0.7 vs bottom kill 0.15;
    # z_top band top 8.8 vs float equilibrium ~9.5 (active settle kept) vs ceiling kill 10.2.
    # ref_step 0.004 = scan_speed_cap(0.2 m/s) x step_dt(0.02 s): the refs RAMP toward
    # the phase targets at the real scan speed (07-26 pacing — see ScanCfg.ref_step note).
    # reach_eps BACK to 0.6 (07-27 r_stage2 probe): a FRESH policy's lazy equilibrium parks
    # ~0.5 m short of the endpoint (ascend ended at 8.0 vs 8.5), outside the 0.3 band — it
    # never tastes a waypoint and sigma dies (bootstrap deadlock, pitfall #2 again). With
    # the ramp-arrival gate the 0.3-era cheat (clearing at the band edge to steal time) is
    # impossible anyway: the band is now only the bootstrap threshold; pacing & precision
    # live in the continuously-moving ramp.
    # ref_step_s 0.002 = 0.1 m/s sway ramp (07-28 tilt mitigation): sidestep thrust heels
    # the hull ~proportionally to speed via the TAM My coupling (6.1 deg at 0.2 m/s);
    # halving the sway leg speed targets ~3 deg. Zero-shot ramp slowing did NOT transfer
    # (the policy dashed ahead of the slower ramp and waited) — needs the short fine-tune.
    scan: ScanCfg = ScanCfg(z_top=8.5, z_bottom=1.0, sway_step=1.0, reach_eps=0.6, reach_hold=10,
                            ref_step=0.004, ref_step_s=0.002)

    # Tank geometry
    tank_radius: float = TANK_RADIUS
    tank_height: float = TANK_HEIGHT

    # Task targets
    d_ref: float = 1.5    # target wall distance, m
    d_min: float = 0.4    # wall-collision threshold, m
    # Initial spin-search: yaw reference sweeps a full turn at this rate to find the nearest
    # wall (sonar minimum); its bearing then becomes the per-env heading target. ~20 s/turn.
    search_omega: float = 0.63  # rad/s (~10 s per turn; 20 s ate too much of the episode budget)
    success_cycles: int = 3  # completed descend->sway->ascend->sway cycles for success termination (spec §9)

    # Reward weights (spec §6)
    wall_dist_scale: float = 5.0
    heading_scale: float = 5.0
    # depth/sway demoted 5.0 -> 1.5 (07-24 f_stage3 park diag): at 5.0 the banked tracking
    # income at a reached target beat advancing the scan — completing SWAY_A swaps z_ref to
    # z_top, collapsing depth reward for the ~29 s ascent, so the policy parked 1 m short of
    # the sway target forever. wall/heading stay 5.0: those are always-on constraints, not
    # phase objectives.
    depth_scale: float = 1.5
    # 2.5 (07-25 straightness spec): with _S_SCALE sharpened to 0.1 the term needs a bit more
    # authority to hold +-1 cm laterally during vertical transit; still well below the old 5.0
    # that caused reward-parking, and waypoint 400 + progress 75 keep dominating.
    sway_scale: float = 2.5
    upright_scale: float = 2.0
    # 200: at 10 the one-time advance bonus was ~0.01% of an episode's tracking reward (~90k over
    # 120 s), giving the policy no incentive to progress the scan (07-23 train120 stall).
    # 400: with tracking demoted to 1.5 this makes phase advance decisively net-positive under
    # the gamma=0.995 horizon (07-24 park diag arithmetic).
    waypoint_bonus: float = 400.0
    # Potential-based progress shaping on z/s target distance (diag: episodes time out
    # mid-ASCEND — exp-tracking pays ~0 far from target, nothing rewarded transit SPEED).
    # 75 (07-24 f_stage3c trace): at 25 a 0.25 m/s climb earned 0.13/step — buried under the
    # ~15/step tracking-reward noise, so ASCEND never trained (policy sat at the bottom
    # generalizing DESCEND's push-down habit). 75 -> 0.4/step, above the noise floor.
    # Telescoping potential term, so per-phase optimality is preserved (Ng 1999).
    progress_scale: float = 75.0
    collision_penalty: float = -50.0
    # -0.06 (07-25 ±1cm polish, was -0.03): u_train4 trace still showed ±3.9 cm descent
    # wobble — double the smoothing pressure alongside the new cross_vel term.
    action_rate_scale: float = -0.06
    action_mag_scale: float = -0.0005
    ang_vel_scale: float = -0.005
    alive_scale: float = 0.1
    # 07-25 ±1cm polish: linear penalty on the velocity component orthogonal to the
    # current transit axis (see wallscan_env cross_vel term). -2.0 -> a 0.07 m/s wobble
    # velocity costs ~0.14/step, same order as the tracking gradient over that error.
    cross_vel_scale: float = -2.0
    # 07-26 real-scan pacing: the real survey scans slower than the RL optimum — the
    # 0.77 m/s sway burst heeled the hull 8 deg via the TAM My coupling (trace-measured;
    # vertical phases sat at 0.1 deg). Cap the ALONG-axis transit speed (heave in
    # DESCEND/ASCEND, sway in SWAY); progress shaping stops paying above the cap so the
    # two never fight. Lower scan_speed_cap to match the real scan rate when known.
    scan_speed_cap: float = 0.2    # m/s along the transit axis
    overspeed_scale: float = -3.0  # linear penalty per m/s above the cap
    # 07-28 posture fix: the a-chain holds a LEARNED 5-7 deg lean in every phase (tilting
    # recruits the heave thrusters laterally) because the cos-based upright term is flat
    # near zero — slowing sway to 0.1 m/s changed nothing (tilt 6.1 -> 6.9 deg, measured).
    # Linear penalty on the tilt ANGLE puts a constant -0.14/deg gradient on it; the
    # m-chain proved 0.1 deg vertical tilt is learnable with this hull.
    tilt_scale: float = -8.0       # per radian of tilt

    # Curriculum flags (spec §8)
    enable_vertical: bool = True
    enable_sway: bool = True
    hard_collision_term: bool = True

    # Domain randomization
    randomization: DomainRandomizationCfg = DomainRandomizationCfg()
    doraemon: WallScanDoraemonCfg = WallScanDoraemonCfg()


@configclass
class WallScanStage1Cfg(WallScanEnvCfg):
    """Stage 1: station-keeping — hold wall distance + heading, no vertical/sway, soft collision term."""

    enable_vertical: bool = False
    enable_sway: bool = False
    hard_collision_term: bool = False


@configclass
class WallScanStage2Cfg(WallScanEnvCfg):
    """Stage 2: + vertical descend/ascend tracking, sway still off."""

    enable_vertical: bool = True
    enable_sway: bool = False
    hard_collision_term: bool = False
    # skip_sway (07-24 curriculum-hole diag, 2nd attempt): with sway rewards off nothing
    # holds s, and ~1 m of free tangential drift during the descend beat the sway_step=0
    # auto-clear band (g_stage2: end_phase pinned at 1.0 all 4000 iters) — so Stage2 never
    # trained ASCEND and Stage3 couldn't unlearn DESCEND's push-down habit. Skip the sway
    # phases structurally: Stage2 drills DESCEND<->ASCEND cycles directly.
    # ref_step/eps aligned with the base cfg (07-27 paced from-scratch chain): Stage2 must
    # also train AT the ramp speed, or it re-imprints the dash habit the chain exists to avoid.
    scan: ScanCfg = ScanCfg(z_top=8.5, z_bottom=1.0, reach_eps=0.6, reach_hold=10, skip_sway=True,
                            ref_step=0.004)


@configclass
class WallScanStage3Cfg(WallScanEnvCfg):
    """Stage 3: FULL cycle (vertical + sway) but NO DR, soft collision — master the scan
    pattern in a clean environment before Train piles on randomization (train_reach
    evidence: sway settling under DR is the learning frontier; strip DR to learn it)."""

    enable_vertical: bool = True
    enable_sway: bool = True
    hard_collision_term: bool = False


@configclass
class WallScanTrainCfg(WallScanEnvCfg):
    """Stage 3 / full task: vertical + sway zig-zag scan, hard collision termination, DR enabled."""

    enable_vertical: bool = True
    enable_sway: bool = True
    hard_collision_term: bool = True

    # Sensor DR (spec §7 sim2real): sonar mount extrinsics + per-episode bias on every
    # sensor (obs-only). Widened to the EVAL ranges (07-25 gap isolation: deterministic
    # >=1-cycle was 65.6% on the train distribution but 25.0% on eval — the held-out 2x
    # sensor bias was the whole gap, so train coverage must include the eval range).
    sensors: SensorCfg = SensorCfg(
        sonar_mount_dr=0.08, sonar_yaw_dr=0.04,
        sonar_bias_dr=0.10, depth_bias_dr=0.10,
        ins_att_bias_dr=0.04, ins_gyro_bias_dr=0.02,
        # dvl: a constant velocity bias INTEGRATES in the dead-reckoned sway obs
        # (drift = bias * 180 s); 0.01 -> <= 1.8 m worst-case drift, borderline vs the
        # 1 m sway_step but matches eval — watch Scan/end_phase if sway stalls return.
        dvl_bias_dr=0.01, ukfm_bias_dr=0.10,
    )

    randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        enable=True,
        # Initial position randomization is NOT via position_x/y/z_range (those are for
        # bluerov's box-shaped open water; _reset_idx does its own disk sample instead --
        # a box range doesn't fit a cylindrical tank).
        roll_range=(-0.628, 0.628),
        pitch_range=(-0.628, 0.628),
        yaw_range=(0.0, 6.283),
        # Hydrodynamic parameter randomization
        added_mass_scale=(0.8, 1.2),
        linear_damping_scale=(0.8, 1.2),
        quadratic_damping_scale=(0.8, 1.2),
        volume_scale=(0.95, 1.05),
        # Thruster randomization
        thrust_coefficient_scale=(0.9, 1.1),
        time_constant_scale=(0.9, 1.1),
        # CoB/CoG offset randomization (hard to measure IRL -> wide, spec §7)
        cob_offset_x=(-0.03, 0.03),
        cob_offset_y=(-0.03, 0.03),
        cob_offset_z=(-0.03, 0.03),
        cog_offset_x=(-0.03, 0.03),
        cog_offset_y=(-0.03, 0.03),
        cog_offset_z=(-0.03, 0.03),
        inertia_scale=(0.9, 1.1),
    )

    doraemon: WallScanDoraemonCfg = WallScanDoraemonCfg(enable=True)


@configclass
class WallScanEvalCfg(WallScanTrainCfg):
    """Evaluation: full task, wider randomization to stress-test policy robustness."""

    # Wider sensor DR for stress-testing: mount (+-8cm/+-0.04rad) + double the per-episode biases.
    sensors: SensorCfg = SensorCfg(
        sonar_mount_dr=0.08, sonar_yaw_dr=0.04,
        sonar_bias_dr=0.10, depth_bias_dr=0.10,
        ins_att_bias_dr=0.04, ins_gyro_bias_dr=0.02,
        dvl_bias_dr=0.01, ukfm_bias_dr=0.10,  # dvl: see Train comment (drift = bias * 120 s)
    )

    randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        enable=True,
        # Position randomization: see WallScanTrainCfg comment above (disk sample in _reset_idx).
        roll_range=(-0.785, 0.785),
        pitch_range=(-0.785, 0.785),
        yaw_range=(0.0, 6.283),
        added_mass_scale=(0.5, 1.5),
        linear_damping_scale=(0.5, 1.5),
        quadratic_damping_scale=(0.5, 1.5),
        volume_scale=(0.85, 1.15),
        thrust_coefficient_scale=(0.7, 1.3),
        time_constant_scale=(0.7, 1.3),
        cob_offset_x=(-0.05, 0.05),
        cob_offset_y=(-0.05, 0.05),
        cob_offset_z=(-0.05, 0.05),
        cog_offset_x=(-0.05, 0.05),
        cog_offset_y=(-0.05, 0.05),
        cog_offset_z=(-0.05, 0.05),
        inertia_scale=(0.8, 1.2),
    )

    doraemon: WallScanDoraemonCfg = WallScanDoraemonCfg(enable=False)
