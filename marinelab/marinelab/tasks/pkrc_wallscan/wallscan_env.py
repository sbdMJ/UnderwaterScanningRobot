# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""WallScanEnv — PKRC cylindrical-tank wall-scan DirectRLEnv.

Wires together Tasks 1-5 (geometry, sensors, scan_state_machine, tank, cfg) with the same
hydrodynamics/thruster physics wiring as bluerov_env.py (same robot family, see spec §10:
"HydrodynamicsModel·ThrusterModel 등 물리 모델은 재사용, 관측 조립·보상·웨이포인트 상태기계·
종료만 신규 구현"). Not a BlueROVEnv subclass -- obs/reward shape are wallscan-specific.
"""
from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_from_euler_xyz

from marinelab.core import HydrodynamicsModel, ThrusterModel

from . import geometry, scan_state_machine
from .scan_state_machine import ScanState
from . import sensors as sensors_mod
from .sensors import _body_up, apply_sensors
from .tank import spawn_tank
from .wallscan_env_cfg import WallScanEnvCfg

# ponytail: exp-tracking "characteristic error" divisors below are placeholder constants (spec
# §13 marks exp-scale k as "구현 중 확정" -- unresolved pending training-curve tuning). The
# reward *weights* (wall_dist_scale etc.) live in cfg; only these inner scales are fixed here.
_D_SCALE = 0.5   # m, wall-distance tracking
_YAW_SCALE = 0.3  # rad, heading tracking
_Z_SCALE = 0.1    # m, depth tracking (07-25 straightness spec: hold/heave ripple +-1 cm; 0.1 makes
#   a 1 cm error cost ~10% of the term — cm-level gradient. Was 0.5. (older note: 1.0 -> 0.5: sharpen the near-target gradient so the
                  # tracking equilibrium sits inside the reach band; see cfg reach_eps note)
_S_SCALE = 0.1    # m, sway tracking (07-25 straightness spec: +-1 cm lateral hold; was 0.5 — at 0.2 m
#   error the gradient was lukewarm and the policy wandered +-10-26 cm during descent, trace-measured)

_REWARD_TERM_NAMES = (
    "wall_dist", "heading", "depth", "sway", "upright", "progress",
    "waypoint", "collision", "action_rate", "action_mag", "ang_vel", "alive",
    "cross_vel", "overspeed", "tilt",
)


def _exp_tracking(error: torch.Tensor, scale: float) -> torch.Tensor:
    """exp(-|error|/scale): 1.0 at zero error, ~0.37 at error==scale."""
    return torch.exp(-error.abs() / scale)


def _exp_tracking2(error: torch.Tensor, sharp: float, wide: float) -> torch.Tensor:
    """Dual-scale tracking: mean of a sharp and a wide exponential. The sharp term gives a
    cm-level gradient near the target; the wide term keeps pulling when the error is far
    outside the sharp scale (07-25: a lone 0.1 scale went flat past ~10 cm — the far-field
    no-gradient pitfall again — and sway-phase depth rippled +-42 cm unopposed)."""
    return 0.5 * (torch.exp(-error.abs() / sharp) + torch.exp(-error.abs() / wide))


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle (rad) to (-pi, pi] so heading error near +-pi doesn't look huge."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class WallScanEnv(DirectRLEnv):
    """PKRC wall-scan environment: hold wall distance/heading while scanning a cylindrical tank
    in a descend/sway/ascend/sway boustrophedon pattern (spec §5)."""

    cfg: WallScanEnvCfg

    def __init__(self, cfg: WallScanEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._body_id = self._robot.find_bodies(self.cfg.body_link_name)[0]

        # No ocean current in the tank (cfg has no ocean_current field, unlike bluerov's open water).
        self._hydro = HydrodynamicsModel(
            num_envs=self.num_envs,
            device=self.device,
            cfg=self.cfg.hydrodynamics,
            dt=self.physics_dt,
            articulation_prim_path=self.cfg.robot.prim_path.replace("env_.*", "env_0"),
        )
        self._thruster = ThrusterModel(
            cfg=self.cfg.thrusters,
            num_envs=self.num_envs,
            device=self.device,
            enable_randomization=self.cfg.randomization.enable,
        )

        # Action buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Force/torque buffers
        self._thrust_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust_torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._hydro_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self._hydro_torques = torch.zeros(self.num_envs, 3, device=self.device)

        self._up_w = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.num_envs, -1)

        # Waypoint scan state machine + sway dead-reckoning estimate (spec §3, §5).
        self._scan_state = ScanState(self.num_envs, device=self.device)
        # Marker-frame sway coordinate (07-25 refactor): ONE ruler for truth/estimate/refs —
        # arc length along the wall from a virtual ArUco marker whose wall angle is locked at
        # search end (the real stack's "first marker fix defines the map origin").
        self._s = torch.zeros(self.num_envs, device=self.device)      # estimate: marker fix when visible, DVL dead-reckon otherwise -> obs + phase timing
        self._s_gt = torch.zeros(self.num_envs, device=self.device)   # ground truth: geometric projection (no integration) -> sway reward
        self._marker_theta = torch.zeros(self.num_envs, device=self.device)  # virtual-marker wall angle (frame origin)
        self._theta_prev = torch.zeros(self.num_envs, device=self.device)    # GT wall angle last step (s unwrapping)
        self._z_ref = torch.full((self.num_envs,), self.cfg.scan.z_bottom, device=self.device)
        self._s_ref = torch.zeros(self.num_envs, device=self.device)
        self._phase_sc = torch.zeros(self.num_envs, 2, device=self.device)
        self._phase_sc[:, 1] = 1.0  # phase 0 (DESCEND): sin=0, cos=1
        self._cycles = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._clearance = torch.zeros(self.num_envs, device=self.device)

        # Initial spin-search: on reset the env sweeps the yaw reference one full turn at
        # cfg.search_omega while tracking the sonar-minimum bearing (= nearest wall), then
        # locks _yaw_ref to it and the scan cycle begins. Replaces the fixed cfg.yaw_ref=0.
        N = self.num_envs
        self._search_active = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._search_swept = torch.zeros(N, device=self.device)
        self._search_best_dist = torch.full((N,), float("inf"), device=self.device)
        self._search_best_yaw = torch.zeros(N, device=self.device)
        self._search_ref = torch.zeros(N, device=self.device)   # sweeping yaw reference
        self._yaw_ref = torch.zeros(N, device=self.device)      # locked per-env heading target
        self._yaw_ref_cur = torch.zeros(N, device=self.device)  # ref in force this step (sweep or locked)

        # Previous-step GT z / sway for potential-based progress shaping (diag run: episodes
        # time out mid-ASCEND — exp-tracking pays ~0 far from target, so nothing rewards
        # moving FAST toward it; shaping = scale * (prev_dist - curr_dist) fixes that without
        # changing the optimal policy, Ng et al. 1999).
        self._prev_z = torch.zeros(N, device=self.device)
        self._prev_s_gt = torch.zeros(N, device=self.device)

        # Termination-cause masks (set in _get_dones; consumed by reset-time telemetry).
        self._term_collided = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._term_oob = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._term_tilted = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Sonar mount extrinsics (body frame): nominal (reward ground truth) + per-episode DR'd
        # copy (observation). In non-DR stages the DR'd copy stays at nominal. (spec §7 sim2real)
        _mp = self.cfg.sensors.sonar_mount_pos
        self._sonar_mount_nom = torch.tensor([_mp], device=self.device).expand(self.num_envs, -1)
        self._sonar_yaw_nom = float(self.cfg.sensors.sonar_yaw_offset)
        self._sonar_mount = self._sonar_mount_nom.clone()
        self._sonar_yaw = torch.full((self.num_envs,), self._sonar_yaw_nom, device=self.device)

        # Per-episode sensor-bias DR buffers (obs-only; zero in non-DR stages). Resampled in
        # _reset_idx from cfg.sensors.*_bias_dr; see _resample_sensor_bias. (spec §7 sim2real)
        N, dev = self.num_envs, self.device
        self._sensor_bias = {
            "sonar": torch.zeros(N, device=dev), "depth": torch.zeros(N, device=dev),
            "up_vec": torch.zeros(N, 3, device=dev), "heading": torch.zeros(N, device=dev),
            "ang_vel": torch.zeros(N, 3, device=dev), "lin_vel": torch.zeros(N, 3, device=dev),
            "ukfm_xy": torch.zeros(N, 2, device=dev), "ukfm_yaw": torch.zeros(N, device=dev),
            "lin_vel_scale": torch.zeros(N, 3, device=dev),
        }

        # DORAEMON adaptive dynamics DR (Train only; spec §6). _dr_xi/_dr_logp hold the
        # sample actually applied to each env's CURRENT episode — record_episodes must
        # use these, never a fresh sample (IS-estimator correctness).
        self._ep_return = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.doraemon.enable:
            from marinelab.algorithms.doraemon import DoraemonCfg

            from . import doraemon_dr

            _d = self.cfg.doraemon
            self._doraemon = doraemon_dr.build_scheduler(
                DoraemonCfg(
                    alpha=_d.alpha, kl_ub=_d.kl_ub, init_concentration=_d.init_concentration,
                    step_interval=_d.step_interval, min_episodes=_d.min_episodes,
                    buffer_size=_d.buffer_size,
                ),
                self.cfg.randomization,
                self.device,
            )
            self._dr_xi = torch.zeros(self.num_envs, len(doraemon_dr.PARAM_DEFS), device=self.device)
            self._dr_logp = torch.zeros(self.num_envs, device=self.device)
            self._ep_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._episode_reward_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in _REWARD_TERM_NAMES
        }

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Per-env tank shell, cloned alongside the robot (same {ENV_REGEX_NS} pattern).
        spawn_tank(prim_path="/World/envs/env_.*/Tank")

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.4, 0.6, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Physics stepping (verbatim from bluerov_env.py -- same robot family)
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thruster.apply_dynamics(self._actions, self.step_dt)

    def _apply_action(self):
        thrust_forces, thrust_torques = self._thruster.compute_wrench()
        self._thrust_forces[:, 0, :] = thrust_forces
        self._thrust_torques[:, 0, :] = thrust_torques

        self._hydro_forces, self._hydro_torques = self._hydro.compute_forces(
            root_lin_vel_w=self._robot.data.root_lin_vel_w,
            root_ang_vel_w=self._robot.data.root_ang_vel_w,
            root_quat_w=self._robot.data.root_quat_w,
        )

        total_forces = self._thrust_forces.clone()
        total_forces[:, 0, :] += self._hydro_forces
        total_torques = self._thrust_torques.clone()
        total_torques[:, 0, :] += self._hydro_torques

        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=total_forces,
            torques=total_torques,
        )

    # ------------------------------------------------------------------
    # State / sensors
    # ------------------------------------------------------------------

    def _read_state(self) -> tuple[dict, dict]:
        """Ground-truth robot state (tank-local frame) + noisy sensor readings for it."""
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        pos_local = root_pos_w - self.scene.env_origins
        _, _, heading = euler_xyz_from_quat(root_quat_w)
        # Sonar measured from the mount pose: nominal mount -> reward ground truth (clean),
        # DR'd mount -> observation (what the real, imperfectly-mounted sonar reads).
        wall_dist = geometry.sonar_wall_distance(
            pos_local[:, :2], heading, self._sonar_mount_nom, self._sonar_yaw_nom, self.cfg.tank_radius
        )
        wall_dist_meas = geometry.sonar_wall_distance(
            pos_local[:, :2], heading, self._sonar_mount, self._sonar_yaw, self.cfg.tank_radius
        )

        gt = dict(
            pos=pos_local,
            quat=root_quat_w,
            up_vec=_body_up(root_quat_w),  # GT body-up (world) -> upright reward, clean of DR bias
            ang_vel=self._robot.data.root_ang_vel_b,
            lin_vel=self._robot.data.root_lin_vel_b,
            lin_vel_w=self._robot.data.root_lin_vel_w,
            heading=heading,
            wall_dist=wall_dist,
            wall_dist_meas=wall_dist_meas,
        )
        sensors = apply_sensors(gt, self.cfg.sensors, gen=None, bias=self._sensor_bias)
        return gt, sensors

    # ------------------------------------------------------------------
    # Observations / rewards / dones
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        if self._hydro.apply_added_mass:
            self._hydro.update_physx_state(
                body_com_acc_w=self._robot.data.body_com_acc_w[:, self._body_id[0], :],
                root_quat_w=self._robot.data.root_quat_w,
            )

        _, sensors = self._read_state()
        cfg = self.cfg

        cmd_err = torch.stack(
            [
                sensors["sonar"] - cfg.d_ref,
                sensors["depth"] - self._z_ref,
                self._s - self._s_ref,
            ],
            dim=-1,
        )

        # Heading error vs the CURRENT per-env yaw reference (sweep or locked) — without this
        # the time-varying reference would be unobservable. Measured (noisy) heading recovered
        # from heading_sc; sin/cos of the error avoids the +-pi wrap discontinuity.
        heading_meas = torch.atan2(sensors["heading_sc"][:, 0], sensors["heading_sc"][:, 1])
        yaw_err = _wrap_to_pi(heading_meas - self._yaw_ref_cur)

        obs = torch.cat(
            [
                sensors["up_vec"],                    # 3
                sensors["heading_sc"],                 # 2
                sensors["ang_vel"],                    # 3
                sensors["lin_vel"],                    # 3
                sensors["sonar"].unsqueeze(-1),        # 1
                sensors["depth"].unsqueeze(-1),        # 1
                sensors["ukfm_xy"],                    # 2
                sensors["ukfm_yaw"].unsqueeze(-1),     # 1
                sensors["ukfm_valid"].unsqueeze(-1),   # 1
                cmd_err,                               # 3
                torch.sin(yaw_err).unsqueeze(-1),      # 1
                torch.cos(yaw_err).unsqueeze(-1),      # 1
                self._search_active.float().unsqueeze(-1),  # 1 (spin-search phase flag)
                self._phase_sc,                        # 2
                self._actions,                         # 6 (action just applied = "prev" for next decision)
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _compute_reward_terms(self, gt: dict, sensors: dict, z_ref, s_ref, advanced) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        collided = self._clearance < cfg.d_min

        return {
            # wall_dist/depth use GROUND TRUTH: reward measures true performance, the policy
            # only ever SEES noise via the observation (sensors["sonar"]/["depth"]).
            "wall_dist": _exp_tracking(gt["wall_dist"] - cfg.d_ref, _D_SCALE) * cfg.wall_dist_scale,
            # heading tracks the per-env reference: the sweeping search ref while searching,
            # the locked nearest-wall bearing afterwards (was: fixed cfg.yaw_ref=0).
            "heading": _exp_tracking(_wrap_to_pi(gt["heading"] - self._yaw_ref_cur), _YAW_SCALE) * cfg.heading_scale,
            "depth": _exp_tracking2(gt["pos"][:, 2] - z_ref, _Z_SCALE, 0.5) * cfg.depth_scale * float(cfg.enable_vertical),
            # sway on GROUND TRUTH (self._s_gt): the DVL-integrated self._s carries per-episode bias
            # drift, which would let the policy cancel a sensor artifact by moving the true body off-target.
            # Search gate on the s terms (07-26 first-cycle settling): before the marker
            # locks, s_gt runs off the SPAWN-angle placeholder — ill-conditioned when the
            # robot spawns near the tank centre (tiny xy motion = metre-scale s spikes,
            # trace-measured) — so grading s during the spin injects random gradients.
            "sway": _exp_tracking2(self._s_gt - s_ref, _S_SCALE, 0.5)
            * cfg.sway_scale * float(cfg.enable_sway) * (~self._search_active).float(),
            # Potential-based progress shaping (GT): prev/curr distances both use the CURRENT
            # ref, so a ref switch on phase advance never mints free reward. Telescopes to
            # scale * total distance covered toward targets — dense incentive to transit fast.
            # Per-step progress is CLAMPED at scan_speed_cap*dt (07-26 pacing): above the
            # cap, moving faster earns nothing — otherwise shaping would fight the
            # overspeed penalty below. Regression (negative delta) passes unclamped.
            "progress": (
                ((self._prev_z - z_ref).abs() - (gt["pos"][:, 2] - z_ref).abs())
                .clamp(max=cfg.scan_speed_cap * self.step_dt) * float(cfg.enable_vertical)
                + ((self._prev_s_gt - s_ref).abs() - (self._s_gt - s_ref).abs())
                .clamp(max=cfg.scan_speed_cap * self.step_dt)
                * float(cfg.enable_sway) * (~self._search_active).float()
            ) * cfg.progress_scale,
            # upright/ang_vel on GROUND TRUTH (gt), not sensors: per-episode sensor bias would
            # otherwise penalize the policy for an offset it can neither observe nor correct.
            "upright": gt["up_vec"][:, 2].clamp(-1.0, 1.0) * cfg.upright_scale,
            "waypoint": advanced.to(sensors["sonar"].dtype) * cfg.waypoint_bonus,
            "collision": collided.to(sensors["sonar"].dtype) * cfg.collision_penalty,
            "action_rate": torch.sum((self._actions - self._prev_actions) ** 2, dim=1) * cfg.action_rate_scale,
            "action_mag": torch.sum(self._actions**2, dim=1) * cfg.action_mag_scale,
            "ang_vel": torch.sum(gt["ang_vel"] ** 2, dim=1) * cfg.ang_vel_scale,
            "alive": torch.full((self.num_envs,), cfg.alive_scale, device=self.device),
            # 07-25 ±1cm straightness: residual wobble (descent ±3.9 cm lateral / sway
            # ±1.5 cm heave, trace-measured) is cross-axis VELOCITY the position terms
            # barely grade — penalize it directly. Transit axis by phase: DESCEND/ASCEND
            # (0/2) move along z, so body-y speed is cross; SWAY (1/3) moves along y, so
            # world-z speed is cross. Linear |v| (not v²) so centimeter-scale wobble
            # still registers. Gated off while searching (spin makes body-y meaningless).
            "cross_vel": (
                torch.where(
                    self._scan_state.phase % 2 == 0,
                    gt["lin_vel"][:, 1].abs(),
                    gt["lin_vel_w"][:, 2].abs(),
                )
                * (~self._search_active).float()
                * cfg.cross_vel_scale
            ),
            # 07-26 real-scan pacing: linear penalty on ALONG-axis speed above
            # scan_speed_cap (complement of cross_vel — heave speed in DESCEND/ASCEND,
            # sway speed in SWAY). The 0.77 m/s sway dash heeled the hull ~8 deg via the
            # TAM My coupling; the real survey scans slower anyway.
            "overspeed": (
                (
                    torch.where(
                        self._scan_state.phase % 2 == 0,
                        gt["lin_vel_w"][:, 2].abs(),
                        gt["lin_vel"][:, 1].abs(),
                    )
                    - cfg.scan_speed_cap
                ).clamp(min=0.0)
                * (~self._search_active).float()
                * cfg.overspeed_scale
            ),
            # 07-28 posture fix: LINEAR tilt-angle penalty — the cos-based upright term is
            # flat near zero, which let the policy adopt a permanent 5-7 deg lean (heave
            # thrusters recruited laterally) at near-zero cost. See cfg.tilt_scale note.
            "tilt": torch.arccos(gt["up_vec"][:, 2].clamp(-1.0, 1.0)) * cfg.tilt_scale,
        }

    def _get_rewards(self) -> torch.Tensor:
        gt, sensors = self._read_state()

        # Marker-frame sway coordinate (07-25 refactor). The real UKF-M outputs an ABSOLUTE
        # pose in the ArUco-marker map frame, so sim uses ONE ruler for truth, estimate and
        # refs: arc length along the wall from the virtual marker. The old scheme mixed two
        # rulers — a DVL body-y integral (s/s_gt) vs a tank-center arc integral (s_ukfm) —
        # whose mismatch (22.5 cm median, trace-measured) dominated descent drift and made
        # tighter tracking rewards counterproductive (policy chased a bent ruler).
        theta_gt = torch.atan2(gt["pos"][:, 1], gt["pos"][:, 0])
        # CONTINUOUS (unwrapped) arc coordinate (07-27 full-loop): the direct wrapped
        # projection jumps by -2*pi*R at the marker antipode (+-18.85 m), breaking refs
        # half-way around the tank. Accumulating the GT angle INCREMENT keeps s continuous
        # over any number of loops and stays drift-free (exact telescoping of exact
        # angles; per-step motion << pi so the wrap never aliases).
        self._s_gt = self._s_gt + _wrap_to_pi(theta_gt - self._theta_prev) * self.cfg.tank_radius
        self._theta_prev = theta_gt
        # Estimate mirrors ukfm_localization.py: absolute marker fix when visible, DVL body-y
        # dead-reckoning otherwise. 0.2 blend = ~0.1 s time constant at 50 Hz — tracks the fix
        # while filtering its ~3 cm noise. No telescoped integration/prev-valid bookkeeping.
        uxy = sensors["ukfm_xy"]
        uvalid = sensors["ukfm_valid"].bool() & ~self._search_active
        s_meas_w = _wrap_to_pi(torch.atan2(uxy[:, 1], uxy[:, 0]) - self._marker_theta) * self.cfg.tank_radius
        # Freeze the estimate during the spin search (07-26 first-cycle settling): body-y
        # integration of a 2π spin is garbage and it leaked into the s_hat-s_ref observation;
        # s_hat restarts from 0 at the marker lock anyway.
        dead_reckon = self._s + sensors["lin_vel"][:, 1] * self.step_dt * (~self._search_active).float()
        # Correct toward the NEAREST wrap image of the (wrapped) marker fix, so the
        # continuous estimate never sees the antipode jump either (07-27 full-loop).
        fix_err = _wrap_to_pi((s_meas_w - dead_reckon) / self.cfg.tank_radius) * self.cfg.tank_radius
        self._s = torch.where(uvalid, dead_reckon + 0.2 * fix_err, dead_reckon)

        # Phase timing: depth from sensor (const bias << reach_eps, usable), but sway from GT
        # (_s_gt) — the DVL-integrated self._s accumulates bias*t drift (metres over a 120 s
        # episode), which made the sway reach criterion unobservable-conflicted and stalled the
        # scan at SWAY (07-23 train120: waypoint ~12/40, success 0). Reference generation is
        # env-side task definition (same precedent as z_latch); the policy still only OBSERVES
        # sensor values. z_latch stays GT for the z_hold reward reference.
        z_ref, s_ref, phase_sc, advanced = scan_state_machine.step(
            self._scan_state, sensors["depth"], self._s_gt, self.cfg.scan, z_latch=gt["pos"][:, 2]
        )

        # Spin-search gate: while active, sweep the yaw reference and track the sonar-min
        # bearing; the scan machine stays parked in DESCEND (no reach at the surface). On
        # sweep completion, lock _yaw_ref to the found bearing. (Measured sonar = what a real
        # search would use; GT heading for the bearing record, same env-side-reference
        # precedent as z_latch.)
        searching = self._search_active
        if searching.any():
            swept, bd, by, active = scan_state_machine.search_step(
                self._search_swept, self._search_best_dist, self._search_best_yaw,
                sensors["sonar"], gt["heading"], self.cfg.search_omega, self.step_dt,
            )
            self._search_swept = torch.where(searching, swept, self._search_swept)
            self._search_best_dist = torch.where(searching, bd, self._search_best_dist)
            self._search_best_yaw = torch.where(searching, by, self._search_best_yaw)
            self._search_ref = torch.where(
                searching, self._search_ref + self.cfg.search_omega * self.step_dt, self._search_ref
            )
            ended = searching & ~active
            self._yaw_ref = torch.where(ended, self._search_best_yaw, self._yaw_ref)
            # Virtual marker drop: at scan start anchor the map frame at the robot's current
            # wall angle — the real stack's "first marker fix defines the origin". s_gt is a
            # geometric projection off this angle (integral re-zeroing is gone, 07-24 s-pollution
            # bug class dissolved); only the estimate and shaping memory restart from 0.
            theta_now = torch.atan2(gt["pos"][:, 1], gt["pos"][:, 0])
            self._marker_theta = torch.where(ended, theta_now, self._marker_theta)
            self._s = torch.where(ended, torch.zeros_like(self._s), self._s)
            self._s_gt = torch.where(ended, torch.zeros_like(self._s_gt), self._s_gt)
            self._prev_s_gt = torch.where(ended, torch.zeros_like(self._prev_s_gt), self._prev_s_gt)
            # Ramp re-anchor: during the search the machine's ramp slewed toward the
            # bottom while the env pinned z_ref at the ceiling — restart the ramp from
            # where the scan actually begins (ceiling depth, s=0).
            ceiling_now = self.cfg.tank_height - 0.5 - 1.0
            self._scan_state.z_ramp = torch.where(
                ended, torch.full_like(self._scan_state.z_ramp, ceiling_now), self._scan_state.z_ramp
            )
            self._scan_state.s_ramp = torch.where(
                ended, torch.zeros_like(self._scan_state.s_ramp), self._scan_state.s_ramp
            )
            self._search_active = searching & active
            # While searching: hold depth at the operating ceiling — 1 m below the surface
            # (07-25: near the surface the camera gets too close to the ArUco marker and
            # ukfm/odom degrades on the real robot, so all z motion stays <= surface - 1 m).
            ceiling_z = self.cfg.tank_height - 0.5 - 1.0
            z_ref = torch.where(self._search_active, torch.full_like(z_ref, ceiling_z), z_ref)
        # Radial heading reference (07-27, replaces the search-locked bearing + s/R
        # feedforward). Full-loop yaw-vs-theta audit: the spin-search argmin locked the
        # bearing 5-84 deg off the true wall normal per episode (the policy TRANSLATES
        # during the spin chasing d_ref, biasing the distance minimum), and once locked
        # NOTHING corrects it — an oblique beam still reads exactly d_ref at a smaller
        # radius, so the vehicle crabbed a whole loop with zero reward loss. The wall
        # normal of a cylinder is simply the outward radial, so reference it directly
        # from position: rotation with curvature is then closed-loop (position turns ->
        # radial turns), needs no R feedforward, and cannot drift. GT position for the
        # reward reference (env-side task definition, same precedent as z_latch/s_gt);
        # the deployed controller computes the same bearing from the UKF-M position.
        self._yaw_ref_cur = torch.where(self._search_active, self._search_ref, theta_gt)

        self._z_ref, self._s_ref, self._phase_sc = z_ref, s_ref, phase_sc

        # A full cycle completes when the state machine wraps SWAY_B(3) -> DESCEND(0).
        self._cycles += (advanced & (self._scan_state.phase == 0)).long()

        terms = self._compute_reward_terms(gt, sensors, z_ref, s_ref, advanced)
        # Progress-shaping memory: update AFTER the terms consumed the previous values.
        self._prev_z = gt["pos"][:, 2].clone()
        self._prev_s_gt = self._s_gt.clone()
        for name, val in terms.items():
            self._episode_reward_sums[name] += val
        total = sum(terms.values())
        self._ep_return += total          # DORAEMON per-episode return (reset in _reset_idx)
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        pos_local = self._robot.data.root_pos_w - self.scene.env_origins

        # Cached for reuse in _compute_reward_terms (same GT pos, no physics step in between).
        self._clearance = geometry.radial_clearance(pos_local[:, :2], cfg.tank_radius)
        collided = (
            self._clearance < cfg.d_min if cfg.hard_collision_term else torch.zeros_like(self._clearance, dtype=torch.bool)
        )

        z = pos_local[:, 2]
        # Upper bound ABOVE the rim (+0.2): the water surface itself is the physical cap — the
        # robot cannot fly, so this is only a physics-blowup guard. The old tank_height-0.2
        # (9.8) killed every ASCEND overshoot: rollout traces show buoyancy+momentum carry the
        # climb to 9.6-9.7 past the z_top band, ending episodes one advance short of a cycle.
        # Floor bound mirrors the ceiling logic: the physical ground plane (z=0) stops the
        # robot, so this too is only a blowup guard. The old z<0.5 sat 5 cm under the widened
        # z_bottom reach band (0.55) and killed half of all episodes on descent overshoot
        # (stage3_fix: term_out_of_bounds=0.51, collided/tilted=0).
        out_of_bounds = (z < 0.15) | (z > cfg.tank_height + 0.2)

        up_b = quat_apply_inverse(self._robot.data.root_quat_w, self._up_w)
        tilted = up_b[:, 2] < 0.3

        success = self._cycles >= cfg.success_cycles

        # Cause masks kept for the reset-time telemetry breakdown.
        self._term_collided, self._term_oob, self._term_tilted = collided, out_of_bounds, tilted

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = collided | out_of_bounds | tilted | success
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self.extras["log"] = {}
        for name, buf in self._episode_reward_sums.items():
            self.extras["log"][f"Episode_Reward/{name}"] = buf[env_ids].mean().item()
            buf[env_ids] = 0.0
        self.extras["log"]["Episode_Termination/terminated"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        # Scan-progress telemetry: WHERE do episodes end? (diagnosing why cycles never complete)
        self.extras["log"]["Scan/end_phase_mean"] = self._scan_state.phase[env_ids].float().mean().item()
        self.extras["log"]["Scan/end_cycles_mean"] = self._cycles[env_ids].float().mean().item()
        self.extras["log"]["Scan/end_still_searching"] = self._search_active[env_ids].float().mean().item()
        self.extras["log"]["Scan/term_collided"] = torch.count_nonzero(self._term_collided[env_ids]).item()
        self.extras["log"]["Scan/term_out_of_bounds"] = torch.count_nonzero(self._term_oob[env_ids]).item()
        self.extras["log"]["Scan/term_tilted"] = torch.count_nonzero(self._term_tilted[env_ids]).item()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        self._thruster.reset(env_ids)
        self._hydro.reset(env_ids)

        if self.cfg.doraemon.enable:
            from . import doraemon_dr

            # Record finished episodes BEFORE self._cycles is zeroed below (spec §6 ordering).
            started = self._ep_started[env_ids]
            if started.any():
                ids = env_ids[started]
                success = (self._cycles[ids] >= self.cfg.doraemon.success_cycles).float()
                self._doraemon.record_episodes(
                    self._dr_xi[ids], self._ep_return[ids], success, self._dr_logp[ids]
                )
            xi, logp = self._doraemon.sample(len(env_ids))
            xi = xi.to(self.device)
            doraemon_dr.apply_xi(self._hydro, self._thruster, env_ids, xi)
            self._dr_xi[env_ids] = xi
            self._dr_logp[env_ids] = logp.to(self.device)
            self._ep_started[env_ids] = True
            # step() no-ops except every step_interval calls / below min_episodes.
            metrics = self._doraemon.step()
            self.extras["log"].update({f"DORAEMON/{k}": v for k, v in metrics.items()})
        elif self.cfg.randomization.enable:
            from ..bluerov.mdp.events import randomize_hydrodynamics

            r = self.cfg.randomization
            randomize_hydrodynamics(
                self,
                env_ids,
                added_mass_scale=r.added_mass_scale,
                linear_damping_scale=r.linear_damping_scale,
                quadratic_damping_scale=r.quadratic_damping_scale,
                volume_scale=r.volume_scale,
                cob_offset_x=r.cob_offset_x,
                cob_offset_y=r.cob_offset_y,
                cob_offset_z=r.cob_offset_z,
                cog_offset_x=r.cog_offset_x,
                cog_offset_y=r.cog_offset_y,
                cog_offset_z=r.cog_offset_z,
                inertia_scale=r.inertia_scale,
            )
            self._thruster.randomize_parameters(
                env_ids=env_ids,
                thrust_coeff_scale=r.thrust_coefficient_scale,
                time_constant_scale=r.time_constant_scale,
            )

        self._ep_return[env_ids] = 0.0

        # Reset waypoint state machine + sway estimate to phase 0 (DESCEND, s_ref=0).
        self._scan_state.phase[env_ids] = 0
        self._scan_state.s_ref[env_ids] = 0.0
        self._scan_state.sway_dir[env_ids] = 1.0
        self._scan_state.z_hold[env_ids] = 0.0
        self._scan_state._hold[env_ids] = 0
        self._scan_state.z_ramp[env_ids] = self.cfg.tank_height - 0.5  # spawn depth
        self._scan_state.s_ramp[env_ids] = 0.0
        self._s[env_ids] = 0.0
        self._s_gt[env_ids] = 0.0
        # Progress-shaping memory: seed with the spawn pose so step 1 sees zero progress.
        self._prev_z[env_ids] = self.cfg.tank_height - 0.5
        self._prev_s_gt[env_ids] = 0.0

        # Sonar mount extrinsics DR: perturb mount xy + beam azimuth per episode (nominal when off).
        _sc = self.cfg.sensors
        if self.cfg.randomization.enable and (_sc.sonar_mount_dr > 0.0 or _sc.sonar_yaw_dr > 0.0):
            _k = len(env_ids)
            self._sonar_mount[env_ids] = self._sonar_mount_nom[env_ids] + (
                torch.rand(_k, 2, device=self.device) * 2.0 - 1.0
            ) * _sc.sonar_mount_dr
            self._sonar_yaw[env_ids] = self._sonar_yaw_nom + (
                torch.rand(_k, device=self.device) * 2.0 - 1.0
            ) * _sc.sonar_yaw_dr
        else:
            self._sonar_mount[env_ids] = self._sonar_mount_nom[env_ids]
            self._sonar_yaw[env_ids] = self._sonar_yaw_nom

        # Per-episode sensor-bias DR (observation only). Uniform +-dr per channel; a knob of 0
        # yields zeros, so DR stages can enable a subset. Zeroed entirely when DR is off.
        sb = self._sensor_bias

        def _u(shape, dr):
            return (torch.rand(shape, device=self.device) * 2.0 - 1.0) * dr

        if self.cfg.randomization.enable:
            k = len(env_ids)
            sb["sonar"][env_ids] = _u((k,), _sc.sonar_bias_dr)
            sb["depth"][env_ids] = _u((k,), _sc.depth_bias_dr)
            sb["up_vec"][env_ids] = _u((k, 3), _sc.ins_att_bias_dr)
            # Heading gets its own half-range: roll/pitch are gravity-referenced (0.5 deg on a
            # 3DM-GV7) while heading leans on the magnetometer (2 deg, and questionable inside a
            # metal tank). sensors.heading_bias_dr falls back to ins_att_bias_dr when unset, so
            # this is a no-op for the legacy SensorCfg.
            sb["heading"][env_ids] = _u((k,), sensors_mod.heading_bias_dr(_sc))
            sb["ang_vel"][env_ids] = _u((k, 3), _sc.ins_gyro_bias_dr)
            sb["lin_vel"][env_ids] = _u((k, 3), _sc.dvl_bias_dr)
            # Multiplicative, per axis: sound-speed error is common-mode but beam-geometry
            # error is not, so drawing per axis is the conservative choice.
            sb["lin_vel_scale"][env_ids] = _u((k, 3), _sc.dvl_scale_dr)
            sb["ukfm_xy"][env_ids] = _u((k, 2), _sc.ukfm_bias_dr)
            sb["ukfm_yaw"][env_ids] = _u((k,), _sc.ukfm_bias_dr)
        else:
            for _v in sb.values():
                _v[env_ids] = 0.0

        self._z_ref[env_ids] = self.cfg.scan.z_bottom
        self._s_ref[env_ids] = 0.0
        self._phase_sc[env_ids, 0] = 0.0
        self._phase_sc[env_ids, 1] = 1.0
        self._cycles[env_ids] = 0

        # Randomize spawn position inside the tank (uniform-area disk sample, mid-range depth).
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        num_reset = len(env_ids)
        margin = 1.0
        max_r = max(self.cfg.tank_radius - margin, 0.1)
        radius = torch.sqrt(torch.rand(num_reset, device=self.device)) * max_r
        theta = torch.rand(num_reset, device=self.device) * 2.0 * math.pi
        origins = self.scene.env_origins[env_ids]
        default_root_state[:, 0] = origins[:, 0] + radius * torch.cos(theta)
        default_root_state[:, 1] = origins[:, 1] + radius * torch.sin(theta)
        # Spawn at the water surface (tank_height - 0.5): the real robot is positively buoyant
        # and starts every mission floating at the surface (UKF-M marker view also valid there).
        default_root_state[:, 2] = origins[:, 2] + self.cfg.tank_height - 0.5

        # Marker-frame placeholder until search end: anchor at the spawn wall angle so s_gt
        # starts near 0 during the spin search (a far-off placeholder would let the progress
        # shaping term mint a huge one-time penalty on step 1).
        self._marker_theta[env_ids] = theta
        self._theta_prev[env_ids] = theta  # s unwrapping restarts from the spawn angle

        spawn_yaw = torch.zeros(num_reset, device=self.device)
        if self.cfg.randomization.enable:
            r = self.cfg.randomization
            rpy_lo = torch.tensor([r.roll_range[0], r.pitch_range[0], r.yaw_range[0]], device=self.device)
            rpy_hi = torch.tensor([r.roll_range[1], r.pitch_range[1], r.yaw_range[1]], device=self.device)
            rpy = torch.rand(num_reset, 3, device=self.device) * (rpy_hi - rpy_lo) + rpy_lo
            default_root_state[:, 3:7] = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            spawn_yaw = rpy[:, 2]

        # Arm the spin-search: sweep starts from the spawn heading; refs hold there until the
        # nearest-wall bearing is locked at sweep completion (see _get_rewards search gate).
        self._search_active[env_ids] = True
        self._search_swept[env_ids] = 0.0
        self._search_best_dist[env_ids] = float("inf")
        self._search_best_yaw[env_ids] = spawn_yaw
        self._search_ref[env_ids] = spawn_yaw
        self._yaw_ref[env_ids] = spawn_yaw
        self._yaw_ref_cur[env_ids] = spawn_yaw

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
