# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the wallscan evaluation metrics.

Synthetic trajectories with hand-computable answers, so a regression in the metric
math is caught without booting Isaac Sim (conftest.py stubs isaaclab).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from marinelab.tasks.pkrc_wallscan import eval_metrics as em
from marinelab.tasks.pkrc_wallscan.scan_state_machine import ASCEND, DESCEND, SWAY_A, SWAY_B

STEP_DT = 0.02


def _row(n=1, **over):
    """One recordable row of n envs; every field defaults to zeros."""
    row = {f: torch.zeros(n) for f in em.FIELDS}
    for k, v in over.items():
        row[k] = v if torch.is_tensor(v) else torch.full((n,), float(v))
    return row


def _log(rows):
    log = em.TrajectoryLog()
    for r in rows:
        log.record(**r)
    return log.as_arrays(step_dt=STEP_DT)


# --------------------------------------------------------------------------- log


def test_record_rejects_missing_and_unknown_fields():
    log = em.TrajectoryLog()
    with pytest.raises(ValueError, match="missing"):
        log.record(z=torch.zeros(1))
    with pytest.raises(ValueError, match="unexpected"):
        log.record(**_row(1), bogus=torch.zeros(1))


def test_empty_log_raises():
    with pytest.raises(ValueError, match="empty"):
        em.TrajectoryLog().as_arrays(step_dt=STEP_DT)


def test_shapes_and_time_axis():
    traj = _log([_row(3) for _ in range(5)])
    assert traj["z"].shape == (5, 3)
    assert traj["t"].shape == (5, 3)
    np.testing.assert_allclose(traj["t"][:, 0], np.arange(5) * STEP_DT)


def test_two_episode_indices_for_the_two_questions_a_done_row_answers():
    """The done row is post-reset STATE but pre-reset OUTCOME.

    Isaac Lab resets terminated envs inside ``step()`` and every logger here reads the state
    after ``step()`` returns, so the row carrying ``done`` already holds the next episode's
    pose: ``episode`` must count it as new. Its ``terminated``/``time_out``/``term_*`` flags
    and the step it consumed still belong to the episode that finished, which is what
    ``ended_episode`` is for. Until 2026-07-31 only the second form existed and was used for
    both, leaking one post-reset row into every scored episode.
    """
    rows = [_row(1), _row(1, done=1.0), _row(1), _row(1)]
    traj = _log(rows)
    np.testing.assert_array_equal(traj["episode"][:, 0], [0, 1, 1, 1])
    np.testing.assert_array_equal(traj["ended_episode"][:, 0], [0, 0, 1, 1])


def test_episode_indices_helper_matches_the_logged_arrays():
    done = np.array([[0.0], [0.0], [1.0], [0.0], [1.0]])
    ep, ended = em.episode_indices(done)
    np.testing.assert_array_equal(ep[:, 0], [0, 0, 1, 1, 2])
    np.testing.assert_array_equal(ended[:, 0], [0, 0, 0, 1, 1])


def test_metrics_recompute_indices_so_stale_npz_files_score_correctly():
    """A .npz written before the fix carries the old `episode` array; it must be ignored."""
    rows = [_row(1, z=5.0), _row(1, z=9.0, done=1.0), _row(1, z=9.0)]
    traj = _log(rows)
    traj["episode"] = np.array([[0], [0], [1]], np.int64)  # the pre-fix value
    m = em.compute_metrics(traj, step_dt=STEP_DT, episode=0)
    assert m["scored_steps"] == 1, "the post-reset row must not be scored as episode 0"


# ------------------------------------------------------------------------ tilt


def test_tilt_split_by_leg():
    rows = [
        _row(1, phase=DESCEND, tilt_deg=1.0),
        _row(1, phase=ASCEND, tilt_deg=3.0),
        _row(1, phase=SWAY_A, tilt_deg=10.0),
        _row(1, phase=SWAY_B, tilt_deg=20.0),
    ]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT)
    assert m["tilt_heave_deg"] == pytest.approx(2.0)   # (1 + 3) / 2
    assert m["tilt_sway_deg"] == pytest.approx(15.0)   # (10 + 20) / 2
    assert m["tilt_deg"] == pytest.approx(8.5)         # all four


def test_searching_rows_are_excluded():
    rows = [
        _row(1, phase=DESCEND, tilt_deg=99.0, searching=1.0),  # dropped
        _row(1, phase=DESCEND, tilt_deg=2.0),
    ]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT)
    assert m["tilt_heave_deg"] == pytest.approx(2.0)
    assert m["scored_steps"] == 1


# ------------------------------------------------------------------------ crab


def test_crab_is_absolute_wrapped_yaw_minus_theta():
    # yaw - theta = -3pi/2 wraps to +pi/2 -> 90 deg, not 270.
    rows = [_row(1, yaw=-math.pi, theta=math.pi / 2)]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT)
    assert m["crab_deg"] == pytest.approx(90.0)


def test_crab_zero_when_yaw_tracks_theta():
    rows = [_row(1, yaw=a, theta=a) for a in (0.3, 1.7, -2.9)]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT)
    assert m["crab_deg"] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------- s_hat error


def test_s_hat_error_in_cm():
    rows = [_row(1, s=1.10, s_gt=1.00), _row(1, s=0.98, s_gt=1.00)]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT)
    assert m["s_hat_err_cm"] == pytest.approx(6.0)  # (10 cm + 2 cm) / 2


# ---------------------------------------------------------------------- speeds


def test_heave_speed_from_z_within_a_phase():
    # 0.004 m per 0.02 s step = 0.2 m/s, the cfg ramp rate. The leg is bracketed by SWAY rows
    # because only legs bounded by real phase transitions are scored (2026-07-31): an unbounded
    # run began or ended outside the log, and averaging over it is what produced a phantom 16%
    # heave overshoot.
    zs = [8.0 - 0.004 * i for i in range(4)]
    rows = ([_row(1, phase=SWAY_A, z=8.0)]
            + [_row(1, phase=DESCEND, z=z) for z in zs]
            + [_row(1, phase=SWAY_A, z=zs[-1])])
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, heave_target=0.2)
    assert m["heave_speed_mps"] == pytest.approx(0.2, rel=1e-3)
    assert m["heave_speed_target_mps"] == pytest.approx(0.2)
    assert m["heave_legs_scored"] == 1


def test_sway_speed_uses_ground_truth_arc_length():
    ss = [0.0 + 0.002 * i for i in range(4)]  # 0.1 m/s
    rows = ([_row(1, phase=DESCEND, s_gt=0.0)]
            + [_row(1, phase=SWAY_A, s_gt=s) for s in ss]
            + [_row(1, phase=DESCEND, s_gt=ss[-1])])
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT)
    assert m["sway_speed_mps"] == pytest.approx(0.1, rel=1e-3)
    assert m["sway_legs_scored"] == 1


def test_speed_ignores_phase_change_and_reset_jumps():
    # Metre-scale z jumps sit exactly on the phase boundaries. Legs never span a boundary by
    # construction now, so this is structural rather than a special case -- but it is the
    # property the whole metric rests on, so it stays pinned.
    rows = [
        _row(1, phase=SWAY_A, z=3.000),    # leading run: unbounded, dropped
        _row(1, phase=DESCEND, z=8.000),   # the scored leg: 0.008 m over 0.04 s = 0.2 m/s
        _row(1, phase=DESCEND, z=7.996),
        _row(1, phase=DESCEND, z=7.992),
        _row(1, phase=SWAY_A, z=1.000),    # 7 m jump at the boundary
        _row(1, phase=SWAY_A, z=1.000),
        _row(1, phase=DESCEND, z=9.000),   # trailing run: unbounded, dropped
    ]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=None)
    assert m["heave_speed_mps"] == pytest.approx(0.2, rel=1e-3)
    assert m["heave_legs_scored"] == 1


def test_a_leg_touching_an_episode_boundary_is_dropped_on_both_sides():
    """A reset ends one leg early AND starts the next one from a teleport.

    Neither side is a phase transition, so neither is scorable, and the 7 m reset jump can
    never be read as motion. This is deliberately conservative: a post-reset phase-0 run is
    in fact a genuine leg start in this task, but resolving that needs knowledge of what the
    reset did to the state machine, and it costs nothing in practice because
    ``compute_metrics`` scores one episode at a time anyway.
    """
    rows = [
        _row(1, phase=SWAY_A, z=5.0),
        _row(1, phase=DESCEND, z=8.000),
        _row(1, phase=DESCEND, z=7.996, done=1.0),
        _row(1, phase=DESCEND, z=1.000),   # new episode: 7 m apparent jump
        _row(1, phase=DESCEND, z=0.996),
        _row(1, phase=SWAY_A, z=0.996),
    ]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=None)
    assert m["heave_legs_scored"] == 0
    assert math.isnan(m["heave_speed_mps"])


def test_speeds_are_nan_when_no_valid_pair_exists():
    m = em.compute_metrics(_log([_row(1, phase=DESCEND)]), step_dt=STEP_DT)
    assert math.isnan(m["heave_speed_mps"])
    assert math.isnan(m["sway_speed_mps"])


# ---------------------------------------------------------------------- cycles


def test_cycles_taken_from_last_in_scope_row_per_env():
    rows = [
        _row(2, cycles=torch.tensor([0.0, 1.0])),
        _row(2, cycles=torch.tensor([2.0, 3.0])),
    ]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT)
    assert m["cycles_per_env"] == [2.0, 3.0]
    assert m["cycles_mean"] == pytest.approx(2.5)


def test_first_episode_scoring_ignores_later_episodes():
    """Cycles come from the last PRE-reset row of the episode.

    The row carrying ``done`` already holds post-reset state (Isaac Lab resets inside
    ``step()``), so it belongs to the next episode — and that is what makes this correct for
    both loggers: ``play.py`` records the env's own ``_cycles``, which ``_reset_idx`` has
    already zeroed by then, so crediting that row to episode 0 would have reported 0 cycles.
    """
    rows = [
        _row(1, cycles=2.0),            # last pre-reset row of episode 0
        _row(1, cycles=0.0, done=1.0),  # post-reset state + a zeroed counter
        _row(1, cycles=99.0),           # episode 1 — must not be scored
    ]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=0)
    assert m["cycles_per_env"] == [2.0], "the zeroed done row must not overwrite the total"
    m_all = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=None)
    assert m_all["cycles_per_env"] == [99.0]


def test_episode_length_counts_the_step_that_ended_it():
    rows = [_row(1) for _ in range(3)] + [_row(1, done=1.0)] + [_row(1)]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=0)
    assert m["episode_steps_mean"] == pytest.approx(4.0), "4 steps were taken in episode 0"
    assert m["scored_steps"] == 3, "but only 3 rows hold episode-0 STATE"


def test_termination_flags_are_read_from_the_ended_episode():
    rows = [_row(1), _row(1, done=1.0, time_out=1.0), _row(1)]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=0)
    assert m["terminations"]["total"] == 1
    assert m["terminations"]["time_out"] == 1


# ----------------------------------------------------------------- wall / terms


def test_wall_distance_error_against_d_ref():
    rows = [_row(1, wall_dist=1.60), _row(1, wall_dist=1.45)]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, d_ref=1.5)
    assert m["wall_dist_err_cm"] == pytest.approx(7.5)  # (10 + 5) / 2


def test_termination_breakdown_counted_on_done_rows_only():
    rows = [
        _row(1, term_collided=1.0),                                    # flagged, but not a done row
        _row(1, done=1.0, terminated=1.0, term_collided=1.0),
        _row(1, done=1.0, terminated=1.0, term_tilted=1.0),
    ]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=None)
    assert m["terminations"] == {
        "total": 2, "time_out": 0, "collided": 1, "out_of_bounds": 0, "tilted": 1, "success": 0,
    }


def test_success_termination_is_a_terminated_row_with_no_cause_flag():
    # _get_dones builds terminated as collided|oob|tilted|success but records no success
    # mask, so "terminated with nothing flagged" is how a success shows up.
    rows = [_row(1, done=1.0, terminated=1.0)]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=None)
    assert m["terminations"]["success"] == 1
    assert m["terminations"]["time_out"] == 0


def test_time_out_is_separated_from_terminated():
    rows = [_row(1, done=1.0, time_out=1.0)]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=None)
    assert m["terminations"] == {
        "total": 1, "time_out": 1, "collided": 0, "out_of_bounds": 0, "tilted": 0, "success": 0,
    }


def test_terminations_still_counted_when_the_done_row_is_already_post_reset():
    # Regression: Isaac Lab resets inside step(), so a done row carries the NEXT episode's
    # searching=1 while term_* still describe the episode that ended. Filtering the
    # breakdown on `searching` hid every termination.
    rows = [
        _row(1, phase=DESCEND, tilt_deg=5.0),
        _row(1, done=1.0, terminated=1.0, term_oob=1.0, searching=1.0),
    ]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=0)
    assert m["terminations"]["total"] == 1
    assert m["terminations"]["out_of_bounds"] == 1
    # ...while the physical metrics still ignore the searching row.
    assert m["tilt_heave_deg"] == pytest.approx(5.0)


def test_episode_duration_is_reported():
    rows = [_row(2) for _ in range(3)] + [_row(2, done=1.0)] + [_row(2) for _ in range(2)]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=0)
    assert m["episode_steps_mean"] == pytest.approx(4.0)          # rows 0..3 inclusive
    assert m["episode_s_mean"] == pytest.approx(4.0 * STEP_DT)


def test_metrics_report_env_and_step_counts():
    m = em.compute_metrics(_log([_row(4) for _ in range(7)]), step_dt=STEP_DT)
    assert m["num_envs"] == 4 and m["num_steps"] == 7
    assert m["step_dt"] == pytest.approx(STEP_DT)


# ---------------------------------------------------------------------- format


def test_format_metrics_renders_all_rows():
    m = em.compute_metrics(_log([_row(2, phase=DESCEND, z=1.0) for _ in range(2)]), step_dt=STEP_DT, d_ref=1.5)
    text = em.format_metrics(m)
    for label in ("tilt (heave / sway)", "scan speed", "crab audit", "arc-length", "scan cycles", "terminations"):
        assert label in text


def test_format_metrics_handles_nan_as_na():
    m = em.compute_metrics(_log([_row(1, phase=SWAY_A)]), step_dt=STEP_DT)
    assert "n/a" in em.format_metrics(m)  # heave leg never entered


# ------------------------------------------------------------------------- plot


def test_plot_trajectory_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    rows = [_row(2, phase=DESCEND, z=8.0 - 0.01 * i, s_gt=0.0, s_ref=0.0, z_ref=1.0) for i in range(20)]
    path = em.plot_trajectory(_log(rows), str(tmp_path / "traj.png"))
    assert path.endswith("traj.png")
    assert (tmp_path / "traj.png").stat().st_size > 0


def test_truncation_warning_fires_for_a_partial_episode():
    # wallscan randomises episode_length_buf on the first reset, so episode 0 is partial.
    rows = [_row(1) for _ in range(50)] + [_row(1, done=1.0, time_out=1.0)]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=0, episode_length_s=180.0)
    assert m["scored_episode_truncated"] is True
    assert "PARTIAL episode" in em.format_metrics(m)


def test_no_truncation_warning_for_a_full_episode():
    n = int(round(180.0 / STEP_DT))
    rows = [_row(1) for _ in range(n)]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=0, episode_length_s=180.0)
    assert m["scored_episode_truncated"] is False
    assert "PARTIAL episode" not in em.format_metrics(m)


# ---------------------------------------------------------------------------
# Leg speeds: net rate over COMPLETE legs (fixed 2026-07-31)
# ---------------------------------------------------------------------------


def _leg_traj(phases, z, n_env=1):
    """Minimal trajectory with a given phase sequence and z channel."""
    import numpy as np

    T = len(phases)
    ph = np.asarray(phases, float).reshape(T, n_env)
    return {
        "phase": ph,
        "episode": np.zeros((T, n_env), np.int64),
        "searching": np.zeros((T, n_env)),
        "z": np.asarray(z, float).reshape(T, n_env),
        "s_gt": np.zeros((T, n_env)),
    }


def _sel(traj):
    from marinelab.tasks.pkrc_wallscan.eval_metrics import _select

    return _select(traj, None)


def test_leg_speed_ignores_the_truncated_final_leg():
    """The bug: a settling leg cut off at the log end reported phantom speed.

    Leg 1 (dropped, partial) and leg 3 (dropped, truncated) both move fast; only the middle,
    properly bounded leg should be scored.
    """
    from marinelab.tasks.pkrc_wallscan.eval_metrics import _leg_speed

    dt = 0.1
    # phase 1 (sway, 3 rows) | phase 0 (descend, 6 rows) | phase 1 (sway, 3 rows)
    phases = [1, 1, 1] + [0] * 6 + [1, 1, 1]
    z = [9.0, 8.0, 7.0] + [6.0, 5.9, 5.8, 5.7, 5.6, 5.5] + [0.0, 5.0, 0.0]
    traj = _leg_traj(phases, z)
    v = _leg_speed(traj, "z", (0, 2), _sel(traj), dt)
    # scored leg: 0.5 m over 5 intervals * 0.1 s = 1.0 m/s
    assert v == pytest.approx(1.0, rel=1e-6)


def test_leg_speed_uses_net_displacement_not_path_length():
    """Oscillation must not be counted as progress — that is what inflated 0.196 to 0.233."""
    from marinelab.tasks.pkrc_wallscan.eval_metrics import _leg_speed

    dt = 0.1
    # a descend leg that ripples: net 0.4 m down, path length 1.0 m
    phases = [1] + [0] * 6 + [1]
    z = [0.0] + [5.0, 4.7, 4.9, 4.6, 4.8, 4.6] + [0.0]
    traj = _leg_traj(phases, z)
    sel = _sel(traj)
    net = _leg_speed(traj, "z", (0, 2), sel, dt)
    path = _leg_speed(traj, "z", (0, 2), sel, dt, path_length=True)
    assert net == pytest.approx(abs(4.6 - 5.0) / (5 * dt), rel=1e-6)
    assert path > net * 2, "path-length rate must expose the ripple the net rate hides"


def test_leg_speed_averages_several_legs_by_time():
    from marinelab.tasks.pkrc_wallscan.eval_metrics import _leg_speed

    dt = 0.1
    phases = [1] + [0] * 3 + [1] + [2] * 6 + [1]
    z = [0.0] + [3.0, 2.9, 2.8] + [0.0] + [1.0, 1.1, 1.2, 1.3, 1.4, 1.5] + [0.0]
    traj = _leg_traj(phases, z)
    v = _leg_speed(traj, "z", (0, 2), _sel(traj), dt)
    # 0.2 m over 0.2 s, plus 0.5 m over 0.5 s -> 0.7 / 0.7 = 1.0 m/s
    assert v == pytest.approx(1.0, rel=1e-6)


def test_leg_count_reports_what_was_scored():
    from marinelab.tasks.pkrc_wallscan.eval_metrics import _leg_count

    phases = [1] + [0] * 3 + [1, 1] + [2] * 3 + [1]
    traj = _leg_traj(phases, list(range(len(phases))))
    assert _leg_count(traj, (0, 2), _sel(traj)) == 2
    # the middle sway run is bounded and 2 rows long, so it yields one interval
    assert _leg_count(traj, (1, 3), _sel(traj)) == 1


def test_leg_speed_is_nan_when_no_complete_leg_exists():
    from marinelab.tasks.pkrc_wallscan.eval_metrics import _leg_speed

    traj = _leg_traj([0] * 5, [5.0, 4.9, 4.8, 4.7, 4.6])
    assert math.isnan(_leg_speed(traj, "z", (0, 2), _sel(traj), 0.1))


def test_leg_speed_skips_legs_containing_excluded_rows():
    """A leg overlapping the spin search must not be scored on its remaining rows."""
    from marinelab.tasks.pkrc_wallscan.eval_metrics import _leg_speed

    phases = [1] + [0] * 4 + [1]
    traj = _leg_traj(phases, [0.0, 5.0, 4.0, 3.0, 2.0, 0.0])
    traj["searching"][2, 0] = 1.0
    assert math.isnan(_leg_speed(traj, "z", (0, 2), _sel(traj), 0.1))


def test_a_leg_truncated_by_a_timeout_is_dropped_even_when_a_stub_run_follows():
    """Regression for the first (wrong) version of the completeness rule.

    A time-out on the final logged row splits off a one-row run, so the genuinely truncated
    leg before it is no longer the LAST run. Judging completeness by run index therefore let
    it through; judging by boundary KIND does not.
    """
    rows = [
        _row(1, phase=SWAY_A, z=5.0),
        _row(1, phase=DESCEND, z=8.000),   # complete leg: 0.2 m/s
        _row(1, phase=DESCEND, z=7.996),
        _row(1, phase=DESCEND, z=7.992),
        _row(1, phase=SWAY_A, z=7.992),
        _row(1, phase=DESCEND, z=7.000),   # truncated leg: ends at a reset, not a transition
        _row(1, phase=DESCEND, z=6.000, done=1.0),
        _row(1, phase=DESCEND, z=1.000),   # the one-row stub after the reset
    ]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=None)
    assert m["heave_legs_scored"] == 1, "only the bounded leg may be scored"
    assert m["heave_speed_mps"] == pytest.approx(0.2, rel=1e-3)


# ---------------------------------------------------------------------------
# Arc-length TRACKING error (coverage), added 2026-08-08.
#
# Distinct from s_hat_err (estimate vs truth). A run can have a perfect estimator and still be
# scanning a metre off the planned arc, which is exactly what stress-DR runs were doing while
# every other metric looked healthy.
# ---------------------------------------------------------------------------
def test_s_track_err_is_reference_tracking_not_estimator_error():
    # estimator PERFECT (s == s_gt) but the vehicle sits 1 m off the arc the scan asked for
    tr = _log([_row(s=2.0, s_gt=2.0, s_ref=1.0) for _ in range(200)])
    m = em.compute_metrics(tr, step_dt=STEP_DT, episode=None)
    assert m["s_hat_err_cm"] == pytest.approx(0.0, abs=1e-6), "estimator error must stay zero"
    assert m["s_track_err_cm"] == pytest.approx(100.0, rel=1e-6), "tracking error must see the 1 m"


def test_s_track_reports_tail_statistics():
    rows = [_row(s=0.0, s_gt=0.0, s_ref=0.0) for _ in range(95)]
    rows += [_row(s=4.0, s_gt=4.0, s_ref=0.0) for _ in range(5)]   # short, large excursion
    tr = _log(rows)
    m = em.compute_metrics(tr, step_dt=STEP_DT, episode=None)
    assert m["s_track_err_cm"] == pytest.approx(20.0, rel=1e-6)      # mean 0.2 m
    assert m["s_track_max_cm"] == pytest.approx(400.0, rel=1e-6)     # the mean hides this
    assert m["s_track_rms_cm"] > m["s_track_err_cm"]


def test_heave_drift_is_zero_for_a_pure_vertical_leg_and_sees_tangential_motion():
    """Reference-free coverage: during DESCEND/ASCEND the plan says hold the arc position, so
    the ideal is zero for every controller regardless of the reference it generated."""
    # the heave leg must be bounded by a phase change on BOTH sides, or _legs drops it as
    # incomplete (boundary KIND, not run index -- see the leg-completeness note in this file)
    clean = [_row(phase=3, z=9.0, s_gt=2.0) for _ in range(50)]
    clean += [_row(phase=0, z=9.0 - 0.004 * i, s_gt=2.0) for i in range(300)]
    clean += [_row(phase=1, z=7.8, s_gt=2.0 + 0.002 * i) for i in range(100)]
    m = em.compute_metrics(_log(clean), step_dt=STEP_DT, episode=None)
    assert m["heave_drift_cm"] == pytest.approx(0.0, abs=1e-6)

    drift = [_row(phase=3, z=9.0, s_gt=2.0) for _ in range(50)]
    drift += [_row(phase=0, z=9.0 - 0.004 * i, s_gt=2.0 + 0.003 * i) for i in range(300)]
    drift += [_row(phase=1, z=7.8, s_gt=2.9) for _ in range(100)]
    m2 = em.compute_metrics(_log(drift), step_dt=STEP_DT, episode=None)
    assert m2["heave_drift_cm"] > 80.0, "0.9 m of tangential drift must show up"


def test_sway_step_error_measures_advance_against_the_planned_step():
    rows = [_row(phase=0, z=9.0 - 0.004 * i, s_gt=0.0) for i in range(200)]
    rows += [_row(phase=1, z=8.2, s_gt=0.006 * i) for i in range(200)]   # advances 1.194 m
    rows += [_row(phase=2, z=8.2 + 0.004 * i, s_gt=1.194) for i in range(200)]
    m = em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=None, sway_step=1.0)
    assert m["sway_step_err_cm"] == pytest.approx(19.4, abs=1.0)
    assert np.isnan(em.compute_metrics(_log(rows), step_dt=STEP_DT, episode=None)["sway_step_err_cm"])
