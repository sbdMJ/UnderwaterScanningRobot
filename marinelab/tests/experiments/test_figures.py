# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Figures F1–F6 render from synthetic artifacts and land on disk (png + pdf)."""

import json

import numpy as np
import pytest

from marinelab.experiments import figures as F


def _stat(mean, sd=0.1, values=None):
    return {"mean": mean, "sd": sd, "values": values if values is not None else [mean - sd, mean, mean + sd]}


def _summary(metric="score.objective", conds=("nominal",)):
    out = {}
    for c in conds:
        out[("nominal", c)] = {"n_trials": 3, metric: _stat(5.0)}
        out[("diff", c)] = {"n_trials": 3, metric: _stat(3.0)}
    return out


def _assert_written(paths):
    assert len(paths) == 2
    for p in paths:
        assert p.endswith((".png", ".pdf"))
        import os
        assert os.path.getsize(p) > 1000


def test_f1_overlay(tmp_path):
    summary = _summary()
    summary[("nominal", "nominal")]["cycles_mean"] = _stat(2.0)
    summary[("diff", "nominal")]["cycles_mean"] = _stat(2.4)
    _assert_written(F.fig_overlay(summary, ["score.objective", "cycles_mean"],
                                  str(tmp_path / "f1")))


def test_f1_handles_inf(tmp_path):
    summary = _summary()
    summary[("nominal", "nominal")]["score.objective"] = _stat(float("inf"), values=[float("inf")])
    _assert_written(F.fig_overlay(summary, ["score.objective"], str(tmp_path / "f1")))


def _write_traj(path, steps=100, n_env=2):
    t = np.linspace(0, 2 * np.pi, steps)
    arr = np.tile(np.sin(t)[:, None], (1, n_env))
    np.savez(path, s_gt=arr, z=5 + arr, s_ref=arr * 0.9, z_ref=5 + arr * 0.9,
             wall_dist=1.5 + 0.1 * arr, tilt_deg=np.abs(arr) * 3)


def test_f2_trajectory(tmp_path):
    _write_traj(tmp_path / "a.npz")
    _write_traj(tmp_path / "b.npz")
    _assert_written(F.fig_trajectory(
        [("nominal", "Nominal NMPC", str(tmp_path / "a.npz")),
         ("diff", "Diff-WMPC (ours)", str(tmp_path / "b.npz"))],
        str(tmp_path / "f2")))


def test_f3_sweep_and_level_parse(tmp_path):
    assert F.cond_level("dr25") == 25.0
    with pytest.raises(ValueError):
        F.cond_level("nominal")
    summary = _summary(conds=("dr25", "dr50", "dr75"))
    _assert_written(F.fig_sweep(summary, "score.objective", str(tmp_path / "f3")))


def test_f4_timeseries_multi_seed_band(tmp_path):
    _write_traj(tmp_path / "a.npz")
    _write_traj(tmp_path / "b.npz", steps=90)  # shorter run: lengths are aligned to the min
    meta = tmp_path / "a.json"
    meta.write_text(json.dumps({"step_dt": 0.02, "d_ref_m": 1.5}))
    _assert_written(F.fig_timeseries(
        [("nominal", "Nominal NMPC", [str(tmp_path / "a.npz"), str(tmp_path / "b.npz")],
          str(meta))],
        str(tmp_path / "f4"), t_event=0.5))


def test_f5_zeroshot_ft(tmp_path):
    named = {"zero-shot": _summary(), "fine-tuned": _summary()}
    named["fine-tuned"][("diff", "nominal")]["score.objective"] = _stat(2.0)
    _assert_written(F.fig_zeroshot_ft(named, "score.objective", str(tmp_path / "f5")))


def test_f7_states(tmp_path):
    _write_traj(tmp_path / "a.npz")
    meta = tmp_path / "a.json"
    meta.write_text(json.dumps({"step_dt": 0.02, "d_ref_m": 1.5}))
    _assert_written(F.fig_states(
        [("nominal", "Nominal NMPC", str(tmp_path / "a.npz"), str(meta)),
         ("diff", "Diff-WMPC (ours)", str(tmp_path / "a.npz"), str(meta))],
        str(tmp_path / "f7")))


def test_f8_task_geometry(tmp_path):
    _assert_written(F.fig_task(str(tmp_path / "f8")))


def test_f9_pred_error(tmp_path):
    steps = 200
    decay = 0.5 * np.exp(-np.arange(steps) / 40.0) + 0.02
    np.savez(tmp_path / "ssi.npz", aux_ssi_pred_err=decay[:, None])
    _write_traj(tmp_path / "plain.npz")  # no aux channel: silently skipped
    meta = tmp_path / "m.json"
    meta.write_text(json.dumps({"step_dt": 0.02, "d_ref_m": 1.5}))
    _assert_written(F.fig_pred_error(
        [("ssi", "SSI-MPC", [str(tmp_path / "ssi.npz")], str(meta)),
         ("nominal", "Nominal NMPC", [str(tmp_path / "plain.npz")], str(meta))],
        str(tmp_path / "f9")))
    with pytest.raises(ValueError):
        F.fig_pred_error([("nominal", "n", [str(tmp_path / "plain.npz")], str(meta))],
                         str(tmp_path / "f9b"))


def test_f10_sensitivity(tmp_path):
    rows = []
    for lr in (0.01, 0.1, 1.0):
        for v in (3.0, 3.2):
            rows.append({"options.ssi_lr": lr, "options.ssi_n_rf": 100,
                         "score.objective": v + lr})
    for m in (25, 100, 500):
        rows.append({"options.ssi_lr": 0.1, "options.ssi_n_rf": m,
                     "score.objective": 3.0 + 10.0 / m})
    _assert_written(F.fig_sensitivity(rows, str(tmp_path / "f10")))


def test_f6_cost(tmp_path):
    offline = {"bo": {"wall_clock_s": 1000.0, "env_steps": 450000},
               "ssi": {"wall_clock_s": 900.0, "env_steps": 450000},
               "ppo": {"wall_clock_s": 90000.0, "env_steps": 8e8}}
    summary = _summary("controller_cost.solve_ms_mean")
    _assert_written(F.fig_cost(offline, summary, str(tmp_path / "f6")))
