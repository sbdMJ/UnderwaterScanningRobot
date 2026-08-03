# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for DORAEMON curriculum record & replay (marinelab.algorithms)."""

import dataclasses
import sys

import pytest
import torch


def _configclass_with_factory(cls):
    for name, value in list(vars(cls).items()):
        if name in getattr(cls, "__annotations__", {}) and isinstance(value, (dict, list, set)):
            default = value
            setattr(cls, name, dataclasses.field(default_factory=lambda d=default: type(d)(d)))
    return dataclasses.dataclass(cls)


sys.modules["isaaclab.utils"].configclass = _configclass_with_factory

from marinelab.algorithms.doraemon import (  # noqa: E402
    DoraemonCfg,
    DoraemonScheduler,
)

# Two simple DR params for tests: (name, field_name, lo, hi).
_PARAM_DEFS = [
    ("p0", "p0_range", 0.0, 1.0),
    ("p1", "p1_range", 0.0, 2.0),
]


class _FakeDRCfg:
    p0_range = (0.0, 1.0)
    p1_range = (0.0, 2.0)


def _make_scheduler(step_interval=2, min_episodes=4, **cfg_kw):
    cfg = DoraemonCfg(
        step_interval=step_interval,
        min_episodes=min_episodes,
        buffer_size=64,
        init_concentration=30.0,
        **cfg_kw,
    )
    return DoraemonScheduler(
        cfg, torch.device("cpu"), dr_cfg=_FakeDRCfg(), param_defs=_PARAM_DEFS
    )


def _fill_buffer(sch, n=8):
    xi, log_probs = sch.sample(n)
    returns = torch.full((n,), 100.0)
    success = torch.ones(n)
    sch.record_episodes(xi, returns, success, log_probs)


def test_record_appends_on_update():
    sch = _make_scheduler(step_interval=2, min_episodes=4)
    _fill_buffer(sch, 8)
    # _step_count starts at 0; step(iteration) called each RL iter.
    # Update fires when _step_count % step_interval == 0 AND buffer >= min_episodes.
    sch.step(iteration=0)  # _step_count 0 -> update fires, records
    n_after_0 = len(sch.export_trajectory())
    sch.step(iteration=1)  # _step_count 1 -> no update, no record
    n_after_1 = len(sch.export_trajectory())
    sch.step(iteration=2)  # _step_count 2 -> update fires, records
    n_after_2 = len(sch.export_trajectory())

    assert n_after_0 == 1
    assert n_after_1 == 1  # unchanged between updates
    assert n_after_2 == 2
    # Recorded iteration is the RL iteration passed in, not _step_count.
    assert sch.export_trajectory()[0]["iter"] == 0
    assert sch.export_trajectory()[1]["iter"] == 2
    # Each entry carries the full (a, b) vectors.
    assert len(sch.export_trajectory()[0]["a"]) == 2
    assert len(sch.export_trajectory()[0]["b"]) == 2


def test_export_recording_has_metadata():
    sch = _make_scheduler(step_interval=2, min_episodes=4)
    _fill_buffer(sch, 8)
    sch.step(iteration=0)
    rec = sch.export_recording()
    assert rec["param_names"] == ["p0", "p1"]
    assert rec["param_bounds"] == [[0.0, 1.0], [0.0, 2.0]]
    assert rec["trajectory"] == sch.export_trajectory()


from marinelab.algorithms.doraemon import CurriculumReplayer, ParamSpec  # noqa: E402


def _specs():
    return [ParamSpec("p0", 0.0, 1.0, 0.5), ParamSpec("p1", 0.0, 2.0, 1.0)]


def _recording(entries):
    return {
        "param_names": ["p0", "p1"],
        "param_bounds": [[0.0, 1.0], [0.0, 2.0]],
        "trajectory": entries,
    }


def test_replay_holds_last():
    rec = _recording([
        {"iter": 0, "a": [10.0, 10.0], "b": [10.0, 10.0]},
        {"iter": 4, "a": [20.0, 20.0], "b": [5.0, 5.0]},
    ])
    rep = CurriculumReplayer(rec, _specs(), torch.device("cpu"))

    rep.step(iteration=0)
    assert rep.dist._a.tolist() == [10.0, 10.0]
    rep.step(iteration=2)  # between 0 and 4 -> hold the iter-0 distribution
    assert rep.dist._a.tolist() == [10.0, 10.0]
    rep.step(iteration=4)  # exactly at next update
    assert rep.dist._a.tolist() == [20.0, 20.0]
    rep.step(iteration=999)  # past the end -> hold last
    assert rep.dist._a.tolist() == [20.0, 20.0]


def test_replay_record_episodes_noop():
    rec = _recording([{"iter": 0, "a": [10.0, 10.0], "b": [10.0, 10.0]}])
    rep = CurriculumReplayer(rec, _specs(), torch.device("cpu"))
    rep.step(iteration=0)
    before = rep.dist._a.clone()
    xi, lp = rep.sample(8)
    rep.record_episodes(xi, torch.ones(8), torch.ones(8), lp)  # must not change dist
    assert torch.equal(rep.dist._a, before)


def test_sample_interface_parity():
    rec = _recording([{"iter": 0, "a": [10.0, 10.0], "b": [10.0, 10.0]}])
    rep = CurriculumReplayer(rec, _specs(), torch.device("cpu"))
    sch = _make_scheduler()
    rep_xi, rep_lp = rep.sample(5)
    sch_xi, sch_lp = sch.sample(5)
    assert rep_xi.shape == sch_xi.shape == (5, 2)
    assert rep_lp.shape == sch_lp.shape == (5,)
    assert rep_xi.dtype == sch_xi.dtype


def test_export_trajectory_empty_for_replayer():
    rec = _recording([{"iter": 0, "a": [10.0, 10.0], "b": [10.0, 10.0]}])
    rep = CurriculumReplayer(rec, _specs(), torch.device("cpu"))
    assert rep.export_trajectory() == []


def test_scheduler_state_dict_round_trips_write_idx():
    # Regression guard: load_state_dict must restore the ring-buffer write pointer.
    # If _write_idx is not restored, a resumed run overwrites the buffer head and
    # corrupts DORAEMON's IS estimate.
    sch = _make_scheduler(step_interval=2, min_episodes=4)  # buffer_size=64
    _fill_buffer(sch, 6)  # write_idx advances to 6
    state = sch.state_dict()
    assert state["buffer_write_idx"] == 6

    fresh = _make_scheduler(step_interval=2, min_episodes=4)
    fresh.load_state_dict(state)
    assert fresh.buffer._write_idx == 6
    assert fresh.buffer._count == 6


def test_replay_matches_recording():
    # Record a real curriculum, then replay it; (a, b) must match bit-exact per update.
    sch = _make_scheduler(step_interval=2, min_episodes=4)
    for it in range(0, 6):
        _fill_buffer(sch, 8)
        sch.step(iteration=it)
    rec = sch.export_recording()
    assert len(rec["trajectory"]) >= 2

    specs = _specs()
    rep = CurriculumReplayer(rec, specs, torch.device("cpu"))
    for entry in rec["trajectory"]:
        rep.step(iteration=entry["iter"])
        assert rep.dist._a.tolist() == entry["a"]
        assert rep.dist._b.tolist() == entry["b"]


def test_dim_mismatch_raises():
    rec = _recording([{"iter": 0, "a": [10.0, 10.0], "b": [10.0, 10.0]}])
    three_specs = [
        ParamSpec("p0", 0.0, 1.0, 0.5),
        ParamSpec("p1", 0.0, 2.0, 1.0),
        ParamSpec("p2", 0.0, 1.0, 0.5),
    ]
    with pytest.raises(ValueError, match="dims"):
        CurriculumReplayer(rec, three_specs, torch.device("cpu"))


def test_bounds_mismatch_raises():
    rec = _recording([{"iter": 0, "a": [10.0, 10.0], "b": [10.0, 10.0]}])
    bad_specs = [ParamSpec("p0", 0.0, 1.0, 0.5), ParamSpec("p1", 0.0, 5.0, 1.0)]  # hi 2.0 -> 5.0
    with pytest.raises(ValueError, match="bounds mismatch"):
        CurriculumReplayer(rec, bad_specs, torch.device("cpu"))


def test_name_mismatch_raises():
    rec = _recording([{"iter": 0, "a": [10.0, 10.0], "b": [10.0, 10.0]}])
    wrong_name = [ParamSpec("p0", 0.0, 1.0, 0.5), ParamSpec("pX", 0.0, 2.0, 1.0)]  # p1 -> pX
    with pytest.raises(ValueError, match="name mismatch"):
        CurriculumReplayer(rec, wrong_name, torch.device("cpu"))


def test_record_to_json_to_replay(tmp_path):
    # End-to-end: record, serialize to JSON exactly as the runner does, reload, replay.
    import json as _json

    sch = _make_scheduler(step_interval=2, min_episodes=4)
    for it in range(0, 6):
        _fill_buffer(sch, 8)
        sch.step(iteration=it)
    recording = sch.export_recording()

    path = tmp_path / "curriculum_trajectory.json"
    path.write_text(_json.dumps(recording))
    loaded = _json.loads(path.read_text())

    rep = CurriculumReplayer(loaded, _specs(), torch.device("cpu"))
    for entry in loaded["trajectory"]:
        rep.step(iteration=entry["iter"])
        assert rep.dist._a.tolist() == entry["a"]
        assert rep.dist._b.tolist() == entry["b"]
    # sample() works after replay and respects physical bounds.
    xi, _ = rep.sample(16)
    assert xi.shape == (16, 2)
    assert (xi[:, 0] >= 0.0).all() and (xi[:, 0] <= 1.0).all()
    assert (xi[:, 1] >= 0.0).all() and (xi[:, 1] <= 2.0).all()


def test_replay_exposes_cfg_performance_lb():
    # Regression: the env reads self._doraemon.cfg.performance_lb in _log_and_reset_rewards
    # (to compute binary success before the no-op record_episodes). The replayer must expose
    # a cfg with performance_lb or the env crashes with AttributeError on the first reset.
    rec = _recording([{"iter": 0, "a": [10.0, 10.0], "b": [10.0, 10.0]}])
    rep = CurriculumReplayer(rec, _specs(), torch.device("cpu"))
    assert hasattr(rep, "cfg")
    assert hasattr(rep.cfg, "performance_lb")
    # Default falls back to DoraemonCfg's default performance_lb.
    assert rep.cfg.performance_lb == DoraemonCfg().performance_lb


def test_replay_preserves_injected_cfg_performance_lb():
    # When the recording run used a non-default J_LB (e.g. 90.0), the replay should carry it,
    # so the (discarded) success computation matches the recording run's threshold.
    rec = _recording([{"iter": 0, "a": [10.0, 10.0], "b": [10.0, 10.0]}])
    cfg = DoraemonCfg(performance_lb=90.0)
    rep = CurriculumReplayer(rec, _specs(), torch.device("cpu"), cfg=cfg)
    assert rep.cfg.performance_lb == 90.0
