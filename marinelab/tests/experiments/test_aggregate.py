# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Metrics collection, grouping statistics, and table output."""

import csv
import json

import pytest

from marinelab.experiments.aggregate import (
    collect, collect_budgets, flatten, summarize, write_table_csv,
)


def _write_metrics(root, method, cond, seed, cycles, solve_ms):
    payload = {
        "cycles_mean": cycles,
        "terminations": {"collided": 0},
        "controller_cost": {"solve_ms_mean": solve_ms},
    }
    (root / f"metrics_{method}_{cond}_s{seed}.json").write_text(json.dumps(payload))


@pytest.fixture
def results(tmp_path):
    for seed, cycles in enumerate([2.0, 1.8, 2.2]):
        _write_metrics(tmp_path, "fixed", "nominal", seed, cycles, 1.5)
    _write_metrics(tmp_path, "diff", "nominal", 0, 2.5, 2.0)
    (tmp_path / "metrics_badname.json").write_text("{}")  # ignored: no cond/seed pattern
    (tmp_path / "table.csv").write_text("")  # non-metrics file ignored
    return tmp_path


def test_flatten_nested():
    assert flatten({"a": {"b": 1}, "c": [1, 2]}) == {"a.b": 1, "c": [1, 2]}


def test_collect_parses_tags(results):
    rows = collect(str(results))
    assert len(rows) == 4
    fixed = [r for r in rows if r["method"] == "fixed"]
    assert {r["seed"] for r in fixed} == {0, 1, 2}
    assert fixed[0]["cond"] == "nominal"
    assert "controller_cost.solve_ms_mean" in fixed[0]


def test_summarize_mean_sd_and_trials(results):
    rows = collect(str(results))
    summary = summarize(rows, ["cycles_mean"])
    fixed = summary[("fixed", "nominal")]
    assert fixed["n_trials"] == 3
    assert fixed["cycles_mean"]["mean"] == pytest.approx(2.0)
    assert fixed["cycles_mean"]["sd"] == pytest.approx(0.2)
    assert sorted(fixed["cycles_mean"]["values"]) == [1.8, 2.0, 2.2]
    assert summary[("diff", "nominal")]["cycles_mean"]["sd"] == 0.0


def test_write_table_csv(results, tmp_path):
    rows = collect(str(results))
    summary = summarize(rows, ["cycles_mean", "controller_cost.solve_ms_mean"])
    out = tmp_path / "out" / "table.csv"
    write_table_csv(summary, ["cycles_mean", "controller_cost.solve_ms_mean"], str(out))
    with open(out) as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == 2 * 2  # 2 groups x 2 metrics
    row = next(r for r in table if r["method"] == "fixed" and r["metric"] == "cycles_mean")
    assert row["values"] == "1.8;2;2.2" or row["values"] == "2;1.8;2.2"  # seed order
    assert row["n_trials"] == "3"


def test_collect_budgets(tmp_path):
    (tmp_path / "bo_nmpc").mkdir()
    (tmp_path / "bo_nmpc" / "budget.json").write_text(json.dumps({"trials": 100, "env_steps": 5}))
    budgets = collect_budgets(str(tmp_path))
    assert budgets["bo_nmpc"]["trials"] == 100
