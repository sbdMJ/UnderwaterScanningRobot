# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Experiment-matrix expansion and naming convention."""

import pytest

from marinelab.experiments.protocol import ExperimentCell, load_cells

YAML = """
exp: e_test
defaults:
  task: Isaac-PKRC-WallScan-Stage3-Direct-v0
  steps: 9000
  num_envs: 1
conditions:
  nominal: {}
  dr50:
    task: Isaac-PKRC-WallScan-Eval-Direct-v0
    steps: 4500
methods:
  fixed:
    tam: fixed
  ppo:
    num_envs: 8
    policy: checkpoints/exported/policy.pt
seeds: [0, 1]
"""


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(YAML)
    return str(path)


def test_full_expansion(config):
    cells = load_cells(config)
    assert len(cells) == 2 * 2 * 2  # methods x conditions x seeds
    tags = {c.tag for c in cells}
    assert "fixed_nominal_s0" in tags and "ppo_dr50_s1" in tags


def test_merge_precedence(config):
    cells = {c.tag: c for c in load_cells(config)}
    # condition overrides defaults
    assert cells["fixed_dr50_s0"].options["steps"] == 4500
    assert cells["fixed_dr50_s0"].options["task"].endswith("Eval-Direct-v0")
    assert cells["fixed_nominal_s0"].options["steps"] == 9000
    # method overrides both
    assert cells["ppo_nominal_s0"].options["num_envs"] == 8
    assert cells["fixed_nominal_s0"].options["num_envs"] == 1


def test_filters(config):
    assert len(load_cells(config, only_method="ppo")) == 4
    assert len(load_cells(config, only_cond="nominal", only_seed=1)) == 2


def test_paths():
    cell = ExperimentCell(exp="e1", method="diff", cond="nominal", seed=3)
    assert cell.trajectory_path("results").as_posix() == "results/e1/raw/trajectory_diff_nominal_s3.npz"
    assert cell.metrics_path("results").as_posix() == "results/e1/metrics/metrics_diff_nominal_s3.json"
    assert cell.plot_path("results").as_posix() == "results/e1/plots/trajectory_diff_nominal_s3.png"
    assert cell.out_dir("results", "tables").as_posix() == "results/e1/tables"


def test_missing_key_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("exp: x\nmethods: {}\nconditions: {}\n")  # no seeds
    with pytest.raises(KeyError):
        load_cells(str(path))
