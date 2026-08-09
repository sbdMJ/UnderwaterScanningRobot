# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""E2 DR-sweep cfg mutation and E3 current-profile driver (pure, duck-typed)."""

from types import SimpleNamespace

import pytest
import torch

from marinelab.experiments.env_variants import CurrentDriver, apply_fluid_dr_scale


def _cfg():
    return SimpleNamespace(randomization=SimpleNamespace(
        added_mass_scale=(0.5, 1.5), linear_damping_scale=(0.5, 1.5),
        quadratic_damping_scale=(0.5, 1.5), volume_scale=(0.85, 1.15),
        roll_range=(-0.785, 0.785)))


def test_fluid_dr_scale_sweeps_only_fluid_coefficients():
    cfg = apply_fluid_dr_scale(_cfg(), 0.25)
    assert cfg.randomization.added_mass_scale == (0.75, 1.25)
    assert cfg.randomization.linear_damping_scale == (0.75, 1.25)
    assert cfg.randomization.quadratic_damping_scale == (0.75, 1.25)
    # non-fluid axes untouched (attitude/volume stress stays at task values)
    assert cfg.randomization.volume_scale == (0.85, 1.15)
    assert cfg.randomization.roll_range == (-0.785, 0.785)
    with pytest.raises(ValueError):
        apply_fluid_dr_scale(_cfg(), 1.5)


class FakeCurrent:
    def __init__(self):
        self.calls = []

    def set(self, env_ids, velocity=None):
        self.calls.append((env_ids.clone(), velocity.clone()))


def _driver(profile, n=2):
    return CurrentDriver(FakeCurrent(), n, "cpu", profile), None


def test_step_profile_reverses_at_t_switch():
    drv, _ = _driver({"type": "step", "speed": 0.2, "heading_deg": 90.0, "t_switch": 60.0})
    before, after = drv.velocity_at(59.9), drv.velocity_at(60.0)
    assert before[1] == pytest.approx(0.2, abs=1e-9) and before[0] == pytest.approx(0.0, abs=1e-9)
    torch.testing.assert_close(after, -before)
    assert before[2:].abs().max() == 0  # linear xy only


def test_step_onset_mode():
    drv, _ = _driver({"type": "step", "speed": 0.1, "t_switch": 30.0, "mode": "onset"})
    assert drv.velocity_at(0.0).abs().max() == 0
    assert drv.velocity_at(30.0)[0] == pytest.approx(0.1)


def test_sine_profile_period_and_amplitude():
    drv, _ = _driver({"type": "sine", "speed": 0.3, "period": 20.0})
    assert drv.velocity_at(5.0)[0] == pytest.approx(0.3)  # quarter period: peak
    assert drv.velocity_at(10.0)[0] == pytest.approx(0.0, abs=1e-9)
    assert drv.velocity_at(15.0)[0] == pytest.approx(-0.3)


def test_apply_writes_shared_component():
    fake = FakeCurrent()
    drv = CurrentDriver(fake, 3, "cpu", {"type": "step", "speed": 0.1, "t_switch": 1.0})
    drv.apply(0.0)
    ids, v = fake.calls[0]
    assert ids.tolist() == [0, 1, 2] and v.shape == (3, 6)
    assert v[2, 0] == pytest.approx(0.1)


def test_from_options_none_and_wired():
    env = SimpleNamespace(_hydro=SimpleNamespace(_current=FakeCurrent()),
                          num_envs=1, device="cpu")
    assert CurrentDriver.from_options({}, env) is None
    drv = CurrentDriver.from_options({"current": {"type": "sine", "speed": 0.1}}, env)
    drv.apply(0.0)
    assert len(env._hydro._current.calls) == 1
    with pytest.raises(ValueError):
        CurrentDriver.from_options({"current": {"type": "gust"}}, env)
