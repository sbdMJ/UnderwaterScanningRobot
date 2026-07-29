# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for marinelab.core.thruster."""

import torch

from marinelab.core.thruster import ThrusterModel


class _Cfg:
    num_thrusters = 6
    max_thrust = 50.0
    thrust_coefficient = 40.0
    time_constant_up = 0.1
    time_constant_down = 0.05
    allocation_matrix = (
        (0.707, 0.707, -0.707, -0.707, 0.0, 0.0),
        (-0.707, 0.707, 0.707, -0.707, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 0.0, 0.1, -0.1),
        (0.0, 0.0, 0.0, 0.0, 0.12, 0.12),
        (0.19, -0.19, 0.19, -0.19, 0.0, 0.0),
    )


def test_first_order_step_up():
    m = ThrusterModel(_Cfg(), num_envs=1, device="cpu")
    m.apply_dynamics(torch.ones(1, 6), dt=0.1)
    # alpha = dt/tau_up = 0.1/0.1 = 1.0 -> state snaps to target
    assert torch.allclose(m.state, torch.ones(1, 6), atol=1e-5)


def test_first_order_partial():
    m = ThrusterModel(_Cfg(), num_envs=1, device="cpu")
    m.apply_dynamics(torch.ones(1, 6), dt=0.005)
    # alpha = 0.005/0.1 = 0.05
    assert torch.allclose(m.state, torch.full((1, 6), 0.05), atol=1e-5)


def test_compute_wrench_clamps_thrust():
    m = ThrusterModel(_Cfg(), num_envs=1, device="cpu")
    m._state = torch.ones(1, 6) * 2.0  # 2*40=80 N -> clamps to 50
    f, t = m.compute_wrench()
    # heave row: VL+VR = 50+50 = 100 N
    assert abs(t.shape[-1] - 3) == 0
    assert abs(f[0, 2].item() - 100.0) < 1e-3


def test_reset_zeros_state():
    m = ThrusterModel(_Cfg(), num_envs=4, device="cpu")
    m.apply_dynamics(torch.ones(4, 6), dt=0.1)
    m.reset(torch.tensor([0, 2]))
    assert torch.all(m.state[0] == 0.0)
    assert torch.all(m.state[1] != 0.0)


# ============================================================================
# Thruster fault injection (per-env per-thruster health, FTC infrastructure)
# ============================================================================
# A fault is a multiplicative health in [0, 1] applied to each thruster's force
# magnitude (0 = dead, 1 = nominal, 0.5 = half degradation). It is distinct from
# DR (thrust_coefficient scaling): DR varies a physical parameter across envs,
# while a fault models an actuator FAILING mid-mission. The toggle-off contract
# (enable_fault=False -> byte-identical) is enforced structurally: with the buffer
# None, compute_wrench skips the health multiply entirely (no float drift, no 1.0
# multiply), exactly like the state_dependent_std toggle-off branch.


def test_fault_disabled_buffer_is_none():
    """enable_fault=False -> no health buffer (structural toggle-off)."""
    m = ThrusterModel(_Cfg(), num_envs=4, device="cpu", enable_fault=False)
    assert m._thruster_health is None


def test_fault_disabled_wrench_byte_identical():
    """With fault off, compute_wrench is byte-identical to the fault-free model.

    Both a fault-disabled model and a model built without the fault arg must
    produce the SAME wrench for the same state -- the health code path must not
    perturb the result by even one ULP.
    """
    state = torch.ones(2, 6) * 0.5
    m_off = ThrusterModel(_Cfg(), num_envs=2, device="cpu", enable_fault=False)
    m_base = ThrusterModel(_Cfg(), num_envs=2, device="cpu")
    m_off._state = state.clone()
    m_base._state = state.clone()
    f_off, t_off = m_off.compute_wrench()
    f_base, t_base = m_base.compute_wrench()
    assert torch.equal(f_off, f_base)
    assert torch.equal(t_off, t_base)


def test_fault_enabled_default_health_is_nominal():
    """enable_fault=True with no injection -> health all ones -> nominal wrench.

    Turning the toggle ON but injecting nothing must still equal the baseline:
    default health is 1.0 everywhere.
    """
    state = torch.ones(3, 6) * 0.7
    m_on = ThrusterModel(_Cfg(), num_envs=3, device="cpu", enable_fault=True)
    m_base = ThrusterModel(_Cfg(), num_envs=3, device="cpu")
    m_on._state = state.clone()
    m_base._state = state.clone()
    assert m_on._thruster_health.shape == (3, 6)
    assert torch.all(m_on._thruster_health == 1.0)
    f_on, _ = m_on.compute_wrench()
    f_base, _ = m_base.compute_wrench()
    assert torch.allclose(f_on, f_base, atol=1e-6)


def test_fault_dead_thruster_contributes_zero():
    """health=0 on a thruster removes its contribution to the wrench.

    Kill thrusters 0 and 1 (the heave row uses thrusters 4,5, so heave is
    unaffected; surge/sway/yaw lose t0,t1). Compare against a model where
    t0,t1 states are manually zeroed -- equivalent physics.
    """
    m = ThrusterModel(_Cfg(), num_envs=1, device="cpu", enable_fault=True)
    m.set_thruster_health(torch.tensor([0]), torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]))
    m._state = torch.ones(1, 6) * 0.5
    f_fault, t_fault = m.compute_wrench()

    # Reference: same model, no fault, but t0,t1 commanded to 0.
    m_ref = ThrusterModel(_Cfg(), num_envs=1, device="cpu")
    m_ref._state = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.5, 0.5]])
    f_ref, t_ref = m_ref.compute_wrench()

    assert torch.allclose(f_fault, f_ref, atol=1e-5)
    assert torch.allclose(t_fault, t_ref, atol=1e-5)


def test_fault_partial_degradation_scales_force():
    """health=0.5 halves that thruster's force contribution (linear)."""
    m = ThrusterModel(_Cfg(), num_envs=1, device="cpu", enable_fault=True)
    m.set_thruster_health(torch.tensor([0]), torch.full((1, 6), 0.5))
    m._state = torch.ones(1, 6) * 0.4
    f_half, _ = m.compute_wrench()

    m_full = ThrusterModel(_Cfg(), num_envs=1, device="cpu")
    m_full._state = torch.ones(1, 6) * 0.4
    f_full, _ = m_full.compute_wrench()

    # All thrusters at 0.5 health -> every force component halved.
    assert torch.allclose(f_half, 0.5 * f_full, atol=1e-5)


def test_fault_health_is_per_env():
    """Different envs can carry different health (per-env failure pattern)."""
    m = ThrusterModel(_Cfg(), num_envs=2, device="cpu", enable_fault=True)
    # env 0 healthy, env 1 has thruster 0 dead.
    m.set_thruster_health(torch.tensor([1]), torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]))
    assert torch.all(m._thruster_health[0] == 1.0)
    assert m._thruster_health[1, 0] == 0.0
    assert torch.all(m._thruster_health[1, 1:] == 1.0)


def test_set_health_noop_when_disabled():
    """set_thruster_health is a safe no-op when fault is disabled."""
    m = ThrusterModel(_Cfg(), num_envs=2, device="cpu", enable_fault=False)
    m.set_thruster_health(torch.tensor([0]), torch.zeros(1, 6))  # must not raise
    assert m._thruster_health is None


class _CurveCfg(_Cfg):
    """T200 nonlinear thrust curve enabled."""

    enable_thrust_curve = True
    thrust_deadband = 0.075


def test_thrust_curve_off_is_linear():
    # Default cfg has no curve flag -> byte-identical to the linear model.
    m = ThrusterModel(_Cfg(), num_envs=1, device="cpu")
    m._state = torch.full((1, 6), 0.5)
    f, _ = m.compute_wrench()
    # heave row VL+VR = 0.5*40 + 0.5*40 = 40 N (linear map, no curve)
    assert abs(f[0, 2].item() - 40.0) < 1e-3


def test_thrust_curve_signed_square():
    m = ThrusterModel(_CurveCfg(), num_envs=1, device="cpu")
    m._state = torch.full((1, 6), 0.5)
    f, _ = m.compute_wrench()
    # signed-square: 0.5**2 * 40 = 10 N per thruster; heave VL+VR = 20 N
    assert abs(f[0, 2].item() - 20.0) < 1e-3


def test_thrust_curve_deadband_zeroes_small_command():
    m = ThrusterModel(_CurveCfg(), num_envs=1, device="cpu")
    m._state = torch.full((1, 6), 0.05)  # below 0.075 deadband -> zero thrust
    f, t = m.compute_wrench()
    assert torch.allclose(f, torch.zeros_like(f), atol=1e-6)
    assert torch.allclose(t, torch.zeros_like(t), atol=1e-6)


def test_thrust_curve_sign_and_bounds_preserved():
    m = ThrusterModel(_CurveCfg(), num_envs=1, device="cpu")
    # Full command keeps magnitude 1 (1**2=1) and sign; heave VL+VR = -80 N.
    m._state = torch.full((1, 6), -1.0)
    f, _ = m.compute_wrench()
    assert abs(f[0, 2].item() - (-80.0)) < 1e-3


# ============================================================================
# Thruster fault injection (per-env per-thruster health, FTC infrastructure)
# ============================================================================
# A fault is a multiplicative health in [0, 1] applied to each thruster's force
# magnitude (0 = dead, 1 = nominal, 0.5 = half degradation). It is distinct from
# DR (thrust_coefficient scaling): DR varies a physical parameter across envs,
# while a fault models an actuator FAILING mid-mission. The toggle-off contract
# (enable_fault=False -> byte-identical) is enforced structurally: with the buffer
# None, compute_wrench skips the health multiply entirely (no float drift, no 1.0
# multiply), exactly like the state_dependent_std toggle-off branch.


def test_fault_disabled_buffer_is_none():
    """enable_fault=False -> no health buffer (structural toggle-off)."""
    m = ThrusterModel(_Cfg(), num_envs=4, device="cpu", enable_fault=False)
    assert m._thruster_health is None


def test_fault_disabled_wrench_byte_identical():
    """With fault off, compute_wrench is byte-identical to the fault-free model.

    Both a fault-disabled model and a model built without the fault arg must
    produce the SAME wrench for the same state -- the health code path must not
    perturb the result by even one ULP.
    """
    state = torch.ones(2, 6) * 0.5
    m_off = ThrusterModel(_Cfg(), num_envs=2, device="cpu", enable_fault=False)
    m_base = ThrusterModel(_Cfg(), num_envs=2, device="cpu")
    m_off._state = state.clone()
    m_base._state = state.clone()
    f_off, t_off = m_off.compute_wrench()
    f_base, t_base = m_base.compute_wrench()
    assert torch.equal(f_off, f_base)
    assert torch.equal(t_off, t_base)


def test_fault_enabled_default_health_is_nominal():
    """enable_fault=True with no injection -> health all ones -> nominal wrench.

    Turning the toggle ON but injecting nothing must still equal the baseline:
    default health is 1.0 everywhere.
    """
    state = torch.ones(3, 6) * 0.7
    m_on = ThrusterModel(_Cfg(), num_envs=3, device="cpu", enable_fault=True)
    m_base = ThrusterModel(_Cfg(), num_envs=3, device="cpu")
    m_on._state = state.clone()
    m_base._state = state.clone()
    assert m_on._thruster_health.shape == (3, 6)
    assert torch.all(m_on._thruster_health == 1.0)
    f_on, _ = m_on.compute_wrench()
    f_base, _ = m_base.compute_wrench()
    assert torch.allclose(f_on, f_base, atol=1e-6)


def test_fault_dead_thruster_contributes_zero():
    """health=0 on a thruster removes its contribution to the wrench.

    Kill thrusters 0 and 1 (the heave row uses thrusters 4,5, so heave is
    unaffected; surge/sway/yaw lose t0,t1). Compare against a model where
    t0,t1 states are manually zeroed -- equivalent physics.
    """
    m = ThrusterModel(_Cfg(), num_envs=1, device="cpu", enable_fault=True)
    m.set_thruster_health(torch.tensor([0]), torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]))
    m._state = torch.ones(1, 6) * 0.5
    f_fault, t_fault = m.compute_wrench()

    # Reference: same model, no fault, but t0,t1 commanded to 0.
    m_ref = ThrusterModel(_Cfg(), num_envs=1, device="cpu")
    m_ref._state = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.5, 0.5]])
    f_ref, t_ref = m_ref.compute_wrench()

    assert torch.allclose(f_fault, f_ref, atol=1e-5)
    assert torch.allclose(t_fault, t_ref, atol=1e-5)


def test_fault_partial_degradation_scales_force():
    """health=0.5 halves that thruster's force contribution (linear)."""
    m = ThrusterModel(_Cfg(), num_envs=1, device="cpu", enable_fault=True)
    m.set_thruster_health(torch.tensor([0]), torch.full((1, 6), 0.5))
    m._state = torch.ones(1, 6) * 0.4
    f_half, _ = m.compute_wrench()

    m_full = ThrusterModel(_Cfg(), num_envs=1, device="cpu")
    m_full._state = torch.ones(1, 6) * 0.4
    f_full, _ = m_full.compute_wrench()

    # All thrusters at 0.5 health -> every force component halved.
    assert torch.allclose(f_half, 0.5 * f_full, atol=1e-5)


def test_fault_health_is_per_env():
    """Different envs can carry different health (per-env failure pattern)."""
    m = ThrusterModel(_Cfg(), num_envs=2, device="cpu", enable_fault=True)
    # env 0 healthy, env 1 has thruster 0 dead.
    m.set_thruster_health(torch.tensor([1]), torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0]]))
    assert torch.all(m._thruster_health[0] == 1.0)
    assert m._thruster_health[1, 0] == 0.0
    assert torch.all(m._thruster_health[1, 1:] == 1.0)


def test_set_health_noop_when_disabled():
    """set_thruster_health is a safe no-op when fault is disabled."""
    m = ThrusterModel(_Cfg(), num_envs=2, device="cpu", enable_fault=False)
    m.set_thruster_health(torch.tensor([0]), torch.zeros(1, 6))  # must not raise
    assert m._thruster_health is None


# ============================================================================
# Combined path: nonlinear thrust curve ON + fault ON (order lock)
# ============================================================================
# When both features are active the order MUST be: curve (signed-square) ->
# * coeff -> * health -> clamp. This test pins that order: the curve shapes the
# command, then coeff scales it, then health degrades it, all before saturation.


class _CurveFaultCfg(_CurveCfg):
    """Both the T200 nonlinear curve and fault injection exercised together."""


def test_curve_and_fault_compose_in_order():
    """curve -> coeff -> health -> clamp: signed-square then coeff then health.

    state=0.5, deadband=0.075 (active) -> signed-square(0.5)=0.25;
    * coeff(40) = 10 N; * health(0.5) = 5 N per thruster (below the 50 N clamp).
    Heave row (VL+VR) => 2 * 5 = 10 N.
    """
    m = ThrusterModel(_CurveFaultCfg(), num_envs=1, device="cpu", enable_fault=True)
    m.set_thruster_health(torch.tensor([0]), torch.full((1, 6), 0.5))
    m._state = torch.full((1, 6), 0.5)
    f, _ = m.compute_wrench()
    assert abs(f[0, 2].item() - 10.0) < 1e-3


# ============================================================================
# Per-step actuation noise (white, multiplicative command noise)
# ============================================================================
# The THIRD actuation channel, independent of DR (thrust_coefficient spread,
# reset-time) and fault (thruster health, reset-time). It perturbs the COMMAND
# every step: applied = commanded * (1 + eps), eps ~ N(0, std), drawn per-env
# per-thruster. Armed only by enable_actuation_noise=True; None guard when off
# means byte-identical (no RNG consumed).


class _NoiseCfg(_Cfg):
    """Actuation-noise cfg: nonzero std so the channel is active when armed."""

    actuation_noise_std = 0.1


def test_actuation_noise_off_is_byte_identical():
    # Default cfg + flag False -> apply_dynamics byte-identical to the no-noise
    # model, and no RNG consumed (re-run from a fresh model gives the same state).
    torch.manual_seed(0)
    m = ThrusterModel(_Cfg(), num_envs=4, device="cpu")
    m.apply_dynamics(torch.full((4, 6), 0.3), dt=0.02)
    ref = m.state.clone()
    m2 = ThrusterModel(_Cfg(), num_envs=4, device="cpu")
    m2.apply_dynamics(torch.full((4, 6), 0.3), dt=0.02)
    assert torch.equal(ref, m2.state)


def test_actuation_noise_disabled_when_flag_false_even_if_std_set():
    # std nonzero in cfg but enable_actuation_noise=False -> still off.
    m = ThrusterModel(_NoiseCfg(), num_envs=4, device="cpu", enable_actuation_noise=False)
    assert m._act_noise_std is None
    m_ref = ThrusterModel(_Cfg(), num_envs=4, device="cpu")
    m.apply_dynamics(torch.full((4, 6), 0.3), dt=0.02)
    m_ref.apply_dynamics(torch.full((4, 6), 0.3), dt=0.02)
    assert torch.equal(m.state, m_ref.state)


def test_actuation_noise_on_perturbs_and_resamples():
    m = ThrusterModel(_NoiseCfg(), num_envs=4, device="cpu", enable_actuation_noise=True)
    assert m._act_noise_std == 0.1
    torch.manual_seed(0)
    m.apply_dynamics(torch.full((4, 6), 0.3), dt=0.02)
    s1 = m.state.clone()
    m.reset(torch.arange(4))
    torch.manual_seed(1)  # different draw
    m.apply_dynamics(torch.full((4, 6), 0.3), dt=0.02)
    s2 = m.state.clone()
    # Two independent noise draws -> different resulting states (per-step resample).
    assert not torch.allclose(s1, s2)


def test_actuation_noise_independent_of_dr_and_fault():
    # Noise on, DR off, fault off -> DR/fault buffers stay None.
    m = ThrusterModel(_NoiseCfg(), num_envs=4, device="cpu", enable_actuation_noise=True)
    assert m._thrust_coeff is None        # DR buffer not allocated
    assert m._thruster_health is None     # fault buffer not allocated
    assert m._act_noise_std == 0.1


def test_randomize_parameters_accepts_per_env_tensor():
    m = ThrusterModel(_Cfg(), num_envs=4, device="cpu", enable_randomization=True)
    base_tc = m._thrust_coeff.clone()
    base_up = m._time_constant_up.clone()
    base_down = m._time_constant_down.clone()
    ids = torch.tensor([1, 3])
    s = torch.tensor([2.0, 0.5])
    m.randomize_parameters(ids, thrust_coeff_scale=s, time_constant_scale=s)
    assert torch.allclose(m._thrust_coeff[1], torch.as_tensor(m.cfg.thrust_coefficient * 2.0).float())
    assert torch.allclose(m._thrust_coeff[3], torch.as_tensor(m.cfg.thrust_coefficient * 0.5).float())
    assert torch.allclose(m._thrust_coeff[0], base_tc[0])  # untouched env
    assert torch.allclose(m._time_constant_up[1], base_up[1] * 0.0 + m.cfg.time_constant_up * 2.0)
    assert torch.allclose(m._time_constant_down[1], torch.as_tensor(m.cfg.time_constant_down * 2.0).float())
    # Regression: float64 tensor input should not raise, should cast correctly
    s64 = s.double()
    m.randomize_parameters(ids, thrust_coeff_scale=s64, time_constant_scale=s64)
    assert torch.allclose(m._thrust_coeff[1], torch.as_tensor(m.cfg.thrust_coefficient * 2.0).float())
