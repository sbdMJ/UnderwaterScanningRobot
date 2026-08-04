import math

import pytest
import torch

from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, apply_sensors


def _gt(n=4096):
    return dict(pos=torch.zeros(n,3), quat=torch.tensor([[1.,0,0,0]]).repeat(n,1),
                ang_vel=torch.zeros(n,3), lin_vel=torch.zeros(n,3),
                heading=torch.zeros(n), wall_dist=torch.full((n,), 4.0))


def test_sonar_noise_scale():
    cfg = SensorCfg(sonar_noise=0.05, ukfm_valid_max_depth=8.0)
    out = apply_sensors(_gt(), cfg, torch.Generator().manual_seed(0))
    err = (out["sonar"] - 4.0)
    assert err.abs().mean() < 0.05 and err.std() > 0.01   # noisy but centered


def test_ukfm_dropout_deep():
    gt = _gt(); gt["pos"][:, 2] = -9.0                     # deep -> marker lost
    cfg = SensorCfg(ukfm_valid_max_depth=8.0)
    out = apply_sensors(gt, cfg, torch.Generator().manual_seed(0))
    assert out["ukfm_valid"].max() == 0.0                 # all invalid at depth


def test_per_episode_bias_shifts_mean():
    # A constant per-episode bias shifts the channel mean by exactly the bias (obs-only DR).
    n = 4096
    cfg = SensorCfg(sonar_noise=0.05, dvl_noise=0.02)
    bias = {"sonar": torch.full((n,), 0.3), "lin_vel": torch.full((n, 3), -0.1)}
    out = apply_sensors(_gt(n), cfg, torch.Generator().manual_seed(0), bias=bias)
    assert abs((out["sonar"] - 4.0).mean().item() - 0.3) < 0.01
    assert abs(out["lin_vel"].mean().item() - (-0.1)) < 0.01


def test_no_bias_equals_none():
    # Empty bias dict must match no-bias (regression guard on the b.get(...) defaults).
    n = 256
    cfg = SensorCfg()
    a = apply_sensors(_gt(n), cfg, torch.Generator().manual_seed(1), bias={})
    b = apply_sensors(_gt(n), cfg, torch.Generator().manual_seed(1), bias=None)
    assert torch.allclose(a["sonar"], b["sonar"]) and torch.allclose(a["depth"], b["depth"])


# ---------------------------------------------------------------------------
# UKF-M validity gate: the shipped one is inverted (2026-07-31)
# ---------------------------------------------------------------------------


def test_legacy_ukfm_gate_is_inverted_and_that_is_pinned_deliberately():
    """Pins the KNOWN defect rather than hiding it.

    The marker is at the water surface and the camera looks up, so validity must IMPROVE with
    height. ``legacy_height`` does the opposite. It stays the default because
    ``checkpoints/rb_train_model_7998.pt`` was trained against it; if this test ever starts
    failing, someone changed the default and that checkpoint's observations no longer match.
    """
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, ukfm_in_range

    cfg = SensorCfg()
    assert cfg.ukfm_gate == "legacy_height"
    z = torch.tensor([1.0, 5.0, 7.9, 8.1, 9.5])
    valid = ukfm_in_range(z, cfg)
    assert valid.tolist() == [True, True, True, False, False]
    assert not bool(valid[-1]), "near the surface — where the marker actually IS — reads invalid"


def test_corrected_ukfm_gate_improves_with_height():
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, ukfm_in_range

    cfg = SensorCfg(ukfm_gate="depth_below_surface")
    z = torch.tensor([1.0, 1.9, 2.1, 5.0, 9.5])
    valid = ukfm_in_range(z, cfg)
    assert valid.tolist() == [False, False, True, True, True]
    # monotone in height: the physical statement the user confirmed
    fine = torch.linspace(0.0, 10.0, 101)
    v = ukfm_in_range(fine, cfg).to(torch.int8)
    assert (v.diff() >= 0).all(), "validity must never decrease as the vehicle rises"


def test_corrected_gate_covers_the_scan_band_except_the_bottom_metre():
    """The scan runs z_top 8.5 -> z_bottom 1.0; only below z=2 does the fix drop out."""
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, ukfm_in_range

    cfg = SensorCfg(ukfm_gate="depth_below_surface")
    assert bool(ukfm_in_range(torch.tensor([8.5]), cfg)[0])
    assert not bool(ukfm_in_range(torch.tensor([1.0]), cfg)[0])


def test_near_standoff_limit_rejects_fixes_right_under_the_marker():
    """wallscan_env.py:485 — too close to the marker degrades ukfm/odom on the real robot."""
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, ukfm_in_range

    cfg = SensorCfg(ukfm_gate="depth_below_surface", ukfm_min_standoff=1.0)
    assert not bool(ukfm_in_range(torch.tensor([9.5]), cfg)[0]), "0.5 m under the marker"
    assert bool(ukfm_in_range(torch.tensor([8.5]), cfg)[0]), "1.5 m under it is fine"


def test_unknown_gate_name_is_rejected_rather_than_silently_defaulted():
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, ukfm_in_range

    with pytest.raises(ValueError, match="unknown ukfm_gate"):
        ukfm_in_range(torch.zeros(1), SensorCfg(ukfm_gate="surface"))


def test_apply_sensors_honours_the_gate_choice():
    """End-to-end: the gate must actually reach the ukfm_valid channel the policy observes."""
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, apply_sensors

    n = 4
    gt = dict(pos=torch.tensor([[0.0, 4.5, 9.5]]).repeat(n, 1),
              quat=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
              ang_vel=torch.zeros(n, 3), lin_vel=torch.zeros(n, 3),
              heading=torch.zeros(n), wall_dist=torch.full((n,), 1.5))
    gen = torch.Generator().manual_seed(0)
    legacy = apply_sensors(gt, SensorCfg(), gen)
    fixed = apply_sensors(gt, SensorCfg(ukfm_gate="depth_below_surface"), gen)
    assert float(legacy["ukfm_valid"].max()) == 0.0, "z=9.5 reads INVALID under the legacy gate"
    assert float(fixed["ukfm_valid"].min()) == 1.0, "... and VALID once corrected"


# ---------------------------------------------------------------------------
# Datasheet-backed sensor model (2026-07-31)
# ---------------------------------------------------------------------------

DEG = math.pi / 180.0


def test_channel_splits_fall_back_to_the_legacy_knobs_when_unset():
    """Adding the split fields must not change SensorCfg's behaviour by itself."""
    from marinelab.tasks.pkrc_wallscan.sensors import (
        SensorCfg, att_noise, gyro_noise, heading_bias_dr,
    )

    cfg = SensorCfg(ins_noise=0.01, ins_att_bias_dr=0.04)
    assert att_noise(cfg) == 0.01
    assert gyro_noise(cfg) == 0.01
    assert heading_bias_dr(cfg) == 0.04


def test_channel_splits_override_when_set():
    from marinelab.tasks.pkrc_wallscan.sensors import (
        SensorCfg, att_noise, gyro_noise, heading_bias_dr,
    )

    cfg = SensorCfg(ins_noise=0.01, ins_att_bias_dr=0.04,
                    ins_att_noise=1e-3, ins_gyro_noise=2.9e-4, ins_heading_bias_dr=3.5e-2)
    assert att_noise(cfg) == 1e-3
    assert gyro_noise(cfg) == 2.9e-4
    assert heading_bias_dr(cfg) == 3.5e-2


def test_gyro_bias_matches_the_gv7_turn_on_spec():
    """0.0054 deg/s turn-on bias, which is what a per-episode constant offset models.

    NOT the 1.5 deg/h bias instability: that is the in-run Allan floor, three orders smaller,
    and using it would understate the drift a 39 s unexcited leg actually accumulates.
    """
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfgDatasheet

    assert SensorCfgDatasheet().ins_gyro_bias_dr == pytest.approx(0.0054 * DEG, rel=1e-3)
    assert SensorCfgDatasheet().ins_gyro_bias_dr != pytest.approx(1.5 / 3600 * DEG, rel=0.5)


def test_attitude_biases_match_the_gv7_dynamic_accuracies():
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfgDatasheet, heading_bias_dr

    cfg = SensorCfgDatasheet()
    assert cfg.ins_att_bias_dr == pytest.approx(0.5 * DEG, rel=1e-2), "roll/pitch dynamic 0.5 deg"
    assert heading_bias_dr(cfg) == pytest.approx(2.0 * DEG, rel=1e-2), "heading dynamic 2 deg"
    assert heading_bias_dr(cfg) > 3 * cfg.ins_att_bias_dr, (
        "heading must stay materially worse than roll/pitch: it needs the magnetometer, "
        "roll/pitch are gravity-referenced"
    )


def test_sonar_noise_matches_ping1d_resolution_at_the_operating_standoff():
    """0.5% of range; the mission holds 1.5 m, so 7.5 mm."""
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfgDatasheet

    assert SensorCfgDatasheet().sonar_noise == pytest.approx(0.005 * 1.5, rel=1e-6)


def test_datasheet_values_are_all_smaller_than_the_placeholders():
    """Every guess was pessimistic, which is the point: the sim was harder than reality."""
    from marinelab.tasks.pkrc_wallscan.sensors import (
        SensorCfg, SensorCfgDatasheet, att_noise, gyro_noise,
    )

    train = SensorCfg(sonar_noise=0.05, ins_noise=0.01,
                      sonar_bias_dr=0.10, ins_att_bias_dr=0.04, ins_gyro_bias_dr=0.02)
    ds = SensorCfgDatasheet()
    assert ds.sonar_noise < train.sonar_noise
    assert ds.sonar_bias_dr < train.sonar_bias_dr
    assert att_noise(ds) < att_noise(train)
    assert gyro_noise(ds) < gyro_noise(train)
    assert ds.ins_att_bias_dr < train.ins_att_bias_dr
    assert ds.ins_gyro_bias_dr < train.ins_gyro_bias_dr
    # the biggest single correction, and the one whose cost was measured (crab +0.899 deg)
    assert train.ins_gyro_bias_dr / ds.ins_gyro_bias_dr > 100


def test_gyro_drift_over_an_unexcited_leg_shrinks_from_45_to_below_1_degree():
    """The vertical legs give the sonar no (r, phi) information, so phi rides on the gyro."""
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfgDatasheet
    from marinelab.tasks.pkrc_wallscan.wall_frame_ekf import gyro_drift_over_leg

    placeholder = math.degrees(gyro_drift_over_leg(0.02, 39.0))
    real = math.degrees(gyro_drift_over_leg(SensorCfgDatasheet().ins_gyro_bias_dr, 39.0))
    assert placeholder == pytest.approx(44.7, abs=0.5)
    assert real < 1.0


def test_datasheet_cfg_leaves_the_unverified_channels_alone():
    """DVL / depth / UKF-M were not looked up, so they must still read as placeholders."""
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, SensorCfgDatasheet

    base, ds = SensorCfg(), SensorCfgDatasheet()
    for field in ("dvl_noise", "depth_noise", "ukfm_noise", "ukfm_valid_max_depth"):
        assert getattr(ds, field) == getattr(base, field), field


def test_apply_sensors_uses_the_split_noise_channels():
    """A tiny attitude noise with a large rate noise must show up on the right channels."""
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, apply_sensors

    n = 4096
    gt = dict(pos=torch.zeros(n, 3), quat=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
              ang_vel=torch.zeros(n, 3), lin_vel=torch.zeros(n, 3),
              heading=torch.zeros(n), wall_dist=torch.full((n,), 1.5))
    cfg = SensorCfg(ins_noise=0.5, ins_att_noise=1e-4, ins_gyro_noise=0.2)
    out = apply_sensors(gt, cfg, torch.Generator().manual_seed(0))
    assert float(out["up_vec"][:, 0].std()) < 1e-3, "attitude must use ins_att_noise"
    assert 0.15 < float(out["ang_vel"][:, 0].std()) < 0.25, "rate must use ins_gyro_noise"


# ---------------------------------------------------------------------------
# DVL scale-factor error (2026-08-03)
# ---------------------------------------------------------------------------


def test_dvl_scale_is_off_by_default():
    """Adding a new error FORM must not change SensorCfg's behaviour."""
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg

    assert SensorCfg().dvl_scale_dr == 0.0


def test_dvl_scale_matches_the_a50_standard_accuracy():
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfgDatasheet

    cfg = SensorCfgDatasheet()
    assert cfg.dvl_scale_dr == pytest.approx(0.0101), "+-1.01% long term accuracy, Standard"
    assert cfg.dvl_bias_dr == 0.0, (
        "the datasheet expresses DVL error as a percentage, so keeping an additive offset "
        "too would double-count and reinstate the drift the scale form removes"
    )


def test_scale_error_is_proportional_to_speed_and_absent_at_rest():
    """The whole point: a percentage error cannot accrue while the vehicle is parked."""
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, apply_sensors

    n = 2048
    cfg = SensorCfg(dvl_noise=0.0)  # isolate the scale term
    scale = torch.full((n, 3), 0.01)

    def measure(v_y):
        gt = dict(pos=torch.zeros(n, 3), quat=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
                  ang_vel=torch.zeros(n, 3),
                  lin_vel=torch.stack([torch.zeros(n), torch.full((n,), v_y), torch.zeros(n)], -1),
                  heading=torch.zeros(n), wall_dist=torch.full((n,), 1.5))
        return apply_sensors(gt, cfg, None, bias={"lin_vel_scale": scale})["lin_vel"][:, 1]

    assert float(measure(0.0).abs().max()) == pytest.approx(0.0, abs=1e-9), "no error at rest"
    assert float(measure(0.1).mean()) == pytest.approx(0.1 * 1.01, rel=1e-6)
    assert float(measure(0.2).mean()) == pytest.approx(0.2 * 1.01, rel=1e-6)


def test_scale_and_additive_forms_differ_by_45x_over_one_episode():
    """Quantifies why the form matters, not just the magnitude.

    ``s`` is the body-y integral. The sway legs cover ~4 m per 180 s episode and v_y is ~0
    through the vertical legs, so an additive offset accrues with TIME while a scale error
    accrues with DISTANCE.
    """
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfgDatasheet

    additive_drift = 0.01 * 180.0                                  # placeholder, m
    scale_drift = SensorCfgDatasheet().dvl_scale_dr * 4.0           # datasheet, m
    assert additive_drift == pytest.approx(1.8)
    assert scale_drift == pytest.approx(0.0404)
    assert additive_drift / scale_drift == pytest.approx(45, abs=2)


def test_apply_sensors_applies_scale_before_the_additive_offset():
    """Order matters for a chain of v*(1+k) + noise + b; pin it."""
    from marinelab.tasks.pkrc_wallscan.sensors import SensorCfg, apply_sensors

    n = 1
    gt = dict(pos=torch.zeros(n, 3), quat=torch.tensor([[1.0, 0, 0, 0]]),
              ang_vel=torch.zeros(n, 3), lin_vel=torch.tensor([[0.0, 1.0, 0.0]]),
              heading=torch.zeros(n), wall_dist=torch.full((n,), 1.5))
    out = apply_sensors(gt, SensorCfg(dvl_noise=0.0), None,
                        bias={"lin_vel_scale": torch.full((n, 3), 0.1),
                              "lin_vel": torch.full((n, 3), 0.05)})
    assert float(out["lin_vel"][0, 1]) == pytest.approx(1.0 * 1.1 + 0.05, rel=1e-6)
