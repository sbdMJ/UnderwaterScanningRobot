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
