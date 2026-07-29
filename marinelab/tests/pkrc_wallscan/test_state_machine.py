import math

import torch

from marinelab.tasks.pkrc_wallscan.scan_state_machine import ScanCfg, ScanState, search_step, step


def test_descend_to_sway_advances():
    st = ScanState(n=1); cfg = ScanCfg(z_top=9.0, z_bottom=1.0, reach_eps=0.1, reach_hold=1)
    # at bottom -> should advance out of DESCEND and bump s_ref by sway_step
    z = torch.tensor([1.05]); s = torch.tensor([0.0])
    z_ref, s_ref, phase_sc, adv = step(st, z, s, cfg)
    assert adv.item() is True or adv.item() == True
    assert st.phase.item() == 1 and torch.isclose(st.s_ref, torch.tensor([1.0]), atol=1e-4).all()


def test_sway_holds_entry_depth():
    st = ScanState(n=1); cfg = ScanCfg(z_top=9.0, z_bottom=1.0, reach_eps=0.1, reach_hold=1)
    # enter SWAY at z=1.05 -> z_ref latches to 1.05
    z_ref, s_ref, _, adv = step(st, torch.tensor([1.05]), torch.tensor([0.0]), cfg)
    assert adv.item() and st.phase.item() == 1
    assert torch.isclose(z_ref, torch.tensor([1.05]), atol=1e-4).all()
    # z drifts while still swaying (s not yet reached) -> z_ref stays latched, not live z
    z_ref2, _, _, adv2 = step(st, torch.tensor([5.3]), torch.tensor([0.0]), cfg)
    assert not adv2.item() and st.phase.item() == 1
    assert torch.isclose(z_ref2, torch.tensor([1.05]), atol=1e-4).all()


def test_search_step_tracks_min_bearing_and_terminates():
    # Sweep a full turn at omega; the recorded bearing must be the heading where the sonar
    # reading was smallest, and `active` must go False once swept >= 2*pi.
    omega, dt = 0.314, 0.02
    n_steps = int(2.0 * math.pi / (omega * dt)) + 1
    swept = torch.zeros(1)
    best_dist = torch.full((1,), float("inf"))
    best_yaw = torch.zeros(1)
    target_heading = 2.0  # rad: where the wall is nearest
    for i in range(n_steps):
        heading = torch.tensor([omega * dt * i])
        # sonar minimum (2.0 m) exactly at target_heading, else 5.0 m
        sonar = torch.where((heading - target_heading).abs() < 0.05, torch.tensor([2.0]), torch.tensor([5.0]))
        swept, best_dist, best_yaw, active = search_step(swept, best_dist, best_yaw, sonar, heading, omega, dt)
    assert not active.item(), "search must terminate after a full sweep"
    assert abs(best_yaw.item() - target_heading) < 0.1, "locked bearing must be the sonar-min heading"
    assert torch.isclose(best_dist, torch.tensor([2.0]))


def test_search_step_active_until_full_turn():
    swept = torch.tensor([2.0 * math.pi - 0.01])
    _, _, _, active = search_step(swept, torch.tensor([5.0]), torch.zeros(1),
                                  torch.tensor([5.0]), torch.zeros(1), omega=0.314, dt=0.001)
    assert active.item()  # not yet a full turn
    _, _, _, active2 = search_step(swept, torch.tensor([5.0]), torch.zeros(1),
                                   torch.tensor([5.0]), torch.zeros(1), omega=0.314, dt=1.0)
    assert not active2.item()  # crossed 2*pi


def test_zhold_latches_ground_truth_not_sensor():
    # Regression (DR bias leak): z_hold must latch z_latch (GT depth), not z (biased sensor).
    # Sensor depth reads +0.05 biased; entering SWAY must latch the true 1.00, so the GT-compared
    # depth reward sees no constant offset.
    st = ScanState(n=1); cfg = ScanCfg(z_top=9.0, z_bottom=1.0, reach_eps=0.1, reach_hold=1)
    z_sensor = torch.tensor([1.05])   # biased: true depth + 0.05
    z_gt = torch.tensor([1.00])       # ground truth
    z_ref, _, _, adv = step(st, z_sensor, torch.tensor([0.0]), cfg, z_latch=z_gt)
    assert adv.item() and st.phase.item() == 1
    assert torch.isclose(z_ref, z_gt, atol=1e-4).all(), "z_ref must latch GT depth, not biased sensor"
