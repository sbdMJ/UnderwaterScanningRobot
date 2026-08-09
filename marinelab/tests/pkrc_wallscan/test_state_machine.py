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


# ---------------------------------------------------------------------------
# snap_ramp_on_vertical (2026-08-08)
#
# A sway leg can clear up to reach_eps short of its target; without the snap the s ramp keeps
# slewing toward that target through the vertical leg, so the reference commands 55 cm of
# sideways motion during a leg that should be vertical. Off by default -- every published
# number ran with the bleed present.
# ---------------------------------------------------------------------------
def _cfg(**over):
    base = dict(z_top=8.5, z_bottom=1.0, sway_step=1.0, reach_eps=0.6, reach_hold=1,
                ref_step=0.004, ref_step_s=0.002)
    base.update(over)
    return ScanCfg(**base)


def test_snap_off_by_default_so_published_behaviour_is_unchanged():
    assert ScanCfg(z_top=8.5, z_bottom=1.0, reach_eps=0.6,
                       reach_hold=1).snap_ramp_on_vertical is False


def test_snap_drops_the_arc_reference_onto_the_vehicle_when_a_vertical_leg_begins():
    n = 1
    for snap in (False, True):
        cfg = _cfg(snap_ramp_on_vertical=snap)
        st = ScanState(n, device="cpu")
        st.phase[:] = 1
        st.s_ref[:] = 1.0              # sway target
        st.s_ramp[:] = 0.7
        z = torch.full((n,), 8.5)      # z_hold satisfied, so the sway gate is on |s - s_ref|
        s = torch.full((n,), 0.55)     # 0.45 m short -- inside reach_eps 0.6, so it clears
        step(st, z, s, cfg, z_latch=z)
        assert (st.phase == 2).all(), "sway must clear at the band edge"
        if snap:
            assert torch.allclose(st.s_ref, s) and torch.allclose(st.s_ramp, s), \
                "the leftover must be abandoned, not carried into the vertical leg"
        else:
            assert torch.allclose(st.s_ref, torch.full((n,), 1.0)), "legacy keeps chasing it"


def test_snap_leaves_the_sway_bump_itself_alone():
    """Only VERTICAL entries snap; entering a sway leg must still command a full sway_step."""
    n = 1
    cfg = _cfg(snap_ramp_on_vertical=True)
    st = ScanState(n, device="cpu")
    st.phase[:] = 0
    z = torch.full((n,), 1.0)          # at z_bottom -> descend clears
    st.z_ramp[:] = 1.0                 # ...but only if the RAMP has arrived too (07-26 gate)
    s = torch.full((n,), 2.0)
    step(st, z, s, cfg, z_latch=z)
    assert (st.phase == 1).all()
    assert torch.allclose(st.s_ref, s + cfg.sway_step), "sway target must still be a real step"
