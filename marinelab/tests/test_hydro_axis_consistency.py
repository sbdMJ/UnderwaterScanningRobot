# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Guard: a vehicle's parameter groups must agree on which axis its hull is long along.

Added mass, translational drag and rigid-body inertia each independently identify the long
axis; if they disagree, at least one group is misassigned, and that is detectable with no
external data at all. This caught PKRC on 2026-07-31: the coefficients said x, the USD-derived
PhysX inertia said z, and the user confirmed the hull is z-slender — so the mesh was right and
heave's damping had been copied from sway using correct reasoning applied to the wrong axis.

Like ``test_thruster_allocation.py``, this pins BOTH the defect (in the shipped config, kept
for checkpoint reproducibility) and its fix.
"""

import pytest
import torch

from marinelab.core.parameters import slender_axis, slender_axis_is_consistent
# tests/ is a package (it has __init__.py), so pytest imports these as tests.*.
# Reusing its stub installer keeps ONE copy of the isaaclab shims the asset cfgs need.
from tests.test_thruster_allocation import _install_asset_cfg_stubs  # noqa: E402

_install_asset_cfg_stubs()

from marinelab.assets.pkrc.pkrc import (  # noqa: E402
    PKRCHydrodynamicsCfg,
    PKRCHydrodynamicsCfgZSlender,
)

# PhysX tensor derived from PKRC.usd, measured by isaaclab/logs/_probe_plant.py. Not in any
# config file, which is exactly why the contradiction went unnoticed.
PHYSX_INERTIA = (1.412, 1.406, 0.393)
X, Y, Z = 0, 1, 2


def _verdicts(cfg):
    return slender_axis(cfg.added_mass, cfg.linear_damping, cfg.quadratic_damping, PHYSX_INERTIA)


# --- the invariant itself ---------------------------------------------------


def test_a_hand_built_consistent_vehicle_passes():
    """z-slender: axial (z) minimizes added mass, drag and inertia together."""
    assert slender_axis_is_consistent(
        (60.0, 60.0, 20.0, 0.5, 0.5, 0.5),
        (120.0, 120.0, 50.0, 15.0, 15.0, 4.0),
        (40.0, 40.0, 16.0, 30.0, 30.0, 8.0),
        (1.4, 1.4, 0.4),
    )


def test_drag_verdict_uses_the_total_not_the_linear_coefficient():
    """L-vs-Q fit ambiguity: a big quadratic spread can hide equal physical drag.

    PKRC's surge/sway differ 4.7x in the quadratic term yet only 5% in total drag at 0.2 m/s,
    so judging on the linear coefficient alone would name the wrong axis.
    """
    lin = (97.79, 119.44, 49.77, 0, 0, 0)
    quad = (180.85, 38.51, 16.05, 0, 0, 0)
    v = 0.2
    totals = [lin[i] * v + quad[i] * v * v for i in range(3)]
    assert totals[2] < totals[1] < totals[0], "heave must be the lightest at operating speed"
    assert abs(totals[0] - totals[1]) / totals[1] < 0.06, "surge/sway agree within 6%"
    # the linear coefficient alone would still pick heave here, but not the quadratic alone
    assert min(range(3), key=lambda i: quad[i]) == Z
    assert min(range(3), key=lambda i: lin[i]) == Z


def test_inconsistency_is_reported_per_group():
    v = slender_axis((19.4, 64.65, 64.65, 0.5, 0.5, 0.5),
                     (97.79, 119.44, 119.44, 15.0, 15.0, 4.0),
                     (180.85, 38.51, 38.51, 30.0, 30.0, 8.0),
                     PHYSX_INERTIA)
    assert v["added_mass"] == X
    assert v["inertia"] == Z
    assert len(set(v.values())) > 1


# --- the shipped configs ---------------------------------------------------


def test_shipped_pkrc_hydro_disagrees_with_its_own_mesh():
    """Pins the KNOWN defect. If this starts failing someone corrected the shipped config —
    fine, but rb_train_model_7998.pt's dynamics no longer match and this test plus
    PKRCHydrodynamicsCfgZSlender should be collapsed into one."""
    v = _verdicts(PKRCHydrodynamicsCfg())
    assert v["added_mass"] == X, "coefficients claim an x-slender hull"
    assert v["inertia"] == Z, "the USD mesh describes a z-slender one"
    assert not slender_axis_is_consistent(
        PKRCHydrodynamicsCfg().added_mass, PKRCHydrodynamicsCfg().linear_damping,
        PKRCHydrodynamicsCfg().quadratic_damping, PHYSX_INERTIA,
    )


def test_z_slender_variant_is_consistent():
    cfg = PKRCHydrodynamicsCfgZSlender()
    v = _verdicts(cfg)
    assert v == {"added_mass": Z, "drag": Z, "inertia": Z}, v


def test_z_slender_leaves_the_measured_axes_untouched():
    """surge/sway came from steady-state tow tests; only the copied heave entry is corrected."""
    base, fixed = PKRCHydrodynamicsCfg(), PKRCHydrodynamicsCfgZSlender()
    for name in ("linear_damping", "quadratic_damping"):
        b, f = getattr(base, name), getattr(fixed, name)
        assert f[0] == b[0], f"{name}: surge is measured, must not move"
        assert f[1] == b[1], f"{name}: sway is measured, must not move"
        assert f[2] == pytest.approx(b[1] / 2.4, rel=1e-3), f"{name}: heave = sway / 2.4"
        assert tuple(f[3:]) == tuple(b[3:]), f"{name}: rotational entries unchanged"


def test_z_slender_added_mass_is_a_permutation_of_the_original():
    """Only the axis assignment changes; the analytic magnitudes are inherited as-is."""
    base = torch.as_tensor(PKRCHydrodynamicsCfg().added_mass, dtype=torch.float64)
    fixed = torch.as_tensor(PKRCHydrodynamicsCfgZSlender().added_mass, dtype=torch.float64)
    assert sorted(base[:3].tolist()) == pytest.approx(sorted(fixed[:3].tolist()))
    assert fixed[2] == pytest.approx(float(base[:3].min())), "the small entry lands on heave"
    assert torch.allclose(base[3:], fixed[3:])


def test_z_slender_pins_the_coriolis_inertia_to_the_physx_tensor():
    """The fallback (added_mass[3:6]*0.5 = 0.25) is 5.6x too small in roll/pitch."""
    from marinelab.core.parameters import default_rigid_inertia

    assert default_rigid_inertia(PKRCHydrodynamicsCfg()) == [0.25, 0.25, 0.25]
    assert default_rigid_inertia(PKRCHydrodynamicsCfgZSlender()) == pytest.approx(
        list(PHYSX_INERTIA)
    )


def test_corrected_heave_roughly_halves_the_descent_thrust():
    """The wallscan's primary axis: quantifies what the misassignment cost."""
    net_buoy = 4.90
    v = 0.2
    def drag(cfg):
        return cfg.linear_damping[2] * v + cfg.quadratic_damping[2] * v * v
    before = drag(PKRCHydrodynamicsCfg()) + net_buoy
    after = drag(PKRCHydrodynamicsCfgZSlender()) + net_buoy
    assert before == pytest.approx(30.3, abs=0.3)
    assert after == pytest.approx(15.5, abs=0.3)
    assert before / after == pytest.approx(2.0, abs=0.1)
