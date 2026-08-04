# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Guard: every shipped thruster allocation matrix must be geometrically realizable.

``tau = r x F`` is perpendicular to ``F``, so each TAM column must satisfy
``(Mx, My, Mz) . normalize(Fx, Fy, Fz) == 0`` no matter where the thruster sits or how it
is canted. This catches the exact defect found on 2026-07-30, where PKRC's sway moment arm
was transcribed into the ``My`` row although the sway force is pure ``+y`` — which cost the
project a wrong tilt conclusion (uncancellable pitch instead of a cancellable roll) and a
needlessly halved sway speed. See ``PKRCThrusterCfgFixedTAM`` for the measurements.

The real asset cfgs are read here (not mirrored copies), so editing a TAM in
``marinelab/assets/`` is what this test actually protects.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

from marinelab.core.thruster import allocation_moment_residual

_REPO = Path(__file__).resolve().parent.parent


def _install_asset_cfg_stubs():
    """Make ``marinelab.assets.*`` importable without Isaac Sim.

    ``conftest`` replaces ``marinelab.assets`` with a bare stub (so ``core`` can import
    ``ThrusterCfg``), which blocks submodule resolution. Giving that stub a real
    ``__path__`` plus permissive stand-ins for the handful of ``isaaclab.sim`` /
    ``isaaclab.assets`` constructors the vehicle cfgs touch is enough: those constructors'
    results are only ever stored on config fields, never called, in the code under test.
    """
    assets = sys.modules["marinelab.assets"]
    if not hasattr(assets, "__path__"):
        assets.__path__ = [str(_REPO / "marinelab" / "assets")]

    class _Cfg:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

        def __getattr__(self, name):
            return _Cfg

        def replace(self, **kwargs):
            return self

    for name in ("isaaclab.sim", "isaaclab.assets"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sim = sys.modules["isaaclab.sim"]
    for name in (
        "UsdFileCfg", "RigidBodyPropertiesCfg", "ArticulationRootPropertiesCfg", "CylinderCfg",
        "PreviewSurfaceCfg", "CollisionPropertiesCfg", "RigidBodyMaterialCfg",
    ):
        if not hasattr(sim, name):
            setattr(sim, name, _Cfg)
    art = sys.modules["isaaclab.assets"]
    if not hasattr(art, "ArticulationCfg"):
        class ArticulationCfg(_Cfg):
            class InitialStateCfg(_Cfg):
                pass

        art.ArticulationCfg = ArticulationCfg


_install_asset_cfg_stubs()

from marinelab.assets.pkrc.pkrc import PKRCThrusterCfg, PKRCThrusterCfgFixedTAM  # noqa: E402
from marinelab.assets.uuv_cfg import ThrusterCfg as BaseThrusterCfg  # noqa: E402

SWAY_COLUMNS = (2, 3)


# ---------------------------------------------------------------------------
# The invariant itself
# ---------------------------------------------------------------------------


def test_residual_is_zero_for_a_hand_built_realizable_tam():
    """Build a TAM from explicit (r, F) pairs; r x F can never violate the invariant."""
    torch.manual_seed(0)
    r = torch.randn(5, 3, dtype=torch.float64)
    f = torch.randn(5, 3, dtype=torch.float64)
    f = f / f.norm(dim=-1, keepdim=True)
    m = torch.cross(r, f, dim=-1)
    tam = torch.cat([f.T, m.T], dim=0)  # (6, 5)
    assert torch.allclose(allocation_moment_residual(tam), torch.zeros(5, dtype=torch.float64), atol=1e-12)


def test_residual_flags_a_moment_that_no_placement_can_produce():
    tam = torch.zeros(6, 1, dtype=torch.float64)
    tam[1, 0] = 1.0  # pure +y force
    tam[4, 0] = 0.09  # ... with a pitch moment: impossible
    assert float(allocation_moment_residual(tam)[0]) == pytest.approx(0.09)


def test_residual_flags_a_moment_with_no_force_behind_it():
    tam = torch.zeros(6, 1, dtype=torch.float64)
    tam[3, 0] = 0.2
    assert float(allocation_moment_residual(tam)[0]) == pytest.approx(0.2)


def test_residual_rejects_wrong_shape():
    with pytest.raises(ValueError):
        allocation_moment_residual(torch.zeros(5, 6))


# ---------------------------------------------------------------------------
# The shipped configs
# ---------------------------------------------------------------------------


def test_fixed_pkrc_tam_is_realizable():
    res = allocation_moment_residual(PKRCThrusterCfgFixedTAM.allocation_matrix)
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-12), f"residual per thruster: {res.tolist()}"


def test_bluerov_default_tam_is_realizable():
    """The inherited BlueROV2-Heavy layout passes, so the invariant is not over-strict."""
    res = allocation_moment_residual(BaseThrusterCfg.allocation_matrix)
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-9), f"residual per thruster: {res.tolist()}"


def test_shipped_pkrc_tam_violates_the_invariant_at_the_sway_columns():
    """Pins the KNOWN defect rather than hiding it.

    ``PKRCThrusterCfg`` is deliberately kept unfixed so ``checkpoints/rb_train_model_7998.pt``
    stays reproducible. If this test ever starts failing, someone corrected the matrix in
    place — that is fine, but the checkpoint's published tilt numbers no longer apply and
    this test plus ``PKRCThrusterCfgFixedTAM`` should be collapsed into one.
    """
    res = allocation_moment_residual(PKRCThrusterCfg.allocation_matrix)
    for col in SWAY_COLUMNS:
        assert float(res[col]) == pytest.approx(0.09), "the misplaced sway moment arm"
    others = [i for i in range(res.shape[0]) if i not in SWAY_COLUMNS]
    assert torch.allclose(res[others], torch.zeros(len(others), dtype=res.dtype), atol=1e-12), (
        "only the sway columns should be affected; anything else is a NEW defect"
    )


def test_the_two_pkrc_tams_differ_only_by_moving_the_sway_arm():
    shipped = torch.as_tensor(PKRCThrusterCfg.allocation_matrix, dtype=torch.float64)
    fixed = torch.as_tensor(PKRCThrusterCfgFixedTAM.allocation_matrix, dtype=torch.float64)
    diff = fixed - shipped
    expected = torch.zeros_like(diff)
    for col in SWAY_COLUMNS:
        expected[3, col] = 0.09   # arm moves INTO the roll row
        expected[4, col] = -0.09  # ... and out of the pitch row
    assert torch.allclose(diff, expected, atol=1e-12)
    # forces, yaw and the heave differential are untouched
    assert torch.allclose(fixed[:3], shipped[:3])
    assert torch.allclose(fixed[5], shipped[5])


def test_fixed_tam_leaves_pitch_unactuated_but_also_unexcited():
    """Pitch loses its (spurious) actuator — acceptable only because nothing drives it either."""
    fixed = torch.as_tensor(PKRCThrusterCfgFixedTAM.allocation_matrix, dtype=torch.float64)
    assert torch.allclose(fixed[4], torch.zeros(6, dtype=torch.float64)), "no thruster produces My"
    assert torch.linalg.matrix_rank(fixed) == 5, "rank drops in the pitch direction only"


def test_heave_differential_authority_covers_the_parasitic_roll():
    """dF = -(arm_sway/arm_heave) * F_y, at both the 0.1 and 0.2 m/s legs, inside 40 N."""
    fixed = torch.as_tensor(PKRCThrusterCfgFixedTAM.allocation_matrix, dtype=torch.float64)
    arm_sway = float(fixed[3, SWAY_COLUMNS[0]])
    arm_heave = float(fixed[3, 4])
    for v_sway, expected_dF in ((0.123, 8.6), (0.2, 14.3)):
        f_y = 119.44 * v_sway + 38.51 * v_sway**2
        dF = arm_sway * f_y / arm_heave
        assert dF == pytest.approx(expected_dF, abs=0.3)
        assert dF < 40.0, "must fit inside one thruster's authority"
