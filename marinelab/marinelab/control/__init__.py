# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Pure controller layer for the wallscan competitor experiments (sim2real seam).

No isaaclab/pxr imports anywhere in this package — same rule as the pkrc_wallscan pure
modules, and enforced the same way: submodules are resolved lazily so importing the package
costs nothing and pulls no simulator. See ``docs/experiments/competitor_framework_plan.md`` §2-3.
"""
from __future__ import annotations

import importlib

_LAZY = {
    "VehicleState": ".types",
    "ScanReference": ".types",
    "ControlOutput": ".types",
    "WallScanController": ".base",
    "ControllerStats": ".base",
    "FixedWeightNMPC": ".fixed_nmpc",
    "load_weight_params": ".fixed_nmpc",
    "DiffWMPCController": ".diff_wmpc_ctrl",
    "PPOPolicyController": ".ppo_policy",
    "SensorSample": ".estimator",
    "WallFrameStateEstimator": ".estimator",
    "TankCalib": ".hw_bridge",
    "TopicSampleAssembler": ".hw_bridge",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
