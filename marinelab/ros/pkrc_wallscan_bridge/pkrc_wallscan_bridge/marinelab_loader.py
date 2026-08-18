# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Import marinelab's pure modules on a machine with no Isaac Lab (the Jetson).

``marinelab/__init__.py`` eagerly imports the gym envs and, through them, ``isaaclab.sim``
— which does not exist on the vehicle. The estimator bridge only needs the pure layers
(``control.estimator``, ``control.hw_bridge``, ``tasks.pkrc_wallscan.wall_frame_ekf``), so
this loader registers bare package modules with a real ``__path__`` (the same shim
``marinelab/tests/conftest.py`` uses) and the heavy ``__init__`` never runs.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def load_marinelab(root: str | None = None) -> None:
    """Make ``import marinelab.control...`` work from a plain repo checkout.

    ``root`` is the directory CONTAINING the ``marinelab`` python package (i.e. the inner
    ``marinelab/`` of the repo). Defaults to $MARINELAB_ROOT, else walks up from this file
    (which covers the case where this ROS package is symlinked into hero_ws from the repo).
    """
    if root is None:
        root = os.environ.get("MARINELAB_ROOT")
    if root is None:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "marinelab" / "marinelab" / "control" / "hw_bridge.py"
            if candidate.is_file():
                root = str(parent / "marinelab")
                break
    if root is None:
        raise ImportError(
            "cannot locate the marinelab package: set MARINELAB_ROOT to the repo's inner "
            "marinelab/ directory (the one whose marinelab/control/hw_bridge.py exists)")

    pkg_dir = Path(root) / "marinelab"
    if not (pkg_dir / "control" / "hw_bridge.py").is_file():
        raise ImportError(f"{pkg_dir} does not look like the marinelab package")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "marinelab" not in sys.modules:
        pkg = types.ModuleType("marinelab")
        pkg.__path__ = [str(pkg_dir)]
        sys.modules["marinelab"] = pkg
    if "marinelab.tasks" not in sys.modules:
        tasks = types.ModuleType("marinelab.tasks")
        tasks.__path__ = [str(pkg_dir / "tasks")]
        sys.modules["marinelab.tasks"] = tasks
