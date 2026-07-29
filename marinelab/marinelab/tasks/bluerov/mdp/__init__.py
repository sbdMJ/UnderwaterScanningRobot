# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP (Markov Decision Process) functions for BlueROV environments.

This module provides event functions for domain randomization specific to
underwater vehicles. ``randomize_hydrodynamics`` is invoked directly from the
environment reset path (BlueROVEnv._reset_idx).
"""

from .events import randomize_hydrodynamics

__all__ = [
    "randomize_hydrodynamics",
]
