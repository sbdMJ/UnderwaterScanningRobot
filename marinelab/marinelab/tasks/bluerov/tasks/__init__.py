# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task abstractions for BlueROV environments.

This module provides a task abstraction layer for underwater vehicle environments,
enabling pluggable task definitions without modifying the core environment logic.

The design follows Isaac Lab conventions while adapting to DirectRLEnv patterns.
"""

from .attitude_task import AttitudeTask, AttitudeTaskCfg
from .hover_task import HoverTask, HoverTaskCfg
from .task_base import TaskBase, TaskBaseCfg

__all__ = [
    "TaskBase",
    "TaskBaseCfg",
    "HoverTask",
    "HoverTaskCfg",
    "AttitudeTask",
    "AttitudeTaskCfg",
]
