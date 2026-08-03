# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""marinelab.algorithms: general-purpose RL/DR algorithms reusable across UUV research.

DORAEMON (Domain Randomization via Entropy Maximization, Tiboni et al. ICLR 2024) is a
robot-agnostic DR-curriculum engine: the caller injects its own parameter definitions
(``param_defs`` / ``nominal_overrides``) so the scheduler works with any robot's DR config.
"""

from .doraemon import (
    BetaDistribution,
    CurriculumReplayer,
    DoraemonCfg,
    DoraemonScheduler,
    EpisodeBuffer,
    ParamSpec,
    build_param_specs,
)

__all__ = [
    "BetaDistribution",
    "CurriculumReplayer",
    "DoraemonCfg",
    "DoraemonScheduler",
    "EpisodeBuffer",
    "ParamSpec",
    "build_param_specs",
]
