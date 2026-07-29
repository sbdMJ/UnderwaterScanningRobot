# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""marinelab: underwater-vehicle environments and shared marine physics for Isaac Lab."""

# Importing the tasks subpackage triggers gym.register() calls (self-registration).
from . import tasks  # noqa: F401
