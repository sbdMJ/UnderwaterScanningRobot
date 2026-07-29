# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit test for the attitude-error sign-correction (B4)."""

import torch


def attitude_error_signed(error_quat: torch.Tensor) -> torch.Tensor:
    """Reference implementation of the corrected attitude error."""
    w = error_quat[:, 0:1]
    vec = error_quat[:, 1:4]
    return 2.0 * torch.sign(w) * vec


def test_sign_correction_keeps_error_continuous_past_180():
    # q and -q are the same rotation; corrected error must agree.
    q = torch.tensor([[0.2, 0.6, 0.0, 0.0]])
    q_neg = -q
    e1 = attitude_error_signed(q)
    e2 = attitude_error_signed(q_neg)
    assert torch.allclose(e1, e2, atol=1e-6)


def test_zero_error_for_identity():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(attitude_error_signed(q), torch.zeros(1, 3))
