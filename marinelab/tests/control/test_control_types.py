# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Round-trip and layout tests for the controller-layer data types."""

import numpy as np

from marinelab.control.types import ControlOutput, ScanReference, VehicleState


def test_x13_roundtrip():
    x = np.arange(13.0)
    st = VehicleState.from_x13(x, stamp=1.5)
    assert st.stamp == 1.5
    np.testing.assert_allclose(st.pos_w, [0, 1, 2])
    np.testing.assert_allclose(st.quat_wb, [3, 4, 5, 6])
    np.testing.assert_allclose(st.lin_vel_b, [7, 8, 9])
    np.testing.assert_allclose(st.ang_vel_b, [10, 11, 12])
    np.testing.assert_allclose(st.to_x13(), x)


def test_frozen_reference_layout():
    ref = ScanReference.frozen(30, z_ref=2.5, s_ref=-1.0, theta_anchor=0.3, s_anchor=4.0, phase=2)
    d = ref.as_dict()
    # exactly the keys param_matrix reads, each (N+1,)
    assert set(d) == {"z_ref", "s_ref", "v_tan_des", "v_z_des"}
    for arr in d.values():
        assert arr.shape == (31,)
    np.testing.assert_allclose(d["z_ref"], 2.5)
    np.testing.assert_allclose(d["v_tan_des"], 0.0)
    assert ref.phase == 2 and ref.theta_anchor == 0.3


def test_control_output_defaults():
    out = ControlOutput(u_cmd=np.zeros(6))
    assert out.status == 0 and out.u_newton is None and out.aux == {}
