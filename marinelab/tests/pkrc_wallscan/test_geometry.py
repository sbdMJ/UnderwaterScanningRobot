import torch

from marinelab.tasks.pkrc_wallscan.geometry import (
    wall_distance,
    radial_clearance,
    sonar_wall_distance,
)


def test_sonar_mount_offset():
    # at center, sonar 0.3m forward, beam along heading -> wall at radius - 0.3 = 5.7
    d = sonar_wall_distance(
        torch.zeros(1, 2), torch.zeros(1), torch.tensor([[0.3, 0.0]]), 0.0, 6.0
    )
    assert torch.allclose(d, torch.tensor([5.7]), atol=1e-4)


def test_sonar_zero_mount_equals_body():
    # zero mount + zero yaw offset == plain wall_distance from body center
    pos = torch.tensor([[2.0, 0.0]])
    d = sonar_wall_distance(pos, torch.zeros(1), torch.zeros(1, 2), 0.0, 6.0)
    assert torch.allclose(d, wall_distance(pos, torch.zeros(1), 6.0), atol=1e-4)


def test_center_facing_out():
    # at center, any heading -> distance == radius
    d = wall_distance(torch.zeros(1, 2), torch.zeros(1), radius=6.0)
    assert torch.allclose(d, torch.tensor([6.0]), atol=1e-4)


def test_offset_forward_ray():
    # 2 m from center along +x, facing +x -> 4 m to wall
    d = wall_distance(torch.tensor([[2.0, 0.0]]), torch.zeros(1), radius=6.0)
    assert torch.allclose(d, torch.tensor([4.0]), atol=1e-4)


def test_radial_clearance():
    c = radial_clearance(torch.tensor([[5.0, 0.0]]), radius=6.0)
    assert torch.allclose(c, torch.tensor([1.0]), atol=1e-4)
