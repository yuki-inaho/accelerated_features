"""Known homography direction, masks and deterministic photometric sampling."""

from __future__ import annotations

import torch

from xfeat_training.augment import photometric, sample_homography, warp_image


def test_translation_moves_image_and_marks_padding() -> None:
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, 12, 10] = 1
    H = torch.tensor([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    warped, mask = warp_image(image, H)
    torch.testing.assert_close(warped[0, :, 10, 13], torch.ones(3), atol=1e-5, rtol=0)
    assert warped[0, :, 12, 10].abs().max() < 1e-5
    assert not mask[0, 0, :, :3].any()
    assert mask[0, 0, 10, 13]


def test_seed_controls_homography_and_photometric() -> None:
    image = torch.full((1, 3, 32, 32), 0.5)
    a, b = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    assert torch.equal(sample_homography(32, 32, a), sample_homography(32, 32, b))
    first, second = photometric(image, a), photometric(image, b)
    assert torch.equal(first, second) and not torch.equal(first, image)
    assert first.min() >= 0 and first.max() <= 1


def test_identity_preserves_all_pixels() -> None:
    image = torch.rand(1, 3, 32, 64)
    warped, valid = warp_image(image, torch.eye(3))
    torch.testing.assert_close(warped, image, atol=1e-5, rtol=1e-5)
    assert valid.all()


def test_known_quarter_turn_preserves_coordinate_direction() -> None:
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, 12, 10] = 1
    H = torch.tensor([[0.0, -1.0, 31.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    warped, valid = warp_image(image, H)
    torch.testing.assert_close(warped[0, :, 10, 19], torch.ones(3), atol=1e-5, rtol=0)
    assert valid.all()
