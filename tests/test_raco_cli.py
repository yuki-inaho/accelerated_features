import numpy as np
import torch

from modules.raco import XFeatRaCo


def test_extraction_npz_and_existing_matcher_compatibility(tmp_path):
    from scripts.extract_raco import extract_file

    model = XFeatRaCo(candidate_limit=64, detection_threshold=0.001)
    weights = tmp_path / "raco.pt"
    torch.save(model.bundle(trained_heads=["rank", "covariance"]), weights)
    output = tmp_path / "features.npz"
    extract_file(weights, "assets/ref.png", output, device="cpu", top_k=32)
    with np.load(output, allow_pickle=False) as data:
        assert data["keypoints"].shape == (32, 2)
        assert data["descriptors"].shape == (32, 64)
        assert data["covariances"].shape == (32, 2, 2)
        assert all(np.isfinite(data[k]).all() for k in data.files)
    # New fields stay alongside the ordinary XFeat matching fields.
    from modules.lighterglue import LighterGlue

    image = torch.rand(1, 3, 64, 96)
    features = model.extract(image, top_k=32)[0]
    matcher = LighterGlue(device="cpu", flash=False)
    inputs = {
        f"{name}{side}": features[name][None] for side in (0, 1) for name in ("keypoints", "descriptors", "image_size")
    }
    with torch.inference_mode():
        result = matcher(inputs)
    assert result["matches"][0].shape[-1] == 2
