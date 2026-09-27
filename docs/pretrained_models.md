# Fine-tuned inference weights

Download the assets from the [RGB-D + RaCo models release](https://github.com/yuki-inaho/accelerated_features/releases/tag/rgbd-raco-v1).

These are validation-selected, domain-fine-tuned inference weights for this fork.
Use the code at release tag `rgbd-raco-v1` or a compatible later revision.

| File | Contents | Load with |
|---|---|---|
| `xfeat-rgbd-best.pt` | Fine-tuned XFeat extractor | `modules.xfeat.XFeat` |
| `xfeat-lighterglue-rgbd-best.pt` | Best LighterGlue matcher and its matching XFeat extractor | `modules.lighterglue.LighterGlue` or `inference.py --lg-weights` |
| `xfeat-raco-rgbd-best.pt` | Same XFeat extractor plus trained ranking and covariance heads | `modules.raco.XFeatRaCo.from_bundle` |

The XFeat extractor in all three files is identical. The matcher bundle uses the
validation-selected extractor/matcher combination. Each added head trained for
2,000 updates; both improved at 1,000 and stopped at the next fixed validation
plateau check. The combined RaCo bundle contains each head's validation best.
Training used one seed. Head validation used four held-out source frames with
fixed synthetic homographies; it does not establish general-domain or pose
accuracy, calibrated uncertainty, or mathematical convergence.

The RaCo-inspired extension is independently implemented for original XFeat;
these are not official RaCo weights. Domain fine-tuning can reduce performance
on unrelated scenes. Keep the original pretrained weights for general use.

## Installation and verification

```bash
uv sync --all-groups
sha256sum -c SHA256SUMS
```

Place the downloaded files in a local `models/` directory for these examples.

```python
from modules.xfeat import XFeat
from modules.raco import XFeatRaCo

extractor = XFeat(weights='models/xfeat-rgbd-best.pt', device='cpu')
enhanced = XFeatRaCo.from_bundle('models/xfeat-raco-rgbd-best.pt', device='cpu')
# features = enhanced.extract(rgb_tensor, top_k=1024)[0]
# fields include keypoints, descriptors, candidate_ids, ranker_scores,
# and covariances (original-image pixel squared).
```

```bash
uv run inference.py image1.png image2.png --method lighterglue --lg-weights models/xfeat-lighterglue-rgbd-best.pt
uv run python -m scripts.extract_raco --weights models/xfeat-raco-rgbd-best.pt --image image.png --output features.npz
```

These files contain inference tensors and, for the head bundle, a small public
architecture configuration. Training data, source-image identifiers, machine
paths, optimizer state and run metadata are excluded. They are inference
exports, not optimizer-resume checkpoints. Private training data is not included.
No privacy-preserving training claim is made for the learned parameters.

See the repository's Apache-2.0 license and upstream attribution. The original
XFeat pretrained assets remain unchanged in the repository.
