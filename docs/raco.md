# XFeat ranking / covariance heads

Frozen sparse XFeat can now predict a per-keypoint ranking score and a symmetric
positive definite 2×2 positional error matrix, inspired by
[RaCo](https://github.com/cvg/RaCo). This extension uses XFeat features and its own
training objective; RaCo checkpoints are not compatible.

## Outputs and coordinates

`modules.raco.XFeatRaCo.extract()` returns one dictionary per image:

| Key | Shape | Meaning |
|---|---|---|
| `keypoints` | N×2 | original image `(x,y)`, integer pixel centers before resize |
| `descriptors` | N×64 | normalized XFeat descriptors |
| `scores`, `keypoint_scores` | N | original detector × reliability score |
| `ranker_scores` | N | learned priority; may be negative, not match probability |
| `covariances` | N×2×2 | effective positional error matrix, original pixel² |
| `candidate_ids` | N | identity within the original candidate pool |
| `image_size` | 2 | original `(width,height)` |

N may be less than the requested K, including zero. All fields are permuted
together. Ranking never determines whether a candidate is valid. Candidate
generation uses fixed XFeat NMS and the original score, with a maximum of 8192
points and a four-pixel valid-support margin.

The two independent heads add 34,688 parameters. They read the unnormalized
64-channel fused feature and the 64-channel detector feature immediately before
the existing 65-channel output layer. Channel LayerNorm does not pool spatial
statistics. The sparse final projection equals dense PixelShuffle at integer
candidate positions. Only selected points need covariance computation at inference.

Input tensors are finite float BCHW RGB or grayscale in `[0,1]`. Images are resized
once to multiples of 32. Sampling consistently uses `align_corners=False`; coordinate
restoration includes the half-pixel translation and covariance uses the inverse
resize Jacobian. No extra stride-squared factor is applied.

The original `XFeat` API/checkpoints retain their existing behavior. The new API
uses a corrected sampling convention, so compare head-on/head-off using this API
to isolate head effects; differences from the original API also include sampling.

## Training on prepared L76 data

Use the repository's uv `train`, `eval`, and `dev` dependency groups. Prepare the
L76 pair manifest with the existing mining workflow. Both tasks use training
frames to generate independent photometric/projective views with known relative
homographies. They never use validation/test frames for updates. Validation uses
four fixed held-out source images and a separate fixed augmentation seed.

```bash
# New heads are initialized; existing XFeat weights are transferred non-strictly.
# A canonical XFeat+LighterGlue export is also accepted as xfeat_weights.
uv run --no-sync python -m xfeat_training.raco_train \
  pairs_dir=runs/pairs xfeat_weights=runs/stage_c/exports/best_lighterglue.pt \
  run_dir=runs/rank_smoke task=raco_rank

# Preserve the learned ranker while training covariance independently.
uv run --no-sync python -m xfeat_training.raco_train \
  pairs_dir=runs/pairs xfeat_weights=runs/stage_c/exports/best_lighterglue.pt \
  init_bundle=runs/rank_smoke/exports/best_raco.pt \
  run_dir=runs/cov_smoke task=raco_covariance
```

Defaults run 12 successful updates with AdamW, fp32, batch 1, clipping 1, and
validation every 3 updates. The backbone, BN buffers, and inactive head are frozen.
Weight decay excludes biases and LayerNorm coefficients. Learning rate follows a
cosine schedule over `max_steps`; ranking temperature falls from 0.5 to 0.05.

Warm starts call `load_state_dict(strict=False)`. `inputs.json` records all loaded,
missing, and unexpected keys. All frozen XFeat weights must be present with the
right shapes. Only new heads may be missing; unrelated matcher weights are logged
and left unused by this extractor. A new-format bundle and a training resume use
strict loading to preserve their full state.

Ranking maximizes relaxed joint selection of geometric pseudo-matches at budgets
128/256/512/1024/2048/4096. The soft selection conserves the budget and differentiates
the threshold implicitly. All valid candidates compete, including non-overlapping
regions. Covariance uses bidirectional Gaussian NLL with
`S = Sigma_target + J Sigma_source J.T` and a 0.05² pixel² eigenvalue floor.
Geometric MNN thresholds are 3 pixels for rank and 8 for covariance; 4/8/12-pixel
covariance diagnostics are reported separately.

Run directories cannot be overwritten. Resume into a new directory with unchanged
semantic configuration and `resume_from=.../checkpoints/step_000005.pt`. To check
exact resume, keep `max_steps=12`, use `stop_after_steps=5` for the cut run, then
remove that stop override for the resumed run. Changing runtime code/configuration
invalidates compatibility deliberately.

Checkpoints keep the best three validation results plus the latest restart state
and required export references. Rank selects maximum fixed-val hard selection
utility; covariance selects minimum fixed-val NLL. Ties prefer earlier updates.
This extends the existing retention ledger without changing legacy F1/TP ranking.
Metrics, predictions, TensorBoard events, and deletion evidence are retained.

## Inference

### Continuing in 1,000-update blocks

For longer experiments enable `auto_stop.enabled=true`, `eval_every=100`, and
`save_every=100`. Use a fixed cosine horizon (`max_steps=50000` for ranking or
`max_steps=20000` for covariance). The trainer checks every 1,000 successful
updates and stops earlier when validation plateaus; the horizon is a ceiling,
not a promise to use every update.

The most recent three validation means and maxima are compared with the preceding
three. Improvement must exceed 0.001 absolute ranking utility or a 0.01 NLL
decrease in either comparison to continue. Otherwise it stops after writing the
checkpoint, metrics, TensorBoard events, and retention ledger. Training loss means
over the last and preceding 100 updates accompany each `convergence/step_*.json`.
`completed.json` records the stopping reason. Thresholds are configurable before
starting; do not retune them from test results. A plateau on four fixed validation
images is a development stopping heuristic, not a global convergence claim.

```bash
uv run --no-sync python -m scripts.extract_raco \
  --weights runs/cov_smoke/exports/best_raco.pt \
  --image assets/ref.png --output runs/features.npz --device cuda --top-k 1024
```

```python
from modules.raco import XFeatRaCo

model = XFeatRaCo.from_bundle("runs/cov_smoke/exports/best_raco.pt", device="cuda")
features = model.extract(image_tensor, top_k=1024)[0]
```

For a rank-only bundle pass `covariance=False` (CLI: `--no-covariance`). Loading
with an enabled untrained head is rejected. NPZ files contain numeric arrays only.
The standard `keypoints`, `descriptors`, and `image_size` can be passed to the
existing MNN/LighterGlue matcher. Match indices index the selected arrays; use
`candidate_ids` to recover identities before reranking. A trained matcher remains
a separate checkpoint. Covariance-aware pose/PnP/BA requires a separate solver.

## What the diagnostics establish

The smoke verifies finite losses/gradients, head updates, frozen backbone/BN,
checkpoint retention, exact boundary resume, and usable outputs. It does not
establish improved matching/pose performance or calibrated metric uncertainty.
Covariance NLL and coverage are conditional on geometric pseudo-correspondence
selection. Truncation, matching ambiguity, shared image bias and correlated errors
prevent interpreting them as unconditional position uncertainty. Fixed-val curves
on four images are useful development diagnostics, not scene-level generalization.

Further experiments include three seeds, independent covariance calibration,
constant/isotropic/orientation controls, transformed-prediction diagnostics,
larger scene-separated validation, and covariance-weighted pose optimization.
Sparse outputs must not be assigned unchanged to XFeat* pair-dependent refinements.
