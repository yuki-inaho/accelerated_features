## XFeat: Accelerated Features for Lightweight Image Matching
[Guilherme Potje](https://guipotje.github.io/) · [Felipe Cadar](https://eucadar.com/) · [Andre Araujo](https://andrefaraujo.github.io/) · [Renato Martins](https://renatojmsdh.github.io/) · [Erickson R. Nascimento](https://homepages.dcc.ufmg.br/~erickson/)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat_matching.ipynb)  
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/qubvel-hf/xfeat)

### [[ArXiv]](https://arxiv.org/abs/2404.19174) | [[Project Page]](https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/) |  [[CVPR'24 Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.html)

- Training code is now available -> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/XFeat_training_example.ipynb)
- 🎉 **New!** XFeat + LighterGlue (smaller version of LightGlue) available! 🚀 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat%2Blg_torch_hub.ipynb)

<div align="center" style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
  <div style="display: flex; justify-content: space-around; width: 100%;">
    <img src='./figs/xfeat.gif' width="400"/>
    <img src='./figs/sift.gif' width="400"/>
  </div>
  
  Real-time XFeat demonstration (left) compared to SIFT (right) on a textureless scene. SIFT cannot handle fast camera movements, while XFeat provides robust matches under adverse conditions, while being faster than SIFT on CPU.
  
</div>

**TL;DR**: Really fast learned keypoint detector and descriptor. Supports sparse and semi-dense matching.

Just wanna quickly try on your images? Check this out: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat_torch_hub.ipynb) [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/qubvel-hf/xfeat)

## Table of Contents
- [Introduction](#introduction) <img align="right" src='./figs/xfeat_quali.jpg' width=360 />
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Real-time demo app](#real-time-demo)
- [XFeat+LightGlue](#xfeat-with-lightglue)
- [Contribute](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
This repository contains the official implementation of the paper: *[XFeat: Accelerated Features for Lightweight Image Matching](https://arxiv.org/abs/2404.19174)*, to be presented at CVPR 2024.

**Motivation.** Why another keypoint detector and descriptor among dozens of existing ones? We noticed that the current trend in the literature focuses on accuracy but often neglects compute efficiency, especially when deploying these solutions in the real-world. For applications in mobile robotics and augmented reality, it is critical that models can run on hardware-constrained computers. To this end, XFeat was designed as an agnostic solution focusing on both accuracy and efficiency in an image matching pipeline.

**Capabilities.**
- Real-time sparse inference on CPU for VGA images (tested on laptop with an i5 CPU and vanilla pytorch);
- Simple architecture components which facilitates deployment on embedded devices (jetson, raspberry pi, custom AI chips, etc..);
- Supports both sparse and semi-dense matching of local features;
- Compact descriptors (64D);
- Performance comparable to known deep local features such as SuperPoint while being significantly faster and more lightweight. Also, XFeat exhibits much better robustness to viewpoint and illumination changes than classic local features as ORB and SIFT;
- Supports batched inference if you want ridiculously fast feature extraction. On VGA sparse setting, we achieved about 1,400 FPS using an RTX 4090.
- For single batch inference on GPU (VGA), one can easily achieve over 150 FPS while leaving lots of room on the GPU for other concurrent tasks.

##

**Paper Abstract.** We introduce a lightweight and accurate architecture for resource-efficient visual correspondence. Our method, dubbed XFeat (Accelerated Features), revisits fundamental design choices in convolutional neural networks for detecting, extracting, and matching local features. Our new model satisfies a critical need for fast and robust algorithms suitable to resource-limited devices. In particular, accurate image matching requires sufficiently large image resolutions -- for this reason, we keep the resolution as large as possible while limiting the number of channels in the network. Besides, our model is designed to offer the choice of matching at the sparse or semi-dense levels, each of which may be more suitable for different downstream applications, such as visual navigation and augmented reality. Our model is the first to offer semi-dense matching efficiently, leveraging a novel match refinement module that relies on coarse local descriptors. XFeat is versatile and hardware-independent, surpassing current deep learning-based local features in speed (up to 5x faster) with comparable or better accuracy, proven in pose estimation and visual localization. We showcase it running in real-time on an inexpensive laptop CPU without specialized hardware optimizations.

**Overview of XFeat's achitecture.**
XFeat extracts a keypoint heatmap $\mathbf{K}$, a compact 64-D dense descriptor map $\mathbf{F}$, and a reliability heatmap $\mathbf{R}$. It achieves unparalleled speed via early downsampling and shallow convolutions, followed by deeper convolutions in later encoders for robustness. Contrary to typical methods, it separates keypoint detection into a distinct branch, using $1 \times 1$ convolutions on an $8 \times 8$ tensor-block-transformed image for fast processing, being one of the few current learned methods that decouples detection & description and can be processed independently.

<img align="center" src="./figs/xfeat_arq.png" width=1000 />


## Timing Analyses on CPU.

We show that both detection branch & match refinement module costs are small and bring significant advantages in accuracy (please check the ablation section in the paper).

<img align="center" src="./figs/timings.png" width=840 />


Furthermore, XFeat performs effectively in both indoor and outdoor scenes, achieving an excellent compute-accuracy trade-off as demonstrated below. Note that in the paper, the teaser figure has a VGA resolution on the x-axis and 1,200 pixels on the y-axis. Below, we present an updated figure for improved clarity, maintaining the same x-y axis resolution.

<img align="center" src="./figs/speed_accuracy.png" width=840 />


## Installation
XFeat has minimal dependencies, only relying on torch. Also, XFeat does not need a GPU for real-time sparse inference (vanilla pytorch w/o any special optimization), unless you run it on high-res images. If you want to run the real-time matching demo, you will also need OpenCV.
We recommend using conda, but you can use any virtualenv of your choice.
If you use conda, just create a new env with:
```bash
git clone https://github.com/verlab/accelerated_features.git
cd accelerated_features

#Create conda env (Python 3.10-3.12)
conda create -n xfeat python=3.12
conda activate xfeat
```

Then, install [pytorch (>=2.2)](https://pytorch.org/get-started/locally/) for your platform and GPU, and then the remaining inference and demo dependencies:
```bash
#CPU only example; for GPU pick the build that matches your CUDA version on the pytorch website.
pip install torch --index-url https://download.pytorch.org/whl/cpu

#Inference/demo dependencies (GUI OpenCV, typed inference API, LighterGlue)
pip install -r requirements.txt
```

### Quickstart with uv (recommended)
This repository is also set up as a [uv](https://docs.astral.sh/uv/) project with a committed `uv.lock`, so the exact dependency set is reproducible on any machine. Python 3.12 is used by default.

```bash
git clone https://github.com/verlab/accelerated_features.git
cd accelerated_features

# Create the .venv and install the core inference dependencies (plus dev tools)
uv sync

# Optional: also install the evaluation and training dependency groups
uv sync --all-groups   # recommended if you plan to run benchmarks or training
```

`uv sync` is exclusive: it removes packages from any group you leave out. Opt into only what you need with `--group eval` (poselib, gdown, h5py, matplotlib, pandas) or `--group train` (torchvision, tensorboard, hydra-core, omegaconf, ...).

The pretrained weights are already included in `weights/`, so inference works right away. CUDA is used automatically when available; pass `device="cpu"` (or `--device cpu`) to force CPU execution.

## Usage

For your convenience, we provide ready to use notebooks for some examples.

|            **Description**     |  **Notebook**                     |
|--------------------------------|-------------------------------|
| Minimal example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/minimal_example.ipynb) |
| Matching & registration example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat_matching.ipynb) |
| Torch hub example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat_torch_hub.ipynb) |
| Training example (synthetic) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/XFeat_training_example.ipynb) |
| XFeat + LightGlue | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat%2Blg_torch_hub.ipynb) |


### Inference
To run XFeat on an image, three lines of code is enough:
```python
from modules.xfeat import XFeat

xfeat = XFeat()

#Simple inference with batch sz = 1
output = xfeat.detectAndCompute(torch.randn(1,3,480,640), top_k = 4096)[0]
```
Or you can use this [script](./minimal_example.py) in the root folder:
```bash
python3 minimal_example.py
```

If you already have pytorch, simply use torch hub if you like it:
```python
import torch

xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)

#Simple inference with batch sz = 1
output = xfeat.detectAndCompute(torch.randn(1,3,480,640), top_k = 4096)[0]
```

### Inference on your own images
[`inference.py`](./inference.py) matches two images end to end with the bundled pretrained weights (`weights/xfeat.pt` by default) and verifies the matches with a RANSAC homography:

```bash
# Sparse XFeat features with mutual-nearest-neighbor matching (default)
uv run inference.py path/to/image1.jpg path/to/image2.jpg --output matches.png

# Semi-dense XFeat* with match refinement
uv run inference.py path/to/image1.jpg path/to/image2.jpg --method xfeat-star

# LighterGlue matcher (kornia based, pretrained weights included)
uv run inference.py path/to/image1.jpg path/to/image2.jpg --method lighterglue
```

It prints the number of matches/inliers and an estimated homography, and optionally saves a match visualization. Useful flags: `--device cpu|cuda|mps`, `--weights path/to/checkpoint.pt`, `--lg-weights path/to/lighterglue_bundle.pt` (a fine-tuned LighterGlue bundle with its own XFeat extractor), `--top-k`, `--max-size`, `--ransac-thr`. Running `uv run inference.py` without arguments matches the sample pair in `assets/`.

The public inference API (`modules/xfeat.py`, `modules/lighterglue.py`, `inference.py`) validates its inputs at runtime through [beartype](https://github.com/beartype/beartype) and [jaxtyping](https://github.com/patrick-kidder/jaxtyping) annotations, so malformed images fail fast with a clear error instead of a cryptic tensor shape mismatch.

### Training
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/XFeat_training_example.ipynb)

To train XFeat as described in the paper, you will need MegaDepth & COCO_20k subset of COCO2017 dataset.
You can obtain the full COCO2017 train data at https://cocodataset.org/.
However, we [make available](https://drive.google.com/file/d/1ijYsPq7dtLQSl-oEsUOGH1fAy21YLc7H/view?usp=drive_link) a subset of COCO for convenience. We simply selected a subset of 20k images according to image resolution. Please check COCO [terms of use](https://cocodataset.org/#termsofuse) before using the data.

To reproduce the training setup from the paper, please follow the steps:
1. Download [COCO_20k](https://drive.google.com/file/d/1ijYsPq7dtLQSl-oEsUOGH1fAy21YLc7H/view?usp=drive_link) containing a subset of COCO2017;
2. Download MegaDepth dataset. You can follow [LoFTR instructions](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md#download-datasets), we use the same standard as LoFTR. Then put the megadepth indices inside the MegaDepth root folder following the standard below:
```bash
{megadepth_root_path}/train_data/megadepth_indices #indices
{megadepth_root_path}/MegaDepth_v1 #images & depth maps & poses
```
3. Finally you can call training
```bash
python3 -m modules.training.train --training_type xfeat_default  --megadepth_root_path <path_to>/MegaDepth --synthetic_root_path <path_to>/coco_20k --ckpt_save_path /path/to/ckpts
```

### Fine-tuning on posed RGB-D sequences

The `xfeat_training` package fine-tunes XFeat and LighterGlue on RGB-D sequences with known camera poses. It uses the [AMUSE](https://github.com/kjeiun/amuse) optimizer (vendored unmodified in `third_party/amuse/`), [Hydra](https://hydra.cc/) configs in `configs/` and TensorBoard logging, and was run on a single RTX 2070 (8 GB) in fp32.

Expected data layout (`colmap_rgbd_v1`): each subset directory holds `dataset.json` and `scenes/scene_000000/{cameras.npz,sequences.npz,overlap.npz,rgb/,depth/}` with `frame_{id:06d}.png` images, uint16 depth in millimetres (0 = invalid) and OpenCV world-to-camera extrinsics. Run everything from the repository root after `uv sync --all-groups`; the examples write to the git-ignored `temp/` directory, which is also where `configs/train.yaml` looks for pairs and caches by default.

1. Mine pairs once with the dedicated config `configs/pair_mining/default.yaml` (chunk splits with guard chunks, depth-reprojection overlap, pairs beyond adjacent frames). Training only reads the resulting files.
   ```bash
   export L76_DATA_ROOT=/path/to/dataset   # directory holding the colmap_rgbd_* subsets
   uv run python -m scripts.mine_pairs --cfg job --resolve   # print the resolved config
   uv run python -m scripts.mine_pairs output_dir=temp/l76_run/pairs_v1   # optional: archive_path=/path/to/dataset.tar.zst records the source archive hash
   ```
2. Cache frozen XFeat features for the mined frames, fix the evaluation anchors/ground truth, and measure the pretrained baseline on val.
   ```bash
   uv run python -m scripts.cache_features --pairs temp/l76_run/pairs_v1 --weights weights/xfeat.pt --output temp/l76_run/cache_official
   uv run python -m scripts.evaluate_l76 --mode prepare --pairs temp/l76_run/pairs_v1 --cache temp/l76_run/cache_official --output temp/l76_run/eval_official_5pct
   uv run python -m scripts.evaluate_l76 --mode baseline --split val --output temp/l76_run/baseline_val
   ```
3. Train in stages with `python -m xfeat_training.train` and Hydra overrides. `experiment=smoke` is a 12-update check on 8 pairs.
   ```bash
   # A: LighterGlue on the frozen official XFeat
   uv run python -m xfeat_training.train experiment=lg_a run_dir=temp/l76_run/stage_a
   # B: XFeat descriptor, reliability, keypoint and fine heads
   uv run python -m xfeat_training.train experiment=xfeat_b run_dir=temp/l76_run/stage_b
   # C: LighterGlue from Stage A on top of the frozen Stage B extractor
   uv run python -m scripts.cache_features --pairs temp/l76_run/pairs_v1 --weights temp/l76_run/stage_b/exports/best_xfeat.pt --output temp/l76_run/cache_stage_b
   uv run python -m xfeat_training.train experiment=lg_c run_dir=temp/l76_run/stage_c cache_dir=temp/l76_run/cache_stage_b xfeat_weights=temp/l76_run/stage_b/exports/best_xfeat.pt lg_weights=temp/l76_run/stage_a/exports/best_lighterglue.pt
   # Optional: AdamW control with the same initial weights and data order as Stage A
   uv run python -m xfeat_training.train experiment=lg_a optimizer=adamw run_dir=temp/l76_run/adamw_control
   ```
   Each run validates on the fixed val pairs every `eval_every` updates and writes `resolved.yaml`, `metrics.jsonl`, TensorBoard events, checkpoints and inference exports (`exports/{best,last}_xfeat.pt`, plus `exports/{best,last}_lighterglue.pt` for LighterGlue runs). Checkpoints keep the top `checkpoint_keep_best` (default 3) by val plus the latest; `null` keeps all. `run_dir` must not exist yet, and Hydra multirun is rejected.
4. Stop and resume. `stop_after_steps` ends the process at that update after saving a checkpoint; resume into a new `run_dir` with the same settings. Configuration, data/cache/weight hashes and the runtime source are checked before anything is written, and the resumed run reproduces the uninterrupted one on the same machine.
   ```bash
   uv run python -m xfeat_training.train experiment=smoke task=lighterglue run_dir=temp/l76_run/smoke_lg_cut max_steps=12 stop_after_steps=5 save_every=5 optimizer.warmup_steps=3
   uv run python -m xfeat_training.train experiment=smoke task=lighterglue run_dir=temp/l76_run/smoke_lg_resume max_steps=12 save_every=5 optimizer.warmup_steps=3 resume_from=temp/l76_run/smoke_lg_cut/checkpoints/step_000005.pt
   ```
   To apply checkpoint retention to a run that has already exited: `uv run python -m scripts.prune_checkpoints temp/l76_run/stage_a --keep-best 3 --training-exited`.
5. Monitor with `uv run tensorboard --logdir temp/l76_run --host 127.0.0.1 --port 6006`.
6. Use the exports. The LighterGlue bundle contains its XFeat extractor; if `--weights` is also given it must be the same extractor.
   ```bash
   uv run inference.py image1.png image2.png --method lighterglue --lg-weights temp/l76_run/stage_c/exports/best_lighterglue.pt
   uv run inference.py image1.png image2.png --weights temp/l76_run/stage_b/exports/best_xfeat.pt
   ```
7. Compare candidates chosen on val once on the test split. `selection.json` lists each candidate's `name`, `xfeat_weights`, `lg_weights` and their SHA-256 `file_hashes`, together with `"selected_on": "val"` and the `evaluation_hash` of the evaluation cache.
   ```bash
   uv run python -m scripts.evaluate_l76 --mode compare --split test --selection temp/l76_run/selection.json --output temp/l76_run/comparison_test
   ```

Fine-tuning on a narrow domain can improve matching there while degrading general scenes considerably. Keep the pretrained weights for general use and evaluate the fine-tuned weights on your own target domain.

## Optional ranking and covariance heads

The [XFeat ranking/covariance extension](docs/raco.md) adds independently trainable
heads, a Hydra training entry point, portable inference bundles, and NPZ extraction.

Validation-selected fine-tuned weights are available in the
[RGB-D + RaCo models release](https://github.com/yuki-inaho/accelerated_features/releases/tag/rgbd-raco-v1).
See [model formats, loading examples, and limitations](docs/pretrained_models.md).

### Evaluation
----
**MegaDepth-1500**

Please note that due to the stochastic nature of RANSAC and major code refactoring, you may observe slightly different AuC results; however, they should be very close to those reported in the paper.

To evaluate on the MegaDepth dataset, you need to first get the dataset:
```bash
python3 -m modules.dataset.download --megadepth-1500 --download_dir </path/to/desired/folder>
```
Then, you call the mega1500 eval script, you can choose between `xfeat, xfeat-star and alike`. It should take about a minute to run the benchmark:
```bash
python3 -m modules.eval.megadepth1500 --dataset-dir </data/Mega1500> --matcher xfeat --ransac-thr 2.5
```
---
**ScanNet-1500**

To evaluate on the ScanNet eval dataset, you need to first get the dataset:
```bash
python3 -m modules.dataset.download --scannet-1500 --download_dir </path/to/desired/folder>
```

Then, you can call the scannet1500 eval script, it should take a couple of minutes:
```bash
python3 -m modules.eval.scannet1500 --scannet_path </data/ScanNet1500> --output </data/ScanNet1500/output> && python3 -m modules.eval.scannet1500 --scannet_path </data/ScanNet1500> --output </data/ScanNet1500/output> --show
```

---

## Real-time Demo
To demonstrate the capabilities of XFeat, we provide a real-time matching demo with Homography registration. Currently, you can experiment with XFeat, ORB and SIFT. You will need a working webcam. To run the demo and show the possible input flags, please run:
```bash
python3 realtime_demo.py -h
```

Don't forget to press 's' to set a desired reference image. Notice that the demo only works correctly for planar scenes and rotation-only motion, because we're using a homography model.

If you want to run the demo with XFeat, please run:
```bash
python3 realtime_demo.py --method XFeat
```

Or test with SIFT or ORB:
```bash
python3 realtime_demo.py --method SIFT
python3 realtime_demo.py --method ORB
```

## XFeat with LightGlue
We have trained a lighter version of LightGlue (LighterGlue). It has fewer parameters and is approximately three times faster than the original LightGlue. Special thanks to the developers of the [GlueFactory](https://github.com/cvg/glue-factory) library, which enabled us to train this version of LightGlue with XFeat.
Below, we compare the original SP + LG using the [GlueFactory](https://github.com/cvg/glue-factory) evaluation script on MegaDepth-1500.
Please follow the example to test on your own images:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/verlab/accelerated_features/blob/main/notebooks/xfeat%2Blg_torch_hub.ipynb)

Metrics (AUC @ 5 / 10 / 20)
| Setup           | Max Dimension | Keypoints | XFeat + LighterGlue           | SuperPoint + LightGlue (Official) |
|-----------------|---------------|-----------|-------------------------------|-----------------------------------|
| **Fast**  | 640           | 1300      | 0.444 / 0.610 / 0.746       | 0.469 / 0.633 / 0.762          |
| **Accurate** | 1024          | 4096      | 0.564 / 0.710 / 0.819       | 0.591 / 0.738 / 0.841            |

## Development

The typed inference path uses [jaxtyping](https://github.com/patrick-kidder/jaxtyping) shape annotations with [beartype](https://github.com/beartype/beartype) runtime validation (`modules/typecheck.py`). Quality gates are pinned in `pyproject.toml`:

```bash
uv run pytest          # end-to-end inference tests, regression tests and RGB-D training tests (geometry, mining, losses, resume, retention)
uv run ruff check .    # lint; legacy research scripts keep the upstream style through documented per-file ignores
uv run ty check        # static type checking of the inference path, xfeat_training, scripts and tests
```

Set `XFEAT_TEST_DEVICE=cpu` (or `cuda`, `mps`) to pin the device used by the test suite.

## Contributing
Contributions to XFeat are welcome! 
Currently, it would be nice to have an export script to efficient deployment engines such as TensorRT and ONNX. Also, it would be cool to train other lightweight learned matchers on top of XFeat local features.

## Citation
If you find this code useful for your research, please cite the paper:

```bibtex
@INPROCEEDINGS{potje2024cvpr,
  author={Potje, Guilherme and Cadar, Felipe and Araujo, André and Martins, Renato and Nascimento, Erickson R.},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={XFeat: Accelerated Features for Lightweight Image Matching}, 
  year={2024},
  pages={2682-2691},
  keywords={Visualization;Accuracy;Image matching;Pose estimation;Feature extraction;Hardware;Real-time systems;Image matching;Local features;Lightweight;Fast},
  doi={10.1109/CVPR52733.2024.00259}}
```

## License
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Acknowledgements
- We thank the agencies CAPES, CNPq, and Google for funding different parts of this work.
- We thank the developers of Kornia for the [kornia library](https://github.com/kornia/kornia)!

**VeRLab:** Laboratory of Computer Vison and Robotics https://www.verlab.dcc.ufmg.br
<br>
<img align="left" width="auto" height="50" src="./figs/ufmg.png">
<img align="right" width="auto" height="50" src="./figs/verlab.png">
<br/>
