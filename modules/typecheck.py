"""
"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

Shared type aliases and runtime type-checking entry points.

Annotations built on `jaxtyping` carry both static (ty) and runtime
(beartype) meaning. The `typechecked` decorator is applied at the public
inference boundary -- `modules/xfeat.py`, `modules/lighterglue.py` and
`inference.py` -- so the model forward path stays free of validation
overhead while user-facing calls fail fast on malformed inputs.
"""

from __future__ import annotations

from typing import TypeAlias, TypedDict

from beartype import BeartypeConf, beartype
from jaxtyping import Bool, Float, UInt8, jaxtyped
from numpy import ndarray as NDArray
from torch import Tensor

typechecked = jaxtyped(typechecker=beartype(conf=BeartypeConf(is_pep484_tower=True)))
"""Validate jaxtyping shape/dtype annotations at runtime through beartype.

The PEP 484 numeric tower is enabled so that ``int`` accepts as ``float``,
matching what static type checkers report.
"""

ImageInput: TypeAlias = (
    Float[Tensor, "B C H W"]
    | UInt8[Tensor, "B C H W"]
    | Float[NDArray, "B H W C"]
    | UInt8[NDArray, "B H W C"]
    | Float[NDArray, "H W C"]
    | UInt8[NDArray, "H W C"]
    | Float[NDArray, "H W"]
    | UInt8[NDArray, "H W"]
)
"""Image accepted by the sparse/dense extraction API.

Float images are expected in [0, 1] with the layout used by torch models
``(B, C, H, W)`` or by OpenCV/NumPy ``(H, W, C)``. Raw ``uint8`` images are
also accepted: XFeat normalizes each sample internally, so pixel scale does
not change the produced features.
"""

Keypoints: TypeAlias = Float[Tensor, "N 2"]
Scores: TypeAlias = Float[Tensor, "N"]  # noqa: F821 -- "N" is a jaxtyping axis
Descriptors: TypeAlias = Float[Tensor, "N 64"]
KeypointsArray: TypeAlias = Float[NDArray, "N 2"]
InlierMask: TypeAlias = Bool[NDArray, "N"]  # noqa: F821 -- "N" is a jaxtyping axis


class SparseFeatures(TypedDict):
    """Sparse keypoints, scores and descriptors produced by `detectAndCompute`."""

    keypoints: Keypoints
    scores: Scores
    descriptors: Descriptors


class DenseFeatures(TypedDict):
    """Coarse keypoints, descriptors and scales produced by `detectAndComputeDense`."""

    keypoints: Keypoints
    descriptors: Descriptors
    scales: Scores


class SparseFeaturesWithSize(SparseFeatures):
    """Sparse features with the ``(width, height)`` of their source image.

    `match_lighterglue` needs the image size to normalize keypoints, and
    kornia's LightGlue expects it in OpenCV ``(width, height)`` order.
    """

    image_size: tuple[int, int]
