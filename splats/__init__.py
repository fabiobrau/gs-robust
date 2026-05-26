"""2D Gaussian Splatting variants for image reconstruction.

Three variants are provided:

* :class:`GaussianSplats` — naive Adam-fitted baseline.
* :class:`GaussianSplatsEM` — closed-form EM / weighted k-means init + optional
  Adam polish.
* :class:`FastInitGaussianSplats` — initialization from Fast-2DGS pretrained
  networks (Wang et al., arXiv:2512.12774) + Adam refinement.

All variants share the same rasterizer: the NYU-ICL image-gs CUDA tile-rasterizer
is used automatically when CUDA is available and the gsplat extension is compiled
(``cd image-gs/gsplat && pip install -e .``); otherwise falls back to PyTorch.
"""
from .base import GaussianSplats, cuda_rasterizer_available
from .em import GaussianSplatsEM
from .fast2dgs import FastInitGaussianSplats

__all__ = [
    "GaussianSplats",
    "GaussianSplatsEM",
    "FastInitGaussianSplats",
    "cuda_rasterizer_available",
]
