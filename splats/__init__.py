"""2D Gaussian Splatting variants for image reconstruction.

Three variants are provided:

* :class:`GaussianSplats` — naive Adam-fitted baseline.
* :class:`GaussianSplatsEM` — closed-form EM / weighted k-means init + optional
  Adam polish.
* :class:`FastInitGaussianSplats` — initialization from Fast-2DGS pretrained
  networks (Wang et al., arXiv:2512.12774) + Adam refinement.
* :class:`GaussianSplatsVarPro` — Variable Projection on a Partition-of-Unity
  basis; colors eliminated analytically each step + optional Adam polish.

All variants share the same rasterizer: the NYU-ICL image-gs CUDA tile-rasterizer
is used automatically when CUDA is available and the gsplat extension is compiled
(``cd image-gs/gsplat && pip install -e .``); otherwise falls back to PyTorch.
"""
from .base import GaussianSplats, cuda_rasterizer_available
from .em import GaussianSplatsEM
from .fast2dgs import FastInitGaussianSplats
from .online import GaussianSplatsOnline
from .additive import GaussianSplatsAdditive
from .varpro import GaussianSplatsVarPro
from .oneshot_demo import OneShotGaussianImage

__all__ = [
    "GaussianSplats",
    "GaussianSplatsEM",
    "GaussianSplatsOnline",
    "FastInitGaussianSplats",
    "GaussianSplatsAdditive",
    "GaussianSplatsVarPro",
    "OneShotGaussianImage",
    "cuda_rasterizer_available",
]
