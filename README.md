# gs-robust

2D Gaussian Splatting for image reconstruction, with three fitting strategies and an optional CUDA tile-rasterizer.

## Overview

A Gaussian mixture is fit to an image so that the weighted-average of colored, oriented ellipses reproduces the target pixel values. Three fitters are provided, all sharing the same rasterizer and parameter layout:

| Class | Strategy |
|---|---|
| `GaussianSplats` | Adam gradient descent (baseline) |
| `GaussianSplatsEM` | Closed-form EM / weighted k-means, optional Adam polish |
| `FastInitGaussianSplats` | Fast-2DGS network initialization, then Adam |

## Repository layout

```
gs-robust/
├── splats/               # main Python package
│   ├── __init__.py       # re-exports the three public classes
│   ├── base.py           # GaussianSplats — model, rasterizer dispatch, Adam fit
│   ├── em.py             # GaussianSplatsEM — closed-form EM fitter
│   └── fast2dgs.py       # FastInitGaussianSplats — Fast-2DGS network init
├── demo.py               # CLI demo: fit a model and save a three-panel figure
├── craters_gs.py         # original Colab notebook (scratch / reference)
├── data/
│   └── gauss.jpeg        # sample image
├── Fast-2DGS/            # git submodule — pretrained UNet networks
└── image-gs/             # git submodule — NYU-ICL CUDA tile-rasterizer
    └── gsplat/           # build this for the CUDA path
```

## Submodules

**`Fast-2DGS`** ([Aztech-Lab/Fast-2DGS](https://github.com/Aztech-Lab/Fast-2DGS))
— contains `HeatmapUNet` and `GaussianUNet_Plus`, pretrained networks that predict
Gaussian positions, scales, rotations, and colors from a single image.
Used by `FastInitGaussianSplats` for a warm start before Adam refinement.
Weights must be present under `Fast-2DGS/weights/`.

**`image-gs`** ([NYU-ICL/image-gs](https://github.com/NYU-ICL/image-gs))
— CUDA tile-rasterizer. When compiled and CUDA is available, all three
fitters use it automatically; otherwise they fall back to pure PyTorch.

## Setup

```bash
# clone with submodules
git clone --recurse-submodules <repo-url>
cd gs-robust

# install Python dependencies
pip install torch torchvision matplotlib pillow

# (optional) compile the CUDA rasterizer
cd image-gs/gsplat && pip install -e . && cd ../..
```

## Running the demo

```bash
# synthetic test pattern, 600 Gaussians, Adam
python3 demo.py

# fit a real image
python3 demo.py data/gauss.jpeg --n-gaussians 1500 --n-iters 300

# closed-form EM with Adam polish
python3 demo.py --em --em-polish-iters 50

# Fast-2DGS network init + Adam (requires Fast-2DGS submodule + weights)
python3 demo.py --fast-init --n-gaussians 600
```

The demo saves a three-panel PNG (`demo_out.png` by default):
`clean | rasterized (PSNR) | 2σ ellipses`.

## Parameter conventions

All three classes share these internal layouts:

| Parameter | Shape | Meaning |
|---|---|---|
| `centers` | `(G, 2)` | `(row, col)` in `[-1, 1]²` |
| `covs` | `(G, 3)` | `(theta, b_std, a_std)`: rotation angle + semi-axes |
| `_colors` | `(G, C)` | logits; `sigmoid` gives colors in `[0, 1]` |
| `_alphas` | `(G,)` | logits (learnable) or fixed buffer |

`a_std` is the std-dev along `(cos θ, sin θ)` in `(row, col)` space;
`b_std` is along the orthogonal direction.
