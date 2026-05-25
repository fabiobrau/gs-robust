"""GaussianSplats demo.

Fits a 2D Gaussian Splatting model to an image (or a synthetic test pattern
if no image is supplied) and saves a three-panel figure:

    clean (left)  |  rasterized (center)  |  2-sigma ellipses (right)

Examples
--------
    python demo.py
    python demo.py data/gauss.jpeg --n-gaussians 1500 --n-iters 300
    python demo.py --learn-opacity         # make per-gaussian alpha learnable
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as tf
from gaussian_splats import GaussianSplats
from matplotlib.collections import LineCollection
from PIL import Image


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synthetic_image(size: int = 128) -> torch.Tensor:
    """A simple synthetic RGB image so the demo runs without external files."""
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing="ij",
    )
    r = (x - 0.25) ** 2 + (y - 0.1) ** 2
    g = (x + 0.3) ** 2 + (y + 0.2) ** 2
    b = x**2 + (y - 0.4) ** 2
    img = torch.stack([
        torch.exp(-6 * r),
        torch.exp(-6 * g),
        torch.exp(-6 * b),
    ])
    # add a checker-ish background to give the splats something more textured
    checker = ((x * 4).floor() + (y * 4).floor()).remainder(2)
    img = 0.7 * img + 0.3 * checker.unsqueeze(0)
    return img.clamp(0, 1)


def load_image(path: str | None, size: int | None) -> torch.Tensor:
    if path is not None and Path(path).exists():
        img = tf.to_tensor(Image.open(path).convert("RGB"))
        if size is not None:
            img = tf.resize(img, [size, size])
        return img
    if path is not None:
        print(f"[demo] image '{path}' not found — using synthetic test pattern.")
    return synthetic_image(size if size is not None else 128)


def tensor_to_np(img: torch.Tensor) -> np.ndarray:
    return img.detach().to("cpu").clamp(0, 1).permute(1, 2, 0).numpy()


def plot_ellipses(ax, model: GaussianSplats, img_size: tuple[int, int]) -> None:
    """Draw empty 2-sigma ellipses on ``ax`` in pixel coordinates."""
    H, W = img_size
    contours = model.ellipse_contours(n_sigma=2.0, n_points=64).to("cpu").numpy()
    # contours[..., 0] is the row coord (dim 0), contours[..., 1] the col coord.
    # map normalized [-1, 1] -> pixel coords for imshow:
    #   matplotlib x-axis = column = contours[..., 1]
    #   matplotlib y-axis = row    = contours[..., 0]
    px = (contours[..., 1] + 1.0) * 0.5 * (W - 1)
    py = (contours[..., 0] + 1.0) * 0.5 * (H - 1)
    segs = np.stack([px, py], axis=-1)  # (G, P, 2)
    lc = LineCollection(segs, colors="black", linewidths=0.4, alpha=0.7)
    ax.add_collection(lc)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)  # invert y to match imshow
    ax.set_aspect("equal")
    ax.set_facecolor("white")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "image",
        type=str,
        nargs="?",
        default=None,
        help="path to input image (omit for synthetic test pattern)",
    )
    p.add_argument(
        "--size",
        type=int,
        default=None,
        help="resize input to size x size; default keeps the image's native dims",
    )
    p.add_argument("--n-gaussians", type=int, default=600)
    p.add_argument("--n-iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument(
        "--learn-opacity",
        action="store_true",
        help="make per-gaussian alpha learnable (default: frozen at --fixed-opacity).",
    )
    p.add_argument("--fixed-opacity", type=float, default=1.0)
    p.add_argument(
        "--local-render",
        action="store_true",
        help="evaluate each gaussian only on its 3-sigma AABB (faster on CUDA "
        "and at small G; usually slower on MPS for G > ~1000).",
    )
    p.add_argument(
        "--fast-init",
        action="store_true",
        help="initialize Gaussians with Fast-2DGS pretrained networks "
        "(submodule under Fast-2DGS/).",
    )
    p.add_argument(
        "--lm",
        action="store_true",
        help="fit with Levenberg-Marquardt instead of Adam (uses --lm-lambda).",
    )
    p.add_argument("--lm-lambda", type=float, default=1e-3)
    p.add_argument(
        "--cg-iters",
        type=int,
        default=30,
        help="max CG iterations per LM step (matrix-free normal equations).",
    )
    p.add_argument("--cg-tol", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="demo_out.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"[demo] device = {device}")

    img = load_image(args.image, args.size).to(device)
    H, W = img.shape[1:]
    print(f"[demo] image size = {H}x{W}")

    if args.fast_init:
        from gaussian_splats_fast2dgs import FastInitGaussianSplats

        model = FastInitGaussianSplats(
            image=img,
            n_gaussians=args.n_gaussians,
            n_channels=img.shape[0],
            learn_opacity=args.learn_opacity,
            fixed_opacity=args.fixed_opacity,
            local_render=args.local_render,
        ).to(device)
    elif args.lm:
        from gaussian_splats_lm import GaussianSplatsLM

        model = GaussianSplatsLM(
            n_gaussians=args.n_gaussians,
            n_channels=img.shape[0],
            learn_opacity=args.learn_opacity,
            fixed_opacity=args.fixed_opacity,
            local_render=args.local_render,
        ).to(device)
    else:
        model = GaussianSplats(
            n_gaussians=args.n_gaussians,
            n_channels=img.shape[0],
            learn_opacity=args.learn_opacity,
            fixed_opacity=args.fixed_opacity,
            local_render=args.local_render,
        ).to(device)
    print(
        f"[demo] {args.n_gaussians} gaussians, "
        f"learn_opacity={model.learn_opacity}, "
        f"fast_init={args.fast_init}, lm={args.lm}, "
        f"n_iters={args.n_iters}"
    )

    if args.lm:
        model.fit(
            img,
            n_iters=args.n_iters,
            lm_lambda=args.lm_lambda,
            cg_iters=args.cg_iters,
            cg_tol=args.cg_tol,
        )
    else:
        model.fit(img, n_iters=args.n_iters, lr=args.lr)

    with torch.no_grad():
        rendered = model().clamp(0, 1)
    mse = torch.mean((rendered - img) ** 2).item()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    print(f"[demo] final MSE = {mse:.5f}  PSNR = {psnr:.2f} dB")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(tensor_to_np(img))
    axes[0].set_title("clean")
    axes[1].imshow(tensor_to_np(rendered))
    axes[1].set_title(f"rasterized  (PSNR {psnr:.1f} dB)")
    plot_ellipses(axes[2], model, (H, W))
    axes[2].set_title(rf"$2\sigma$ ellipses ({args.n_gaussians} gaussians)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[demo] saved figure -> {args.out}")


if __name__ == "__main__":
    main()
