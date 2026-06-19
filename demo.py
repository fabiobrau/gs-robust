"""GaussianSplats demo.

Fits a 2D Gaussian Splatting model to an image (or a synthetic test pattern
if no image is supplied) and saves a three-panel figure:

    clean (left)  |  rasterized (center)  |  2-sigma ellipses (right)

Three fitting modes are available:
    default     — naive Adam optimisation (``GaussianSplats``).
    --fast-init — initialize with Fast-2DGS pretrained networks, then Adam.
    --em        — closed-form EM / weighted k-means (+ optional Adam polish).
    --varpro    — Variable Projection on PoU basis; colors solved analytically.

Examples
--------
    python3 demo.py
    python3 demo.py data/gauss.jpeg --n-gaussians 1500 --n-iters 300
    python3 demo.py --em --em-polish-iters 50
    python3 demo.py --fast-init --n-gaussians 600
    python3 demo.py --varpro --n-iters 20 --em-polish-iters 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as tf
from matplotlib.collections import LineCollection
from PIL import Image
from splats import (
    FastInitGaussianSplats,
    GaussianSplats,
    GaussianSplatsAdditive,
    GaussianSplatsEM,
    GaussianSplatsOnline,
    GaussianSplatsVarPro,
    OneShotGaussianImage,
    cuda_rasterizer_available,
)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synthetic_image(size: int = 128) -> torch.Tensor:
    """Simple synthetic RGB image so the demo runs without external files."""
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
    contours = (
        model.ellipse_contours(n_sigma=2.0, n_points=64).detach().to("cpu").numpy()
    )
    px = (contours[..., 1] + 1.0) * 0.5 * (W - 1)
    py = (contours[..., 0] + 1.0) * 0.5 * (H - 1)
    segs = np.stack([px, py], axis=-1)
    lc = LineCollection(segs, colors="black", linewidths=0.4, alpha=0.7)
    ax.add_collection(lc)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
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
        help="resize input to size×size; default keeps the image's native dims",
    )
    p.add_argument("--n-gaussians", type=int, default=600)
    p.add_argument("--n-iters", type=int, default=10)
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
        help="hint: tile-based CUDA rasterizer handles this natively.",
    )
    p.add_argument(
        "--fast-init",
        action="store_true",
        help="initialize Gaussians with Fast-2DGS pretrained networks (Fast-2DGS submodule).",
    )
    p.add_argument(
        "--additive",
        action="store_true",
        help="fit unnormalized additive model Î(x)=Σ c_k G_k(x), colors unconstrained.",
    )
    p.add_argument(
        "--additive-lam",
        type=float,
        default=1e-4,
        help="Tikhonov λ added to Gram diagonal for the exact color solve.",
    )
    p.add_argument(
        "--em",
        action="store_true",
        help="fit with closed-form EM / weighted k-means (then optional Adam polish).",
    )
    p.add_argument(
        "--em-lambda",
        type=float,
        default=1e-4,
        help="covariance damping λ added to Σ_k each M-step.",
    )
    p.add_argument("--em-sigma-min", type=float, default=0.005)
    p.add_argument("--em-sigma-max", type=float, default=0.5)
    p.add_argument(
        "--em-reseed-every",
        type=int,
        default=5,
        help="reseed dead splats every N EM iters (0 disables).",
    )
    p.add_argument(
        "--em-polish-iters", type=int, default=200, help="post-EM Adam polish steps."
    )
    p.add_argument("--em-polish-lr", type=float, default=0.01)
    p.add_argument(
        "--online",
        action="store_true",
        help="fit with EM init + patch-based Adam (GaussianSplatsOnline).",
    )
    p.add_argument(
        "--online-patch-size", type=int, default=32,
        help="patch side length (pixels) for the online Adam loop.",
    )
    p.add_argument(
        "--online-em-init-iters", type=int, default=10,
        help="closed-form EM warm-up steps before patch Adam.",
    )
    p.add_argument(
        "--online-log-every", type=int, default=50,
        help="full-image PSNR refresh interval during online Adam.",
    )
    p.add_argument(
        "--online-polish-iters", type=int, default=0,
        help="full-image Adam polish steps after the patch loop (0 to skip).",
    )
    p.add_argument("--online-polish-lr", type=float, default=0.01)
    p.add_argument(
        "--varpro",
        action="store_true",
        help="fit with Variable Projection on PoU basis (colors solved analytically each step).",
    )
    p.add_argument(
        "--varpro-ridge",
        type=float,
        default=1e-6,
        help="ridge regularization added to the Gram matrix in the VarPro color solve.",
    )
    p.add_argument(
        "--oneshot",
        action="store_true",
        help="fit with the deterministic NumPy one-shot model (splats/oneshot.py, CPU).",
    )
    p.add_argument(
        "--oneshot-adam",
        type=int,
        default=0,
        help="post one-shot, run N torch Adam steps on GPU/MPS (exact one-shot model).",
    )
    p.add_argument("--oneshot-adam-lr", type=float, default=0.2)
    p.add_argument("--out", type=str, default="demo_out.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    cuda_ok = cuda_rasterizer_available()
    print(f"[demo] device = {device}  |  CUDA rasterizer = {cuda_ok}")

    img = load_image(args.image, args.size).to(device)
    H, W = img.shape[1:]
    print(f"[demo] image size = {H}×{W}")

    if args.fast_init:
        model = FastInitGaussianSplats(
            image=img,
            n_gaussians=args.n_gaussians,
            n_channels=img.shape[0],
            learn_opacity=args.learn_opacity,
            fixed_opacity=args.fixed_opacity,
            local_render=args.local_render,
        ).to(device)
    elif args.additive:
        model = GaussianSplatsAdditive(
            n_gaussians=args.n_gaussians,
            n_channels=img.shape[0],
            local_render=args.local_render,
        ).to(device)
    elif args.online:
        model = GaussianSplatsOnline(
            n_gaussians=args.n_gaussians,
            n_channels=img.shape[0],
            learn_opacity=args.learn_opacity,
            fixed_opacity=args.fixed_opacity,
            local_render=args.local_render,
        ).to(device)
    elif args.em:
        model = GaussianSplatsEM(
            n_gaussians=args.n_gaussians,
            n_channels=img.shape[0],
            learn_opacity=args.learn_opacity,
            fixed_opacity=args.fixed_opacity,
            local_render=args.local_render,
        ).to(device)
    elif args.varpro:
        model = GaussianSplatsVarPro(
            n_gaussians=args.n_gaussians,
            n_channels=img.shape[0],
            learn_opacity=args.learn_opacity,
            fixed_opacity=args.fixed_opacity,
            local_render=args.local_render,
        ).to(device)
    elif args.oneshot:
        model = OneShotGaussianImage(
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
        f"[demo] {args.n_gaussians} gaussians | "
        f"learn_opacity={model.learn_opacity} | "
        f"fast_init={args.fast_init} | em={args.em} | "
        f"additive={args.additive} | varpro={args.varpro} | "
        f"oneshot={args.oneshot} | n_iters={args.n_iters}"
    )

    if args.additive:
        assert isinstance(model, GaussianSplatsAdditive)
        model.fit(img, n_iters=args.n_iters, lr=args.lr, lam=args.additive_lam)
    elif args.online:
        assert isinstance(model, GaussianSplatsOnline)
        model.fit(
            img,
            n_iters=args.n_iters,
            lr=args.lr,
            patch_size=args.online_patch_size,
            em_init_iters=args.online_em_init_iters,
            em_sigma_min=args.em_sigma_min,
            em_sigma_max=args.em_sigma_max,
            em_lam=args.em_lambda,
            em_reseed_every=args.em_reseed_every,
            log_every=args.online_log_every,
            polish_iters=args.online_polish_iters,
            polish_lr=args.online_polish_lr,
        )
    elif args.em:
        model.fit(  # type: ignore[union-attr]
            img,
            n_iters=args.n_iters,
            sigma_min=args.em_sigma_min,
            sigma_max=args.em_sigma_max,
            lam=args.em_lambda,
            reseed_every=args.em_reseed_every,
            polish_iters=args.em_polish_iters,
            polish_lr=args.em_polish_lr,
        )
    elif args.varpro:
        assert isinstance(model, GaussianSplatsVarPro)
        model.fit(
            img,
            n_iters=args.n_iters,
            sigma_min=args.em_sigma_min,
            sigma_max=args.em_sigma_max,
            lam=args.em_lambda,
            ridge=args.varpro_ridge,
            reseed_every=args.em_reseed_every,
            polish_iters=args.em_polish_iters,
            polish_lr=args.em_polish_lr,
        )
    elif args.oneshot:
        assert isinstance(model, OneShotGaussianImage)
        model.fit(img, n_iters=args.n_iters)
        model.refine_adam(args.oneshot_adam, lr=args.oneshot_adam_lr)
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
    print(f"[demo] saved figure → {args.out}")


if __name__ == "__main__":
    main()
