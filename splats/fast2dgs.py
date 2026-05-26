"""GaussianSplats initialized from the Fast-2DGS pretrained networks.

Same API as :class:`~splats.base.GaussianSplats`, but the constructor also
takes the target image.  It runs the Fast-2DGS networks (``HeatmapUNet`` +
``GaussianUNet_Plus``) once to predict initial Gaussian positions, colors,
scales, and rotations, then fits with Adam (or any parent-class method).

Reference: Wang et al., "Fast 2DGS: Efficient Image Representation with Deep
Gaussian Prior", arXiv:2512.12774. Weights live in ``Fast-2DGS/weights/``.

Coordinate / parameter conventions:
    Fast-2DGS: xy in [0, 1]² with (x=col_frac, y=row_frac); scale = (sx, sy)
      in pixel-like units; rot in [0, 2π].
    Ours: centers in [-1, 1]² with (row, col); covs = (theta, b_std, a_std)
      where a is the std-dev along the rotated row-axis and b along the
      rotated col-axis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import GaussianSplats

_FAST2DGS_DIR = Path(__file__).resolve().parent.parent / "Fast-2DGS"


def _ensure_path() -> None:
    p = str(_FAST2DGS_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_networks(device: torch.device, plus: bool = True):
    _ensure_path()
    from models.GS_UNet import GaussianUNet, GaussianUNet_Plus, HeatmapUNet  # noqa

    heat = HeatmapUNet(base_ch=32)
    feat = GaussianUNet_Plus(base_ch=32) if plus else GaussianUNet(base_ch=32)
    heat_w = _FAST2DGS_DIR / "weights" / "smp_heat_500_0.0005.pth"
    feat_w = _FAST2DGS_DIR / "weights" / (
        "smp_feat_best_psnr_26_plus.pth" if plus else "smp_feat_best_psnr_26.pth"
    )
    if not heat_w.exists() or not feat_w.exists():
        raise FileNotFoundError(
            f"Fast-2DGS weights missing under {_FAST2DGS_DIR / 'weights'}. "
            "Did you `git submodule update --init Fast-2DGS`?"
        )
    heat.load_state_dict(torch.load(heat_w, map_location="cpu"))
    feat.load_state_dict(torch.load(feat_w, map_location="cpu"))
    return heat.to(device).eval(), feat.to(device).eval()


def _ceil_to_multiple(n: int, m: int = 16) -> int:
    return max(m, ((n + m - 1) // m) * m)


@torch.no_grad()
def predict_init(
    image: Tensor,
    n_gaussians: int,
    device: torch.device,
    plus: bool = True,
    sampling: str = "multinomial",
) -> dict[str, Tensor]:
    """Run Fast-2DGS networks on ``image`` and return init params in our format.

    Args:
        image:       ``(C, H, W)`` float tensor in ``[0, 1]``.
        n_gaussians: number of Gaussians to sample (K).
        device:      device to run the networks on.
        plus:        use ``GaussianUNet_Plus`` (also predicts a sub-pixel offset map).
        sampling:    ``"multinomial"`` (default) or ``"topk"``.

    Returns:
        dict with keys ``centers``, ``covs``, ``colors``, ``alphas`` — all in
        our coordinate conventions.  ``colors`` are raw [0,1] values (the
        caller converts to logits before storing in ``_colors``).
    """
    C, H, W = image.shape
    H_pad = _ceil_to_multiple(H, 16)
    W_pad = _ceil_to_multiple(W, 16)
    x = image.unsqueeze(0).to(device)
    if (H_pad, W_pad) != (H, W):
        x = F.interpolate(x, size=(H_pad, W_pad), mode="bilinear", align_corners=False)

    heat, feat = _load_networks(device, plus=plus)
    batch_K = torch.full((1, 1), float(n_gaussians), device=device)

    heatmap = heat(x, batch_K)
    out = feat(x, batch_K)
    if plus:
        offset_map, scale_map, color_map, rot_map = out
    else:
        scale_map, color_map, rot_map = out
        offset_map = None

    prob = heatmap.reshape(1, -1).clamp(min=1e-8).float()
    prob = prob / prob.sum(dim=1, keepdim=True)
    if sampling == "topk":
        _, idx = prob.topk(n_gaussians, dim=1)
    else:
        idx = torch.multinomial(prob.cpu(), num_samples=n_gaussians, replacement=True).to(device)
    idx = idx[0]
    ys = torch.div(idx, W_pad, rounding_mode="floor").long()
    xs = (idx % W_pad).long()

    x_frac = xs.float() / (W_pad - 1)
    y_frac = ys.float() / (H_pad - 1)
    if offset_map is not None:
        off = offset_map[0, :, ys, xs]  # (2, K)
        x_frac = (x_frac + off[0]).clamp(0, 1)
        y_frac = (y_frac + off[1]).clamp(0, 1)

    # (row, col) in [-1, 1]
    centers = torch.stack([2.0 * y_frac - 1.0, 2.0 * x_frac - 1.0], dim=-1)

    colors = color_map[0, :, ys, xs].t().clamp(0.0, 1.0)  # (K, C) in [0,1]

    # Scales: Fast-2DGS outputs pixel-ish widths; convert to normalized stds
    scales = scale_map[0, :, ys, xs].t()  # (K, 2): (sx, sy)
    b = (scales[:, 0] * 2.0 / W_pad).clamp(min=1e-3)  # col (x) direction → b
    a = (scales[:, 1] * 2.0 / H_pad).clamp(min=1e-3)  # row (y) direction → a

    theta = rot_map[0, 0, ys, xs]  # (K,) in [0, 2π]

    covs = torch.stack([theta, b, a], dim=-1)  # (K, 3)
    alphas = torch.ones(n_gaussians, device=device)
    return {"centers": centers, "covs": covs, "colors": colors, "alphas": alphas}


class FastInitGaussianSplats(GaussianSplats):
    """``GaussianSplats`` initialized from the Fast-2DGS networks."""

    def __init__(
        self,
        image: Tensor,
        n_gaussians: int,
        n_channels: int = 3,
        learn_opacity: bool = True,
        fixed_opacity: float = 1.0,
        local_render: bool = False,
        plus: bool = True,
        sampling: str = "multinomial",
    ) -> None:
        if n_channels != 3:
            raise ValueError("Fast-2DGS networks are RGB-only (n_channels=3).")
        super().__init__(
            n_gaussians=n_gaussians,
            n_channels=n_channels,
            learn_opacity=learn_opacity,
            fixed_opacity=fixed_opacity,
            local_render=local_render,
        )
        device = image.device
        self.to(device)
        init = predict_init(
            image=image.clamp(0, 1),
            n_gaussians=n_gaussians,
            device=device,
            plus=plus,
            sampling=sampling,
        )
        with torch.no_grad():
            self.centers.copy_(init["centers"])
            self.covs.copy_(init["covs"])
            # Store logits so that sigmoid(_colors) = network-predicted color
            self._colors.copy_(init["colors"].clamp(1e-6, 1.0 - 1e-6).logit())
            if self.learn_opacity:
                # init["alphas"] are target opacities in [0,1]; store as logits
                self._alphas.copy_(  # type: ignore[union-attr]
                    init["alphas"].clamp(1e-6, 1.0 - 1e-6).logit()
                )
        self.img_size = tuple(image.shape[1:])
