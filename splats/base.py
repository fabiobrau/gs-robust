"""Base 2D Gaussian Splatting model with optional CUDA rasterization.

CUDA path: uses the NYU-ICL image-gs CUDA kernels (submodule under image-gs/)
when CUDA is available AND the gsplat extension has been compiled.
Build once via::

    cd image-gs/gsplat && pip install -e .

Otherwise falls back to pure PyTorch automatically.

Coordinate conventions used throughout this package:
    centers  (G, 2)  in [-1, 1]² with layout (row, col).
    covs     (G, 3)  = (theta, b_std, a_std): theta is the rotation angle of
             the principal axis in (row, col) space; a_std is the std along
             (cos θ, sin θ) and b_std along the orthogonal (-sin θ, cos θ).
    _colors  (G, C)  raw logits; sigmoid gives colors in (0, 1).
    _alphas  (G,)    raw logits when learn_opacity=True; fixed buffer otherwise.
"""
from __future__ import annotations

import math
import sys
from collections.abc import Generator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Size, Tensor, nn
from torch.optim import Adam

# ---------------------------------------------------------------------------
# Optional CUDA rasterizer (NYU-ICL image-gs)
# ---------------------------------------------------------------------------
_IMAGE_GS_GSPLAT = Path(__file__).resolve().parent.parent / "image-gs" / "gsplat"

_cuda_project = None
_cuda_rasterize = None

try:
    _gs_path = str(_IMAGE_GS_GSPLAT)
    if _gs_path not in sys.path:
        sys.path.insert(0, _gs_path)
    from gsplat import project_gaussians_2d_scale_rot as _cuda_project   # type: ignore[assignment]
    from gsplat import rasterize_gaussians_sum as _cuda_rasterize         # type: ignore[assignment]
except (ImportError, OSError):
    _cuda_project = None   # type: ignore[assignment]
    _cuda_rasterize = None  # type: ignore[assignment]


def cuda_rasterizer_available() -> bool:
    """True if the image-gs CUDA extension was successfully imported."""
    return _cuda_project is not None


_LOCAL_N_SIGMA = 3.0  # truncation radius for the local PyTorch rasterizer

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GaussianSplats(nn.Module):
    """2D Gaussian Splatting for image reconstruction.

    Rendering dispatches to the NYU-ICL image-gs CUDA tile-rasterizer when CUDA
    is available and the gsplat extension is compiled; falls back to PyTorch.
    """

    def __init__(
        self,
        n_gaussians: int,
        n_channels: int = 3,
        learn_opacity: bool = False,
        fixed_opacity: float = 1.0,
        local_render: bool = False,
    ) -> None:
        super().__init__()
        self.n_gaussians = n_gaussians
        self.n_channels = n_channels
        self.learn_opacity = learn_opacity
        self.fixed_opacity = fixed_opacity
        self.local_render = local_render  # hint: CUDA path tiles natively; PyTorch always full
        self.img_size: tuple[int, int] | None = None

        self.centers = nn.Parameter(2.0 * torch.rand(n_gaussians, 2) - 1.0)
        self.covs = nn.Parameter(torch.rand(n_gaussians, 3) * 0.1)  # (theta, b_std, a_std)
        # _colors: raw logits; sigmoid → (0,1) colors
        self._colors = nn.Parameter(torch.zeros(n_gaussians, n_channels))
        if learn_opacity:
            self._alphas = nn.Parameter(torch.zeros(n_gaussians))  # logits
        else:
            self.register_buffer("_alphas", torch.full((n_gaussians,), fixed_opacity))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def colors(self) -> Tensor:
        return self._colors.sigmoid()

    @property
    def alphas(self) -> Tensor:
        if self.learn_opacity:
            return self._alphas.sigmoid()  # type: ignore[union-attr]
        return self._alphas  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Rasterizers
    # ------------------------------------------------------------------

    def _rasterize_cuda(self, img_size: tuple[int, int]) -> Tensor:
        """Tile-based CUDA rasterizer via NYU-ICL image-gs gsplat."""
        H, W = img_size
        BLOCK = 16
        tile_bounds = ((W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK, 1)

        # image-gs: xy[:,0]=x=col∈[0,1], xy[:,1]=y=row∈[0,1]
        # ours:     centers[:,0]=row∈[-1,1], centers[:,1]=col∈[-1,1]
        xy = torch.stack(
            [(self.centers[:, 1] + 1.0) * 0.5,   # col → x
             (self.centers[:, 0] + 1.0) * 0.5],  # row → y
            dim=1,
        )

        # scale: image-gs [0,1] space; ours [-1,1] (range 2) → divide by 2
        # covs[:,1]=b_std (col/x direction), covs[:,2]=a_std (row/y direction)
        scale = torch.stack(
            [self.covs[:, 1].abs() * 0.5,   # x / col direction
             self.covs[:, 2].abs() * 0.5],  # y / row direction
            dim=1,
        )

        # Our θ is measured from the row-axis in (row,col) space.
        # image-gs measures from x=col axis → angle = π/2 − θ.
        rot = (0.5 * torch.pi - self.covs[:, 0]).unsqueeze(1)  # (G, 1)

        xys, radii, conics, num_tiles_hit = _cuda_project(  # type: ignore[misc]
            xy, scale, rot, H, W, tile_bounds
        )

        # Blend: sum(α·g·c) / sum(α·g) via C+1 feature channels
        alphas = self.alphas.unsqueeze(1)                        # (G, 1)
        feat = torch.cat([alphas * self.colors, alphas], dim=1)  # (G, C+1)

        out = _cuda_rasterize(  # type: ignore[misc]
            xys, radii, conics, num_tiles_hit, feat, H, W, BLOCK, BLOCK,
            False,  # topk_norm=False → raw sums; we normalise below
        )  # (H, W, C+1)

        num = out[..., : self.n_channels].permute(2, 0, 1)  # (C, H, W)
        den = out[..., self.n_channels :].permute(2, 0, 1)  # (1, H, W)
        return num / (den + 1e-7)

    def _rasterize_pytorch(self, img_size: tuple[int, int]) -> Tensor:
        """Pure-PyTorch weighted-average rasterizer (O(G·H·W) memory)."""
        H, W = img_size
        device, dtype = self.centers.device, self.centers.dtype
        eps = 1e-7

        x = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")  # (H, W)

        d_row = xx.unsqueeze(0) - self.centers[:, 0, None, None]  # (G, H, W)
        d_col = yy.unsqueeze(0) - self.centers[:, 1, None, None]

        theta = self.covs[:, 0, None, None]
        ct, st = theta.cos(), theta.sin()
        rot_x = d_row * ct - d_col * st
        rot_y = d_row * st + d_col * ct

        inv_a = 1.0 / (self.covs[:, 2].abs() + eps)[:, None, None]
        inv_b = 1.0 / (self.covs[:, 1].abs() + eps)[:, None, None]
        gs = torch.exp(-0.5 * ((rot_x * inv_a) ** 2 + (rot_y * inv_b) ** 2))  # (G,H,W)

        w = self.alphas[:, None, None, None] * gs.unsqueeze(1)  # (G,1,H,W)
        colors = self.colors[:, :, None, None]                   # (G,C,1,1)
        return (w * colors).sum(0) / (w.sum(0) + eps)           # (C,H,W)

    def _rasterize_pytorch_local(self, img_size: tuple[int, int]) -> Tensor:
        """PyTorch rasterizer that evaluates each Gaussian only within its 3σ bbox.

        Trades the O(G·H·W) memory of the full rasterizer for a Python loop over
        Gaussians, each touching only a small (pH×pW) patch.  Useful when Gaussians
        are small relative to the image.
        """
        H, W = img_size
        device, dtype = self.centers.device, self.centers.dtype
        eps = 1e-7

        x = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)

        num = torch.zeros(self.n_channels, H, W, device=device, dtype=dtype)
        den = torch.zeros(1, H, W, device=device, dtype=dtype)

        # Bounding-box indices are non-differentiable; detach to avoid graph overhead.
        centers_d = self.centers.detach()
        covs_d = self.covs.detach()

        for k in range(self.n_gaussians):
            cx = centers_d[k, 0].item()
            cy = centers_d[k, 1].item()
            th = covs_d[k, 0].item()
            a = abs(covs_d[k, 2].item())
            b = abs(covs_d[k, 1].item())
            ct_k = math.cos(th)
            st_k = math.sin(th)

            # Half-extents of the tight axis-aligned bbox of the n_sigma ellipse.
            row_ext = _LOCAL_N_SIGMA * math.sqrt((a * ct_k) ** 2 + (b * st_k) ** 2)
            col_ext = _LOCAL_N_SIGMA * math.sqrt((a * st_k) ** 2 + (b * ct_k) ** 2)

            r0 = max(0, math.floor((cx - row_ext + 1.0) * 0.5 * (H - 1)))
            r1 = min(H, math.ceil((cx + row_ext + 1.0) * 0.5 * (H - 1)) + 1)
            c0 = max(0, math.floor((cy - col_ext + 1.0) * 0.5 * (W - 1)))
            c1 = min(W, math.ceil((cy + col_ext + 1.0) * 0.5 * (W - 1)) + 1)
            if r0 >= r1 or c0 >= c1:
                continue

            xx_g, yy_g = torch.meshgrid(x[r0:r1], y[c0:c1], indexing="ij")
            d_row = xx_g - self.centers[k, 0]
            d_col = yy_g - self.centers[k, 1]
            ct_t = self.covs[k, 0].cos()
            st_t = self.covs[k, 0].sin()
            rot_x = d_row * ct_t - d_col * st_t
            rot_y = d_row * st_t + d_col * ct_t
            inv_a = 1.0 / (self.covs[k, 2].abs() + eps)
            inv_b = 1.0 / (self.covs[k, 1].abs() + eps)
            gs = torch.exp(-0.5 * ((rot_x * inv_a) ** 2 + (rot_y * inv_b) ** 2))

            w = self.alphas[k] * gs
            num[:, r0:r1, c0:c1] = num[:, r0:r1, c0:c1] + self.colors[k, :, None, None] * w
            den[0, r0:r1, c0:c1] = den[0, r0:r1, c0:c1] + w

        return num / (den + eps)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, img_size: Size | None = None) -> Tensor:
        """Render image of shape (C, H, W) in [0, 1]."""
        img_size = self.img_size if img_size is None else tuple(img_size)
        assert img_size is not None, "provide img_size or call fit() first"
        if cuda_rasterizer_available() and self.centers.is_cuda:
            return self._rasterize_cuda(img_size)
        if self.local_render:
            return self._rasterize_pytorch_local(img_size)
        return self._rasterize_pytorch(img_size)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def ellipse_contours(self, n_sigma: float = 2.0, n_points: int = 64) -> Tensor:
        """Return (G, n_points, 2) ellipse contour points in [-1,1] (row, col)."""
        t = torch.linspace(0.0, 2.0 * torch.pi, n_points, device=self.centers.device)
        ct_t, st_t = t.cos(), t.sin()
        theta = self.covs[:, 0]          # (G,)
        ct, st = theta.cos(), theta.sin()
        a = self.covs[:, 2].abs() * n_sigma  # row-axis std × n_sigma
        b = self.covs[:, 1].abs() * n_sigma  # col-axis std × n_sigma
        # Rotate unit ellipse back from (rot_x, rot_y) to (d_row, d_col)
        row = ct[:, None] * a[:, None] * ct_t + st[:, None] * b[:, None] * st_t
        col = -st[:, None] * a[:, None] * ct_t + ct[:, None] * b[:, None] * st_t
        pts = torch.stack([row, col], dim=-1) + self.centers[:, None, :]  # (G,P,2)
        return pts

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit_iterable(
        self,
        img: Tensor,
        n_iters: int = 300,
        lr: float = 0.05,
        verbose: bool = True,
    ) -> Generator[tuple[Tensor, Tensor], None, None]:
        self.img_size = tuple(img.shape[1:])
        assert self._colors.shape[-1] == img.shape[0], "channel count mismatch"
        optim = Adam(self.parameters(), lr=lr)
        for i in range(n_iters):
            rendered = self.forward()
            err = F.mse_loss(rendered, img)
            if verbose:
                psnr = 10.0 * torch.log10(1.0 / err.clamp(min=1e-12))
                sys.stdout.write(f"\r[Adam] {i + 1}/{n_iters}  PSNR {psnr.item():.2f} dB")
                sys.stdout.flush()
            yield rendered.detach(), err.detach().cpu()
            err.backward()
            optim.step()
            optim.zero_grad()
        if verbose:
            sys.stdout.write("\n")

    def fit(self, img: Tensor, n_iters: int = 300, lr: float = 0.05) -> None:
        """Fit parameters to ``img`` (C, H, W) with Adam."""
        for _ in self._fit_iterable(img, n_iters=n_iters, lr=lr):
            pass
