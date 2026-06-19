"""Pixel-anchored *sparse* 2D Gaussian splatting (normalised blending).

Idea: instead of a flat list of splats each carrying a *free* 2-D mean, anchor
every candidate splat to a fixed **pixel** of an image-sized grid.  A candidate
is then identified by its pixel index plus 6 free variables — 3 colors + 3
covariance parameters (theta, b_std, a_std).  This:

* removes the **permutation ambiguity** — the same image no longer has many
  equivalent splat orderings; the canonical order is the pixel raster order;
* **caps** the number of splats at the number of (candidate) pixels;
* makes the model a genuinely **sparse** object: only a subset of the candidate
  pixels stay *active*.

Forward model — the same **normalised** weighted average as
:class:`splats.base.GaussianSplats` (the original design), with **no opacity**:

    g_k(x) = exp(-½ (x-μ_k)ᵀ Σ_k⁻¹ (x-μ_k))               (peak = 1)
    Î(x)   = Σ_{k active} g_k(x) c_k  /  Σ_{k active} g_k(x)

Each output pixel is a convex combination of the colors of the Gaussians that
reach it, so the render is naturally bounded in [0, 1] and there is no
overlap over-counting.

How sparsity is obtained
------------------------
Because the blend is normalised and has no opacity, **no loss penalty can switch
a splat off**: a zero color is not "off" (it contributes black to the average),
and any weight that appears in both numerator and denominator is scale-invariant
(an L1 on it just shrinks everything uniformly).  Sparsity is therefore realised
by **pruning** redundant / low-contribution splats — see
``sparse_optim.prune_and_refit``.  Pruning flips a per-splat **active** flag;
deactivated splats drop out of *both* sums, so they truly vanish from the model.

Implementation note: the inherited ``_alphas`` buffer is repurposed as that
binary active mask (1.0 = on, 0.0 = pruned) — it is **not** a learnable opacity
(``learn_opacity=False`` throughout).  The base rasterisers already multiply by
it, so masked rendering works on both the CUDA and PyTorch paths for free.

Coordinate conventions are inherited from :mod:`splats.base`:
    centers (N, 2) in [-1, 1]² with layout (row, col);
    covs    (N, 3) = (theta, b_std, a_std).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import GaussianSplats


def _grid_pixels(H: int, W: int, stride: int) -> Tensor:
    """Integer (row, col) coordinates of a strided pixel grid → (N, 2) long."""
    rows = torch.arange(0, H, stride)
    cols = torch.arange(0, W, stride)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=1)  # (N, 2)


class SparseSplatImage(GaussianSplats):
    """Sparse, pixel-anchored 2D Gaussian splatting with normalised blending.

    Candidate centers are laid out on a strided pixel grid and stored as a fixed
    buffer (``stride=1`` ⇒ every pixel is a candidate, the theoretical maximum).
    Only the 6 per-splat variables (colors + covariances) are learnable; the
    means never move.  The active set is sparsified by pruning, tracked in the
    ``_alphas`` buffer used as a 0/1 mask.

    Inherits the normalised weighted-average rasterisers and the sigmoid
    ``colors`` property (colors ∈ (0,1)) from :class:`GaussianSplats`.
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        n_channels: int = 3,
        stride: int = 1,
        sigma_init: float | None = None,
        local_render: bool = False,
        opacity: bool = False,
        bg_weight: float = 0.1,
    ) -> None:
        H, W = img_size
        pix = _grid_pixels(H, W, stride)  # (N, 2) long
        n = pix.shape[0]
        super().__init__(
            n_gaussians=n,
            n_channels=n_channels,
            learn_opacity=False,   # base opacity unused — see opacity handling below
            fixed_opacity=1.0,     # all candidates start active
            local_render=local_render,
        )
        self.img_size = (H, W)
        self.stride = stride

        # --- Sparsity strategy ---------------------------------------------
        # opacity=False (default): _alphas stays a fixed 0/1 active mask;
        #   sparsity comes from pruning (sparse_optim.prune_and_refit).
        # opacity=True: _alphas becomes a learnable non-negative opacity, an
        #   L1/lasso + proximal soft-threshold drives it to exactly 0, and a
        #   background term (bg_weight·c_bg) breaks the scale-invariance of the
        #   normalised blend so the lasso selects *which* splats to drop
        #   instead of shrinking them all uniformly.
        self.opacity = opacity
        self.bg_weight = float(bg_weight) if opacity else 0.0
        if opacity:
            del self._buffers["_alphas"]                  # base set it as a buffer
            self._alphas = nn.Parameter(torch.ones(n))    # raw opacity ≥ 0
            self.register_buffer("c_bg", torch.full((n_channels,), 0.5))

        # --- Replace the learnable centers with FIXED pixel-anchored buffers ---
        # base registers ``centers`` as an nn.Parameter; here the means are not
        # optimised, so drop the parameter and register a buffer instead (so it
        # still follows ``.to(device)`` but never accumulates gradient).
        del self._parameters["centers"]
        rows = pix[:, 0].to(torch.float32)
        cols = pix[:, 1].to(torch.float32)
        centers = torch.stack(
            [
                rows * (2.0 / max(H - 1, 1)) - 1.0,  # row → [-1, 1]
                cols * (2.0 / max(W - 1, 1)) - 1.0,  # col → [-1, 1]
            ],
            dim=1,
        )
        self.register_buffer("centers", centers)
        self.register_buffer("pixel_rc", pix)  # integer (row, col) of each candidate

        # --- Init: isotropic covariance ≈ half the candidate spacing ---
        if sigma_init is None:
            spacing = stride * (2.0 / max(H - 1, 1))  # candidate spacing in [-1,1]
            sigma_init = 0.5 * spacing
        with torch.no_grad():
            self.covs[:, 0].zero_()            # theta
            self.covs[:, 1].fill_(sigma_init)  # b_std
            self.covs[:, 2].fill_(sigma_init)  # a_std

    # ------------------------------------------------------------------
    # Opacity / background (opacity-lasso mode only)
    # ------------------------------------------------------------------

    @property
    def alphas(self) -> Tensor:
        """Per-splat weight ≥ 0 — the 0/1 prune mask, or the learnable opacity."""
        return self._alphas.clamp(min=0.0)  # type: ignore[union-attr]

    @torch.no_grad()
    def init_background(self, img: Tensor) -> None:
        """Set the background color c_bg to the image mean (opacity mode)."""
        if self.opacity:
            self.c_bg.copy_(img.reshape(img.shape[0], -1).mean(dim=1))

    # ------------------------------------------------------------------
    # Active-set bookkeeping (sparsity)
    # ------------------------------------------------------------------

    def active_mask(self, tol: float = 1e-2) -> Tensor:
        """Boolean (N,) mask of active splats (weight > ``tol``).

        Works for both modes: the 0/1 prune mask and the learnable opacity.
        """
        return self.alphas > tol

    @property
    def n_candidates(self) -> int:
        """Number of candidate pixels (the theoretical maximum splat count)."""
        return self.n_gaussians

    def n_active(self, tol: float = 1e-2) -> int:
        """Number of currently active splats."""
        return int(self.active_mask(tol).sum().item())

    @torch.no_grad()
    def set_active(self, mask: Tensor) -> None:
        """Set the active set from a boolean (N,) ``mask`` (prunes the rest)."""
        self._alphas.copy_(mask.to(self._alphas.dtype))  # type: ignore[union-attr]

    @torch.no_grad()
    def deactivate(self, idx: Tensor) -> None:
        """Prune the splats at the given flat indices."""
        self._alphas[idx] = 0.0  # type: ignore[index]

    # ------------------------------------------------------------------
    # Rendering — adds the background term in opacity mode
    # ------------------------------------------------------------------

    def _rasterize_pytorch(self, img_size: tuple[int, int]) -> Tensor:
        """Normalised blend; adds (bg_weight·c_bg) to num/den when in opacity mode."""
        if self.bg_weight <= 0.0:
            return super()._rasterize_pytorch(img_size)
        H, W = img_size
        device, dtype = self.centers.device, self.centers.dtype
        eps = 1e-7
        x = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        d_row = xx.unsqueeze(0) - self.centers[:, 0, None, None]  # (N,H,W)
        d_col = yy.unsqueeze(0) - self.centers[:, 1, None, None]
        theta = self.covs[:, 0, None, None]
        ct, st = theta.cos(), theta.sin()
        rot_x = d_row * ct - d_col * st
        rot_y = d_row * st + d_col * ct
        inv_a = 1.0 / (self.covs[:, 2].abs() + eps)[:, None, None]
        inv_b = 1.0 / (self.covs[:, 1].abs() + eps)[:, None, None]
        gs = torch.exp(-0.5 * ((rot_x * inv_a) ** 2 + (rot_y * inv_b) ** 2))  # (N,H,W)
        w = self.alphas[:, None, None, None] * gs.unsqueeze(1)                # (N,1,H,W)
        num = (w * self.colors[:, :, None, None]).sum(0)                      # (C,H,W)
        den = w.sum(0)                                                        # (1,H,W)
        num = num + self.bg_weight * self.c_bg[:, None, None]
        den = den + self.bg_weight
        return num / (den + eps)

    def forward(self, img_size=None) -> Tensor:  # type: ignore[override]
        # Opacity+background currently runs on the dense PyTorch path only.
        if self.bg_weight > 0.0:
            size = self.img_size if img_size is None else tuple(img_size)
            assert size is not None, "provide img_size or set self.img_size first"
            return self._rasterize_pytorch(size)
        return super().forward(img_size)

    # ------------------------------------------------------------------
    # Per-splat contribution (drives contribution-based pruning)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def contribution(self, img_size: tuple[int, int] | None = None) -> Tensor:
        """Per-splat assignment mass m_k = Σ_x ω_k(x), ω_k = α_k g_k / Σ_j α_j g_j.

        m_k is how many output pixels splat k effectively "owns" (Σ_k m_k = #px).
        Low m_k ⇒ the splat barely influences the render ⇒ a safe prune target.
        Returns (N,); pruned splats get exactly 0.
        """
        H, W = self.img_size if img_size is None else img_size
        device, dtype = self.centers.device, self.centers.dtype
        eps = 1e-7
        x = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        d_row = xx.unsqueeze(0) - self.centers[:, 0, None, None]  # (N,H,W)
        d_col = yy.unsqueeze(0) - self.centers[:, 1, None, None]
        theta = self.covs[:, 0, None, None]
        ct, st = theta.cos(), theta.sin()
        rot_x = d_row * ct - d_col * st
        rot_y = d_row * st + d_col * ct
        inv_a = 1.0 / (self.covs[:, 2].abs() + eps)[:, None, None]
        inv_b = 1.0 / (self.covs[:, 1].abs() + eps)[:, None, None]
        gs = torch.exp(-0.5 * ((rot_x * inv_a) ** 2 + (rot_y * inv_b) ** 2))  # (N,H,W)
        w = self.alphas[:, None, None] * gs                                   # (N,H,W)
        omega = w / (w.sum(0, keepdim=True) + eps)
        return omega.sum(dim=(1, 2))                                          # (N,)

    # ------------------------------------------------------------------
    # Initialisation helpers (centers stay fixed — never moved)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_colors_from_image(self, img: Tensor) -> None:
        """Warm-start color logits from the image pixel under each candidate."""
        C, H, W = img.shape
        idx = self.pixel_rc[:, 0].long() * W + self.pixel_rc[:, 1].long()
        sampled = img.reshape(C, -1)[:, idx].t().contiguous().clamp(1e-4, 1 - 1e-4)
        self._colors.data.copy_(torch.log(sampled / (1.0 - sampled)))  # logit

    @torch.no_grad()
    def init_from_image(self, img: Tensor, edge_floor: float = 0.05) -> None:  # noqa: ARG002
        """Warm-start colors only — the pixel-anchored centers must not move."""
        self.init_colors_from_image(img)

    # ------------------------------------------------------------------
    # Fitting lives in a separate module on purpose
    # ------------------------------------------------------------------

    def fit(self, *args, **kwargs):  # type: ignore[override]
        msg = (
            "SparseSplatImage is fitted in a dedicated module — use "
            "`sparse_optim.fit_sparse(model, img, ...)` (Adam on the 6 per-splat "
            "variables) and `sparse_optim.prune_and_refit(...)` for sparsity."
        )
        raise NotImplementedError(msg)
