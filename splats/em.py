"""2D Gaussian Splatting fitted with EM / weighted k-means.

Setting ``∇L = 0`` for the soft-blended reconstruction loss yields a coupled
fixed-point system whose updates have the structure of weighted k-means. The
closed-form M-step is::

    c_k ← Σ_x w_k(x) I(x)     / Σ_x w_k(x)              (color)
    μ_k ← Σ_x ρ_k(x) x        / Σ_x ρ_k(x)              (position)
    Σ_k ← Σ_x ρ_k(x) d_k d_kᵀ / Σ_x ρ_k(x)  + λI        (covariance)

with E-step quantities::

    G_k(x) = exp(-½ (x-μ_k)ᵀ Σ_k⁻¹ (x-μ_k))
    w_k(x) = α_k G_k(x) / Σ_j α_j G_j(x)                (responsibility)
    Î(x)   = Σ_k w_k(x) c_k                             (current render)
    r(x)   = Î(x) - I(x)                                (residual)
    ρ_k(x) = w_k(x) · rᵀ (c_k - Î(x))                   (signed drive)

Colors are stored as sigmoid logits in ``_colors`` (consistent with the base
class). The M-step converts optimal [0,1] color values to logits before
committing them, so ``self.colors = sigmoid(_colors)`` always gives the
correct blending weights both during and after EM.

Dead splats (``Σ_x w_k(x) ≈ 0``) are reseeded periodically at the
highest-error pixel.  After EM plateaus an optional Adam polish (delegated to
:class:`~splats.base.GaussianSplats`) cleans up residual coupling.
"""
from __future__ import annotations

import sys
from collections.abc import Generator

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import GaussianSplats


def _sobel_magnitude(img: Tensor) -> Tensor:
    """Per-pixel gradient magnitude of ``(C, H, W)``, averaged over channels."""
    C = img.shape[0]
    device, dtype = img.device, img.dtype
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device, dtype=dtype,
    )
    ky = kx.t().contiguous()
    wx = kx.view(1, 1, 3, 3).expand(C, 1, 3, 3)
    wy = ky.view(1, 1, 3, 3).expand(C, 1, 3, 3)
    img_b = img.unsqueeze(0)
    gx = F.conv2d(img_b, wx, padding=1, groups=C)
    gy = F.conv2d(img_b, wy, padding=1, groups=C)
    return (gx.pow(2) + gy.pow(2)).sqrt().mean(dim=1).squeeze(0)


def _eigh_2x2(
    M: Tensor, var_min: float, var_max: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Closed-form symmetric 2×2 eigendecomposition with eigenvalue clamping.

    Returns ``(theta, std_b, std_a)`` in the parent's ``covs`` layout — ``a``
    is the std along the principal eigenvector (cos θ, sin θ) in (row, col)
    coords, ``b`` along the orthogonal direction.
    """
    M = 0.5 * (M + M.transpose(-1, -2))
    a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 1, 1]
    tr = a + c
    disc = ((a - c).pow(2) + 4.0 * b.pow(2)).clamp(min=0.0).sqrt()
    lam1 = (0.5 * (tr + disc)).clamp(var_min, var_max)
    lam2 = (0.5 * (tr - disc)).clamp(var_min, var_max)
    vx = lam1 - c
    vy = b
    norm = (vx.pow(2) + vy.pow(2)).sqrt().clamp(min=1e-12)
    axis_aligned = b.abs() < 1e-12
    vx_aa = torch.where(a >= c, torch.ones_like(a), torch.zeros_like(a))
    vy_aa = torch.where(a >= c, torch.zeros_like(a), torch.ones_like(a))
    vx = torch.where(axis_aligned, vx_aa, vx / norm)
    vy = torch.where(axis_aligned, vy_aa, vy / norm)
    theta = torch.atan2(vy, vx)
    return theta, lam2.sqrt(), lam1.sqrt()


class GaussianSplatsEM(GaussianSplats):
    """EM / weighted-k-means fit variant of :class:`~splats.base.GaussianSplats`."""

    @torch.no_grad()
    def init_from_image(self, img: Tensor, edge_floor: float = 0.05) -> None:
        """Saliency-weighted init: centers sampled from gradient-magnitude PMF."""
        C, H, W = img.shape
        device, dtype = self.centers.device, self.centers.dtype
        sal = _sobel_magnitude(img.to(device=device, dtype=dtype))
        sal = sal + (edge_floor * sal.mean() + 1e-7)
        probs = (sal / sal.sum()).flatten()
        G = self.n_gaussians
        idx = torch.multinomial(probs, G, replacement=True)
        rows = idx // W
        cols = idx % W
        new_row = rows.to(dtype) * (2.0 / max(H - 1, 1)) - 1.0
        new_col = cols.to(dtype) * (2.0 / max(W - 1, 1)) - 1.0
        self.centers.data = torch.stack([new_row, new_col], dim=1)
        # Store logits so that sigmoid(_colors) = pixel color
        pixel_c = (
            img.to(device=device, dtype=dtype)
            .reshape(C, -1)[:, idx]
            .t()
            .contiguous()
            .clamp(1e-6, 1.0 - 1e-6)
        )
        self._colors.data = pixel_c.logit()
        sigma = float(2.0 / (G ** 0.5))
        self.covs.data = torch.stack(
            [
                torch.zeros(G, device=device, dtype=dtype),
                torch.full((G,), sigma, device=device, dtype=dtype),
                torch.full((G,), sigma, device=device, dtype=dtype),
            ],
            dim=1,
        )

    @torch.no_grad()
    def _em_step(
        self,
        img: Tensor,
        xx_grid: Tensor,
        yy_grid: Tensor,
        sigma_min: float,
        sigma_max: float,
        lam: float,
        do_reseed: bool,
    ) -> tuple[Tensor, Tensor]:
        """One E-step + closed-form M-step. Returns ``(rendered, mse)`` pre-step."""
        eps = 1e-7
        C, H, W = img.shape

        # ---------- E-step ----------
        d_row = xx_grid.unsqueeze(0) - self.centers[:, 0, None, None]
        d_col = yy_grid.unsqueeze(0) - self.centers[:, 1, None, None]
        theta = self.covs[:, 0, None, None]
        ct, st = theta.cos(), theta.sin()
        rot_x = d_row * ct - d_col * st
        rot_y = d_row * st + d_col * ct
        del d_row, d_col
        a_std = (self.covs[:, 2].abs() + eps)[:, None, None]
        b_std = (self.covs[:, 1].abs() + eps)[:, None, None]
        gk = torch.exp(-0.5 * ((rot_x / a_std).pow(2) + (rot_y / b_std).pow(2)))
        del rot_x, rot_y
        A_k = gk * self.alphas[:, None, None]
        del gk
        S = A_k.sum(dim=0) + eps
        w_k = A_k / S                                                       # (G,H,W)
        del A_k

        # Use sigmoid colors (consistent with forward())
        colors = self.colors                                                # (G,C)
        I_hat = torch.einsum("ghw,gc->chw", w_k, colors)                    # (C,H,W)
        r = I_hat - img                                                     # (C,H,W)

        r_dot_ck = torch.einsum("gc,chw->ghw", colors, r)                   # (G,H,W)
        r_dot_I = (r * I_hat).sum(dim=0)                                    # (H,W)
        rho_k = w_k * (r_dot_ck - r_dot_I.unsqueeze(0))                     # (G,H,W)
        del r_dot_ck

        # ---------- M-step: color ----------
        w_sum = w_k.sum(dim=(1, 2))                                         # (G,)
        col_num = torch.einsum("ghw,chw->gc", w_k, img)                     # (G,C)
        # Optimal colors in [0,1]; store as logits for sigmoid consistency
        new_colors = (col_num / (w_sum.unsqueeze(1) + eps)).clamp(1e-6, 1.0 - 1e-6)
        del w_k

        # ---------- M-step: position ----------
        rho_sum = rho_k.sum(dim=(1, 2))
        rho_x = (rho_k * xx_grid.unsqueeze(0)).sum(dim=(1, 2))
        rho_y = (rho_k * yy_grid.unsqueeze(0)).sum(dim=(1, 2))
        live = rho_sum.abs() > eps
        rho_safe = torch.where(live, rho_sum, torch.ones_like(rho_sum))
        new_mu_x = torch.where(live, rho_x / rho_safe, self.centers[:, 0])
        new_mu_y = torch.where(live, rho_y / rho_safe, self.centers[:, 1])
        new_centers = torch.stack([new_mu_x, new_mu_y], dim=1).clamp(-1.0, 1.0)

        # ---------- M-step: covariance ----------
        d_row_n = xx_grid.unsqueeze(0) - new_centers[:, 0, None, None]
        d_col_n = yy_grid.unsqueeze(0) - new_centers[:, 1, None, None]
        Mxx = (rho_k * d_row_n.pow(2)).sum(dim=(1, 2))
        Myy = (rho_k * d_col_n.pow(2)).sum(dim=(1, 2))
        Mxy = (rho_k * d_row_n * d_col_n).sum(dim=(1, 2))
        del rho_k, d_row_n, d_col_n
        var_floor = sigma_min ** 2
        Mxx = torch.where(live, Mxx / rho_safe, torch.full_like(Mxx, var_floor)) + lam
        Myy = torch.where(live, Myy / rho_safe, torch.full_like(Myy, var_floor)) + lam
        Mxy = torch.where(live, Mxy / rho_safe, torch.zeros_like(Mxy))
        M = torch.stack(
            [torch.stack([Mxx, Mxy], dim=-1), torch.stack([Mxy, Myy], dim=-1)],
            dim=-2,
        )
        new_theta, new_b, new_a = _eigh_2x2(M, var_floor, sigma_max ** 2)

        # ---------- Commit ----------
        self._colors.data.copy_(new_colors.logit())
        self.centers.data.copy_(new_centers)
        self.covs.data.copy_(torch.stack([new_theta, new_b, new_a], dim=1))

        # ---------- Reseed dead splats ----------
        if do_reseed:
            dead = w_sum < eps
            n_dead = int(dead.sum().item())
            if n_dead > 0:
                err_map = r.pow(2).sum(dim=0).flatten()
                top = err_map.topk(n_dead).indices
                rows = (top // W).to(self.centers.dtype)
                cols_ = (top % W).to(self.centers.dtype)
                self.centers.data[dead, 0] = rows * (2.0 / max(H - 1, 1)) - 1.0
                self.centers.data[dead, 1] = cols_ * (2.0 / max(W - 1, 1)) - 1.0
                self.covs.data[dead, 0] = 0.0
                self.covs.data[dead, 1] = 2.0 * sigma_min
                self.covs.data[dead, 2] = 2.0 * sigma_min
                pixel_c = img.reshape(C, -1)[:, top].t().clamp(1e-6, 1.0 - 1e-6)
                self._colors.data[dead] = pixel_c.logit()

        mse = r.pow(2).mean()
        return I_hat.detach(), mse.detach()

    def _fit_iterable(  # type: ignore[override]
        self,
        img: Tensor,
        n_iters: int = 50,
        sigma_min: float = 0.005,
        sigma_max: float = 0.5,
        lam: float = 1e-4,
        reseed_every: int = 5,
        polish_iters: int = 0,
        polish_lr: float = 0.01,
        saliency_init: bool = True,
        verbose: bool = True,
    ) -> Generator[tuple[Tensor, Tensor], None, None]:
        self.img_size = tuple(img.shape[1:])
        assert self._colors.shape[-1] == img.shape[0], "channel count mismatch"
        if saliency_init:
            self.init_from_image(img)

        device, dtype = self.centers.device, self.centers.dtype
        H, W = self.img_size
        x = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        xx_grid, yy_grid = torch.meshgrid(x, y, indexing="ij")

        for it in range(n_iters):
            do_reseed = reseed_every > 0 and (it + 1) % reseed_every == 0
            rendered, mse = self._em_step(
                img, xx_grid, yy_grid, sigma_min, sigma_max, lam, do_reseed
            )
            if verbose:
                psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))
                sys.stdout.write(
                    f"\r[EM] iter {it + 1}/{n_iters}, PSNR {psnr.item():.2f} dB"
                )
                sys.stdout.flush()
            yield rendered, mse.to("cpu")

        if verbose and n_iters > 0:
            sys.stdout.write("\n")

        if polish_iters > 0:
            if verbose:
                print(f"[EM] polishing with Adam for {polish_iters} iters @ lr={polish_lr}")
            yield from super()._fit_iterable(
                img, n_iters=polish_iters, lr=polish_lr, verbose=verbose
            )

    def fit(  # type: ignore[override]
        self,
        img: Tensor,
        n_iters: int = 50,
        sigma_min: float = 0.005,
        sigma_max: float = 0.5,
        lam: float = 1e-4,
        reseed_every: int = 5,
        polish_iters: int = 0,
        polish_lr: float = 0.01,
        saliency_init: bool = True,
        verbose: bool = True,
    ) -> None:
        """Fit with closed-form EM (+ optional Adam polish)."""
        for _ in self._fit_iterable(
            img,
            n_iters=n_iters,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            lam=lam,
            reseed_every=reseed_every,
            polish_iters=polish_iters,
            polish_lr=polish_lr,
            saliency_init=saliency_init,
            verbose=verbose,
        ):
            pass
