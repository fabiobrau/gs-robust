"""2D Gaussian Splatting fit by interleaving a color M-step and Adam.

Each outer iteration does:

    1. closed-form color M-step:   c_k ← Σ_x w_k(x) I(x) / Σ_x w_k(x)
    2. one Adam gradient step on geometry (centers, covs, [alphas])

Why this works (unlike full-EM interleave): for fixed geometry, the color
update is the exact minimizer of ‖Î − I‖² along c — so right after step 1,
∂L/∂c_k ≈ 0. Adam never sees a useful color gradient anyway, so we exclude
``_colors`` from the optimizer entirely and let the M-step own that block.
Adam's momentum buffers for μ, Σ are not disrupted, so it accumulates
normally — block-coordinate descent on the convex sub-problem, gradient on
the rest.
"""
from __future__ import annotations

import sys
from collections.abc import Generator

import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from gaussian_splats_em import GaussianSplatsEM


class GaussianSplatsEMPolish(GaussianSplatsEM):
    """Color M-step + Adam interleaved fit."""

    @torch.no_grad()
    def _color_m_step(self, img: Tensor, xx_grid: Tensor, yy_grid: Tensor) -> None:
        """Closed-form optimal colors at the current geometry."""
        eps = 1e-7
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
        w_k = A_k / S
        del A_k
        w_sum = w_k.sum(dim=(1, 2))
        col_num = torch.einsum("ghw,chw->gc", w_k, img)
        new_colors = (col_num / (w_sum.unsqueeze(1) + eps)).clamp(0.0, 1.0)
        self._colors.data.copy_(new_colors)

    def _fit_iterable(  # type: ignore[override]
        self,
        img: Tensor,
        n_iters: int = 200,
        lr: float = 0.02,
        saliency_init: bool = True,
        verbose: bool = True,
    ) -> Generator[tuple[Tensor, Tensor], None, None]:
        self.img_size = tuple(img.shape[1:])
        assert self._colors.shape[-1] == img.shape[0], "channel count mismatch"
        if saliency_init:
            self.init_from_image(img)

        device, dtype = self.centers.device, self.centers.dtype
        H, W = self.img_size
        x = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        y = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        xx_grid, yy_grid = torch.meshgrid(x, y, indexing="ij")

        # Adam runs on geometry only — colors are owned by the closed-form
        # M-step, and including them would just have Adam apply stale momentum
        # to a parameter that's exact-optimal each iter.
        geom_params: list = [self.centers, self.covs]
        if self.learn_opacity:
            geom_params.append(self._alphas)
        optim = Adam(geom_params, lr=lr)
        scheduler = CosineAnnealingLR(optim, T_max=n_iters)

        for it in range(n_iters):
            self._color_m_step(img, xx_grid, yy_grid)

            rendered = self.forward()
            loss = ((rendered - img) ** 2).mean()
            loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()

            mse = loss.detach()
            if verbose:
                psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))
                sys.stdout.write(
                    f"\r[color-EM+Adam] iter {it + 1}/{n_iters}, PSNR {psnr.item():.2f} dB"
                )
                sys.stdout.flush()
            yield rendered.detach(), mse.to("cpu")

        if verbose and n_iters > 0:
            sys.stdout.write("\n")

    def fit(  # type: ignore[override]
        self,
        img: Tensor,
        n_iters: int = 200,
        lr: float = 0.02,
        saliency_init: bool = True,
        verbose: bool = True,
    ) -> None:
        """Fit by alternating a color M-step and an Adam step on geometry."""
        for _ in self._fit_iterable(
            img,
            n_iters=n_iters,
            lr=lr,
            saliency_init=saliency_init,
            verbose=verbose,
        ):
            pass
