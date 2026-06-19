"""Online 2D Gaussian Splatting: Adam on random image patches.

Memory footprint during the backward pass scales with ``patch_size²`` rather
than the full image resolution, which is the main motivation.

The model is always initialised with ``GaussianSplatsEM.init_from_image``
followed by ``em_init_iters`` closed-form EM steps before the patch-based
Adam loop begins.
"""
from __future__ import annotations

import sys
from collections.abc import Generator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from .em import GaussianSplatsEM


class GaussianSplatsOnline(GaussianSplatsEM):
    """Patch-based online Adam variant initialised with EM.

    Each gradient step minimises the MSE over a single randomly sampled
    ``patch_size × patch_size`` crop of the target image.  All Gaussians
    contribute to every patch (the summation is over all ``G`` splats), so
    the computational graph is ``O(G · patch_size²)`` rather than
    ``O(G · H · W)``.

    Parameters
    ----------
    n_gaussians, n_channels, learn_opacity, fixed_opacity, local_render:
        Same as :class:`~splats.base.GaussianSplats`.
    """

    def _rasterize_patch(self, xx_patch: Tensor, yy_patch: Tensor) -> Tensor:
        """Differentiable render of a patch given its coordinate grids.

        Parameters
        ----------
        xx_patch : (pH, pW) row coordinates in [-1, 1].
        yy_patch : (pH, pW) col coordinates in [-1, 1].

        Returns
        -------
        (C, pH, pW) rendered patch in [0, 1].
        """
        eps = 1e-7
        d_row = xx_patch.unsqueeze(0) - self.centers[:, 0, None, None]   # (G, pH, pW)
        d_col = yy_patch.unsqueeze(0) - self.centers[:, 1, None, None]
        theta = self.covs[:, 0, None, None]
        ct, st = theta.cos(), theta.sin()
        rot_x = d_row * ct - d_col * st
        rot_y = d_row * st + d_col * ct
        inv_a = 1.0 / (self.covs[:, 2].abs() + eps)[:, None, None]
        inv_b = 1.0 / (self.covs[:, 1].abs() + eps)[:, None, None]
        gs = torch.exp(-0.5 * ((rot_x * inv_a) ** 2 + (rot_y * inv_b) ** 2))  # (G, pH, pW)
        w = self.alphas[:, None, None, None] * gs.unsqueeze(1)  # (G, 1, pH, pW)
        colors = self.colors[:, :, None, None]                  # (G, C, 1,  1)
        return (w * colors).sum(0) / (w.sum(0) + eps)           # (C, pH, pW)

    def _fit_iterable(  # type: ignore[override]
        self,
        img: Tensor,
        n_iters: int = 1000,
        lr: float = 0.05,
        patch_size: int = 32,
        em_init_iters: int = 10,
        em_sigma_min: float = 0.005,
        em_sigma_max: float = 0.5,
        em_lam: float = 1e-4,
        em_reseed_every: int = 5,
        log_every: int = 50,
        polish_iters: int = 0,
        polish_lr: float = 0.01,
        verbose: bool = True,
    ) -> Generator[tuple[Tensor, Tensor], None, None]:
        self.img_size = tuple(img.shape[1:])
        assert self._colors.shape[-1] == img.shape[0], "channel count mismatch"

        C, H, W = img.shape
        device, dtype = self.centers.device, self.centers.dtype

        # ------------------------------------------------------------------
        # EM initialisation
        # ------------------------------------------------------------------
        self.init_from_image(img)
        if em_init_iters > 0:
            x = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
            y = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
            xx_grid, yy_grid = torch.meshgrid(x, y, indexing="ij")
            for it in range(em_init_iters):
                do_reseed = em_reseed_every > 0 and (it + 1) % em_reseed_every == 0
                rendered_em, mse_em = self._em_step(
                    img, xx_grid, yy_grid,
                    em_sigma_min, em_sigma_max, em_lam, do_reseed,
                )
                if verbose:
                    psnr = 10.0 * torch.log10(1.0 / mse_em.clamp(min=1e-12))
                    sys.stdout.write(
                        f"\r[Online/EM init] {it + 1}/{em_init_iters}"
                        f"  PSNR {psnr.item():.2f} dB"
                    )
                    sys.stdout.flush()
                yield rendered_em, mse_em.cpu()
            if verbose:
                sys.stdout.write("\n")

        # ------------------------------------------------------------------
        # Patch-based Adam
        # ------------------------------------------------------------------
        x_full = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        y_full = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        ph = min(patch_size, H)
        pw = min(patch_size, W)

        optim = Adam(self.parameters(), lr=lr)
        last_rendered: Tensor | None = None

        for i in range(n_iters):
            r0 = int(torch.randint(0, max(H - ph + 1, 1), (1,)).item())
            c0 = int(torch.randint(0, max(W - pw + 1, 1), (1,)).item())

            xx_p, yy_p = torch.meshgrid(
                x_full[r0 : r0 + ph], y_full[c0 : c0 + pw], indexing="ij"
            )
            patch_target = img[:, r0 : r0 + ph, c0 : c0 + pw]

            rendered_patch = self._rasterize_patch(xx_p, yy_p)
            err = F.mse_loss(rendered_patch, patch_target)

            # Full-image render under no_grad for progress logging only.
            if i % log_every == 0 or last_rendered is None:
                with torch.no_grad():
                    last_rendered = self.forward()
                if verbose:
                    full_mse = F.mse_loss(last_rendered, img)
                    psnr = 10.0 * torch.log10(1.0 / full_mse.clamp(min=1e-12))
                    sys.stdout.write(
                        f"\r[Online] {i + 1}/{n_iters}  PSNR {psnr.item():.2f} dB"
                    )
                    sys.stdout.flush()

            yield last_rendered.detach(), err.detach().cpu()
            err.backward()
            optim.step()
            optim.zero_grad()

        if verbose:
            sys.stdout.write("\n")

        # ------------------------------------------------------------------
        # Full-image Adam polish (skips the em-init in GaussianSplatsEM)
        # ------------------------------------------------------------------
        if polish_iters > 0:
            if verbose:
                print(f"[Online] polishing with full-image Adam for {polish_iters} iters @ lr={polish_lr}")
            yield from super(GaussianSplatsEM, self)._fit_iterable(
                img, n_iters=polish_iters, lr=polish_lr, verbose=verbose
            )

    def fit(  # type: ignore[override]
        self,
        img: Tensor,
        n_iters: int = 1000,
        lr: float = 0.05,
        patch_size: int = 32,
        em_init_iters: int = 10,
        em_sigma_min: float = 0.005,
        em_sigma_max: float = 0.5,
        em_lam: float = 1e-4,
        em_reseed_every: int = 5,
        log_every: int = 50,
        polish_iters: int = 0,
        polish_lr: float = 0.01,
    ) -> None:
        """Fit with EM initialisation followed by patch-based Adam.

        Parameters
        ----------
        img : (C, H, W) target image in [0, 1].
        n_iters : number of Adam patch steps.
        lr : Adam learning rate.
        patch_size : height and width of each random patch in pixels.
        em_init_iters : number of closed-form EM warm-up steps (0 to skip).
        em_sigma_min, em_sigma_max, em_lam, em_reseed_every : EM hyperparameters.
        log_every : recompute full-image PSNR every this many Adam steps.
        polish_iters : full-image Adam steps run after the patch loop (0 to skip).
        polish_lr : learning rate for the full-image Adam polish.
        """
        for _ in self._fit_iterable(
            img,
            n_iters=n_iters,
            lr=lr,
            patch_size=patch_size,
            em_init_iters=em_init_iters,
            em_sigma_min=em_sigma_min,
            em_sigma_max=em_sigma_max,
            em_lam=em_lam,
            em_reseed_every=em_reseed_every,
            log_every=log_every,
            polish_iters=polish_iters,
            polish_lr=polish_lr,
        ):
            pass
