"""Unnormalized additive 2D Gaussian Splatting fitted via Variable Projection (VarPro).

Forward model (no normalisation, no opacity):

    G_i(x) = exp(-½ (x-μ_i)ᵀ Σ_i⁻¹ (x-μ_i))   (peak = 1)
    Î(x)   = Σ_i c_i G_i(x)

Colors c_i ∈ ℝ^C are the *linear* parameters — eliminated analytically at each
step by solving the normal equations (Kaufman/Golub-Pereyra VarPro).  Only the
geometric parameters (centers, covariances) are optimised with Adam on the
reduced loss.  Negative colors are allowed; they let Gaussians cancel overlaps.

Covariance is parameterised as (θ, b_std, a_std) matching the base class
convention so ellipse_contours and all existing tooling just works.
"""
from __future__ import annotations

import math
import sys
from collections.abc import Generator

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import (
    GaussianSplats,
    _LOCAL_N_SIGMA,
    _cuda_project,
    _cuda_rasterize,
    cuda_rasterizer_available,
)
from .em import _sobel_magnitude


class GaussianSplatsAdditive(GaussianSplats):
    """Unnormalized additive 2D Gaussian Splatting fitted via VarPro.

    Rendered image: Î(x) = Σ_k c_k G_k(x) — purely additive, no normalisation
    denominator.  Colors c_k are solved in closed form each iteration (not
    gradient-optimised); they can be negative.  No opacity.

    Fitting uses the Kaufman/Golub-Pereyra variable-projection method: for
    fixed geometry θ the optimal colors are C*(θ) = (BᵀB + λI)⁻¹ Bᵀ I
    where B[x,i] = G_i(x; θ).  Adam then minimises the reduced loss
    L_VP(θ) = (1/P)‖I − BC*(θ)‖²_F w.r.t. centers and covariances only,
    using autograd with c_star detached (Danskin's theorem: ∂L*/∂B = ∂L/∂B|_{c=c*}).

    Inherits the (θ, b_std, a_std) covariance layout and all geometry utilities
    from :class:`~splats.base.GaussianSplats`.  Only the rasterisers and the
    ``colors`` property are overridden.
    """

    # ------------------------------------------------------------------
    # Colors are raw (no sigmoid)
    # ------------------------------------------------------------------

    @property
    def colors(self) -> Tensor:
        return self._colors  # closed-form VarPro solution; can be negative

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

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
        # Colors stored directly (no logit transform)
        pixel_c = (
            img.to(device=device, dtype=dtype)
            .reshape(C, -1)[:, idx]
            .t()
            .contiguous()
        )
        self._colors.data = pixel_c
        sigma = float(2.0 / (G ** 0.5))
        self.covs.data = torch.stack(
            [
                torch.zeros(G, device=device, dtype=dtype),
                torch.full((G,), sigma, device=device, dtype=dtype),
                torch.full((G,), sigma, device=device, dtype=dtype),
            ],
            dim=1,
        )

    # ------------------------------------------------------------------
    # Rasterisers — additive: Î(x) = Σ_k c_k G_k(x), no normalisation
    # ------------------------------------------------------------------

    def _rasterize_cuda(self, img_size: tuple[int, int]) -> Tensor:
        H, W = img_size
        BLOCK = 16
        tile_bounds = ((W + BLOCK - 1) // BLOCK, (H + BLOCK - 1) // BLOCK, 1)
        xy = torch.stack(
            [(self.centers[:, 1] + 1.0) * 0.5,
             (self.centers[:, 0] + 1.0) * 0.5],
            dim=1,
        )
        scale = torch.stack(
            [self.covs[:, 1].abs() * 0.5, self.covs[:, 2].abs() * 0.5],
            dim=1,
        )
        rot = (0.5 * torch.pi - self.covs[:, 0]).unsqueeze(1)
        xys, radii, conics, num_tiles_hit = _cuda_project(  # type: ignore[misc]
            xy, scale, rot, H, W, tile_bounds
        )
        # Pass colors only (no alpha channel): rasterise_gaussians_sum gives Σ_k c_k G_k
        out = _cuda_rasterize(  # type: ignore[misc]
            xys, radii, conics, num_tiles_hit, self.colors,
            H, W, BLOCK, BLOCK, False,
        )  # (H, W, C)
        return out.permute(2, 0, 1)  # (C, H, W)

    def _rasterize_pytorch(self, img_size: tuple[int, int]) -> Tensor:
        H, W = img_size
        device, dtype = self.centers.device, self.centers.dtype
        eps = 1e-7
        x = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        d_row = xx.unsqueeze(0) - self.centers[:, 0, None, None]   # (G,H,W)
        d_col = yy.unsqueeze(0) - self.centers[:, 1, None, None]
        theta = self.covs[:, 0, None, None]
        ct, st = theta.cos(), theta.sin()
        rot_x = d_row * ct - d_col * st
        rot_y = d_row * st + d_col * ct
        inv_a = 1.0 / (self.covs[:, 2].abs() + eps)[:, None, None]
        inv_b = 1.0 / (self.covs[:, 1].abs() + eps)[:, None, None]
        gs = torch.exp(-0.5 * ((rot_x * inv_a) ** 2 + (rot_y * inv_b) ** 2))  # (G,H,W)
        return (self.colors[:, :, None, None] * gs.unsqueeze(1)).sum(0)  # (C,H,W)

    def _rasterize_pytorch_local(self, img_size: tuple[int, int]) -> Tensor:
        H, W = img_size
        device, dtype = self.centers.device, self.centers.dtype
        eps = 1e-7
        x = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        out = torch.zeros(self.n_channels, H, W, device=device, dtype=dtype)
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
            out[:, r0:r1, c0:c1] = out[:, r0:r1, c0:c1] + self.colors[k, :, None, None] * gs
        return out

    # forward() is inherited — dispatches to the overridden _rasterize_* above ✓

    # ------------------------------------------------------------------
    # Fitting — reduced-variable approach
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_colors(Bt: Tensor, img_flat: Tensor, lam: float) -> Tensor:
        """Closed-form VarPro color solution for fixed geometry.

        Solves  (BᵀB + λI) C* = BᵀI  →  C* ∈ ℝ^{N×C}.

        B ∈ ℝ^{P×N} is the basis matrix (B[x,i] = G_i(x)); Bt = Bᵀ is passed
        as (N, P) for efficiency.  No gradient is tracked through the solve —
        by Danskin's theorem ∂L_VP/∂B = ∂L/∂B|_{C=C*}, so the geometry gradient
        is correct when C* is treated as a constant during backprop.

        Bt      : (N, P)  — basis matrix transposed; Bt[i,x] = G_i(x)
        img_flat: (C, P)  — target image flattened over pixels
        lam     : float   — Tikhonov regularisation on the Gram diagonal (ε in spec)
        """
        A = Bt @ Bt.t()                         # (N, N)  Gram: BᵀB
        A.diagonal().add_(lam)                  # BᵀB + λI, in-place
        b = Bt @ img_flat.t()                   # (N, C)  BᵀI
        return torch.linalg.solve(A, b)         # (N, C)  C*

    def _fit_iterable(
        self,
        img: Tensor,
        n_iters: int = 300,
        lr: float = 0.05,
        lam: float = 1e-4,
        verbose: bool = True,
    ) -> Generator[tuple[Tensor, Tensor], None, None]:
        """VarPro iteration: solve C*(θ) exactly, then Adam step on geometry.

        Each iteration (Kaufman/Golub-Pereyra VarPro):
          1. Build Bt = Bᵀ ∈ ℝ^{N×P} (dense; suitable for N ≲ 10³) with grad
             w.r.t. centers, theta, and log_scales. B[x,i] = G_i(x; θ).
          2. Solve C* = (BᵀB + λI)⁻¹ BᵀI  without grad  (O(N²P) + O(N³)).
          3. Render Î = BC* = (C*ᵀ Bt)ᵀ — gradient flows only through Bt.
          4. Adam step on L_VP = MSE(Î, I) w.r.t. centers, theta, and log_scales.
             Autograd computes the Kaufman gradient ∂L_VP/∂B with C* detached.
          5. Sync optimised geometry back to self.covs for rasterisers.
          6. Write C* into self._colors so forward() stays consistent.

        Scales are parameterised as exp(log_s) (spec Option B) to keep them
        strictly positive everywhere — no kink, no collapse to zero, no sign
        flip in the gradient when Adam crosses zero.

        theta (rotation) and log_scales are kept as separate leaf tensors so
        that self.covs is never in the autograd graph, avoiding any in-place
        version-counter conflict between the yield and err.backward().
        """
        self.img_size = tuple(img.shape[1:])
        assert self._colors.shape[-1] == img.shape[0], "channel count mismatch"
        C, H, W = img.shape
        N = self.n_gaussians
        device, dtype = img.device, img.dtype

        # --- Log-scale reparameterisation (spec Option B) ---
        # Separate leaf tensors so self.covs is never part of the autograd graph.
        with torch.no_grad():
            theta_init = self.covs[:, 0].clone()                              # (N,)
            log_b_init = self.covs[:, 1].abs().clamp(min=1e-6).log().clone()  # log b_std
            log_a_init = self.covs[:, 2].abs().clamp(min=1e-6).log().clone()  # log a_std
        theta_p = theta_init.requires_grad_(True)
        log_scales = torch.stack([log_b_init, log_a_init], dim=1).requires_grad_(True)  # (N,2)

        from torch.optim import Adam
        optim = Adam([self.centers, theta_p, log_scales], lr=lr)

        I_flat = img.reshape(C, H * W)          # (C, P) — fixed throughout
        x_g = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        y_g = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x_g, y_g, indexing="ij")  # (H, W) each

        for i in range(n_iters):
            # --- Bt = Bᵀ ∈ R^{N×P} with grad w.r.t. centers, theta_p, log_scales ---
            d_row = xx.unsqueeze(0) - self.centers[:, 0, None, None]  # (N,H,W)
            d_col = yy.unsqueeze(0) - self.centers[:, 1, None, None]
            ct, st = theta_p[:, None, None].cos(), theta_p[:, None, None].sin()
            rot_x = d_row * ct - d_col * st
            rot_y = d_row * st + d_col * ct
            # exp(log_s) is strictly positive — smooth, no kink at zero
            inv_a = torch.exp(-log_scales[:, 1, None, None])  # 1/a_std
            inv_b = torch.exp(-log_scales[:, 0, None, None])  # 1/b_std
            Bt = torch.exp(
                -0.5 * ((rot_x * inv_a).pow(2) + (rot_y * inv_b).pow(2))
            ).reshape(N, H * W)                 # (N, P)

            # --- Solve C* = (BᵀB + λI)⁻¹ BᵀI without grad (Danskin) ---
            with torch.no_grad():
                c_star = self._solve_colors(Bt.detach(), I_flat, lam)  # (N, C)

            # --- Render Î = BC* = (C*ᵀ Bt)ᵀ; grad flows only through Bt ---
            I_hat_flat = c_star.t() @ Bt        # (C, P)
            err = F.mse_loss(I_hat_flat, I_flat)

            if verbose:
                psnr = 10.0 * torch.log10(1.0 / err.clamp(min=1e-12))
                sys.stdout.write(
                    f"\r[Additive] {i + 1}/{n_iters}  PSNR {psnr.item():.2f} dB"
                )
                sys.stdout.flush()

            yield I_hat_flat.detach().reshape(C, H, W), err.detach().cpu()

            err.backward()
            optim.step()
            optim.zero_grad()

            # Sync actual scales/theta back to self.covs for rasterisers / display
            with torch.no_grad():
                self.covs[:, 0].copy_(theta_p)
                self.covs[:, 1].copy_(log_scales[:, 0].exp())  # b_std
                self.covs[:, 2].copy_(log_scales[:, 1].exp())  # a_std
                self._colors.data.copy_(c_star)

        if verbose:
            sys.stdout.write("\n")

    def fit(
        self,
        img: Tensor,
        n_iters: int = 300,
        lr: float = 0.05,
        lam: float = 1e-4,
        saliency_init: bool = True,
        verbose: bool = True,
    ) -> None:
        """Fit via VarPro: closed-form colors + Adam on geometry.

        lam : Tikhonov regularisation (ε) added to BᵀB diagonal to keep
              the color solve well-conditioned when Gaussians overlap heavily.
        """
        if saliency_init:
            self.init_from_image(img)
        for _ in self._fit_iterable(
            img, n_iters=n_iters, lr=lr, lam=lam, verbose=verbose
        ):
            pass
