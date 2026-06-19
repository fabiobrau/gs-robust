"""2D Gaussian Splatting fitted with Variable Projection (VarPro) over a
Partition-of-Unity (PoU) basis.

================================================================================
Theory
================================================================================
Forward model (PoU, no alphas):

    G_k(x) = exp(-½ (x-μ_k)ᵀ Σ_k⁻¹ (x-μ_k))
    w_k(x) = G_k(x) / (Σ_j G_j(x) + ε)        ← partition-of-unity weights
    Î(x)  = Σ_k w_k(x) c_k                    ← convex combination → bounded

Stacking pixels:  Î = W(θ) C,  with  W ∈ ℝ^(|Ω| × G),  C ∈ ℝ^(G × C_ch).

Reduced (variable-projected) loss — eliminate C analytically:

    C*(θ) = (Wᵀ W + δI)⁻¹ Wᵀ I
    L_VP(θ) = ‖W C*(θ) - I‖² / |Ω|

Optimize geometry θ = {μ_k, Σ_k} only via Adam. C is *not* a learnable
parameter — it's a deterministic function of θ solved fresh each step.

================================================================================
Algorithm (per outer iter — Kaufman/Golub-Pereyra simplification)
================================================================================
  1. Build W(θ) on the pixel grid with autograd enabled on θ
  2. Solve C*(θ) = (WᵀW + δI)⁻¹ Wᵀ I   [Cholesky on detached W — no grad]
  3. Î = W C*                             [differentiable w.r.t. θ via W]
  4. L_VP = ‖Î - I‖² / |Ω|              [differentiable w.r.t. θ only]
  5. Backprop through W(θ) and Adam-step θ
  6. Periodically: reseed dead splats, reset their Adam moments

Dropping the ∂C*/∂θ term (Kaufman simplification) is standard practice; the
omitted term is second-order at convergence and saves a dense Jacobian-vector
product.

Covariance parameterisation: covs[:, 0] = rotation angle; covs[:, 1:] = log σ
(unconstrained). Log-scale gives smooth, well-conditioned gradients everywhere
and is clamped to [log σ_min, log σ_max] after each step.  The covs are
converted back to linear scale before the optional Adam polish so the base-
class rasterizer (which uses .abs()) works correctly.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Generator

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import GaussianSplats


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sobel_magnitude(img: Tensor) -> Tensor:
    """Per-pixel gradient magnitude, averaged over channels."""
    C = img.shape[0]
    device, dtype = img.device, img.dtype
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    ky = kx.t().contiguous()
    wx = kx.view(1, 1, 3, 3).expand(C, 1, 3, 3)
    wy = ky.view(1, 1, 3, 3).expand(C, 1, 3, 3)
    img_b = img.unsqueeze(0)
    gx = F.conv2d(img_b, wx, padding=1, groups=C)
    gy = F.conv2d(img_b, wy, padding=1, groups=C)
    return (gx.pow(2) + gy.pow(2)).sqrt().mean(dim=1).squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────
class GaussianSplatsVarPro(GaussianSplats):
    """True VarPro + PoU fit.

    Colors are eliminated by closed-form least squares at every step (Kaufman
    simplification of Golub-Pereyra).  Only geometry (centers, covs) is
    optimized — by a persistent Adam instance that accumulates moment estimates
    across outer iterations.  Alphas are not used.

    During VarPro, covs[:, 1:] store log(σ) (unconstrained).  They are
    converted to linear scale before the optional Adam polish so the base-class
    rasterizer (.abs() convention) sees the correct values.
    """

    # ------------------------------------------------------------------
    # Init: saliency-weighted center placement with log-σ covariances
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_from_image(self, img: Tensor, edge_floor: float = 0.05) -> None:
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
        # Colors will be overwritten by the VarPro solve; init for consistency.
        pixel_c = (
            img
            .to(device=device, dtype=dtype)
            .reshape(C, -1)[:, idx]
            .t()
            .contiguous()
            .clamp(1e-6, 1.0 - 1e-6)
        )
        self._colors.data = pixel_c.logit()
        # covs[:, 1:] in log-scale
        log_sigma = math.log(2.0 / (G ** 0.5))
        self.covs.data = torch.stack(
            [
                torch.zeros(G, device=device, dtype=dtype),
                torch.full((G,), log_sigma, device=device, dtype=dtype),
                torch.full((G,), log_sigma, device=device, dtype=dtype),
            ],
            dim=1,
        )

    # ------------------------------------------------------------------
    # One outer VarPro step: build W, solve C*, Adam-step geometry
    # ------------------------------------------------------------------
    def _varpro_step(
        self,
        img: Tensor,
        xx_grid: Tensor,
        yy_grid: Tensor,
        optimizer: torch.optim.Adam,
        sigma_min: float,
        sigma_max: float,
        ridge: float,
        do_reseed: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One outer step. Returns (rendered, mse, C_star)."""
        eps = 1e-7
        C, H, W = img.shape
        G_n = self.n_gaussians
        log_sigma_min = math.log(sigma_min)
        log_sigma_max = math.log(sigma_max)

        # === Build PoU weights — autograd graph live on centers/covs ===
        d_row = xx_grid.unsqueeze(0) - self.centers[:, 0, None, None]
        d_col = yy_grid.unsqueeze(0) - self.centers[:, 1, None, None]
        theta = self.covs[:, 0, None, None]
        ct, st = theta.cos(), theta.sin()
        rot_x = d_row * ct - d_col * st
        rot_y = d_row * st + d_col * ct
        # log-scale → std via exp (smooth, no abs kink)
        a_std = self.covs[:, 2].exp()[:, None, None]
        b_std = self.covs[:, 1].exp()[:, None, None]
        gk = torch.exp(-0.5 * ((rot_x / a_std).pow(2) + (rot_y / b_std).pow(2)))
        S = gk.sum(dim=0) + eps
        w_k = gk / S  # (G, H, W)

        # === Solve C* in closed form — detach W so ∂C*/∂θ is dropped ===
        # This is the Kaufman simplification of Golub-Pereyra: ignore the
        # term coupling C* back to θ.  Standard practice; near-exact at conv.
        W_flat = w_k.reshape(G_n, -1)   # (G, HW)
        I_flat = img.reshape(C, -1).t().contiguous()  # (HW, C)
        with torch.no_grad():
            Gram = W_flat.detach() @ W_flat.detach().t()
            Gram = Gram + ridge * torch.eye(G_n, device=Gram.device, dtype=Gram.dtype)
            rhs = W_flat.detach() @ I_flat
            L_chol = torch.linalg.cholesky(Gram)
            C_star = torch.cholesky_solve(rhs, L_chol)  # (G, C) — constant

        # === Reconstruct (differentiable w.r.t. θ through w_k) ===
        # C_star has no grad; gradient flows through W only (Kaufman).
        I_hat = torch.einsum("ghw,gc->chw", w_k, C_star)
        L_VP = F.mse_loss(I_hat, img)

        # === Adam step on geometry ===
        optimizer.zero_grad()
        L_VP.backward()
        optimizer.step()

        # === Post-step clamps and color warm-start ===
        with torch.no_grad():
            self.centers.data.clamp_(-1.0, 1.0)
            self.covs.data[:, 1:].clamp_(log_sigma_min, log_sigma_max)
            # Keep _colors updated so polish phase starts from a good point.
            # Clamp only to avoid logit(0)/logit(1) = ±inf; not to bias solve.
            self._colors.data.copy_(C_star.clamp(1e-6, 1.0 - 1e-6).logit())

        # === Reseed splats with vanishing PoU mass ===
        if do_reseed:
            with torch.no_grad():
                w_sum = w_k.detach().sum(dim=(1, 2))  # (G,)
                dead = w_sum < eps
                n_dead = int(dead.sum().item())
                if n_dead > 0:
                    r = I_hat.detach() - img
                    err_map = r.pow(2).sum(dim=0).flatten()
                    top = err_map.topk(n_dead).indices
                    rows = (top // W).to(self.centers.dtype)
                    cols_ = (top % W).to(self.centers.dtype)
                    log_init = math.log(2.0 * sigma_min)
                    self.centers.data[dead, 0] = rows * (2.0 / max(H - 1, 1)) - 1.0
                    self.centers.data[dead, 1] = cols_ * (2.0 / max(W - 1, 1)) - 1.0
                    self.covs.data[dead, 0] = 0.0
                    self.covs.data[dead, 1] = log_init
                    self.covs.data[dead, 2] = log_init
                    pixel_c = img.reshape(C, -1)[:, top].t().clamp(1e-6, 1.0 - 1e-6)
                    self._colors.data[dead] = pixel_c.logit()
                    # Zero stale Adam moments so reseeded splats start fresh.
                    try:
                        for param in (self.centers, self.covs):
                            state = optimizer.state[param]
                            state['exp_avg'][dead] = 0.0
                            state['exp_avg_sq'][dead] = 0.0
                    except KeyError:
                        pass  # first reseed before Adam has seen any step

        mse = L_VP.detach()
        return I_hat.detach(), mse, C_star.detach()

    # ------------------------------------------------------------------
    # Driver loop
    # ------------------------------------------------------------------
    def _fit_iterable(  # type: ignore[override]
        self,
        img: Tensor,
        n_iters: int = 10,
        sigma_min: float = 0.005,
        sigma_max: float = 0.5,
        lam: float = 1e-4,   # accepted for API compatibility; unused in VarPro
        ridge: float = 1e-6,
        lr_centers: float = 1e-2,
        lr_covs: float = 1e-2,
        reseed_every: int = 5,
        polish_iters: int = 100,
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
        pixel = 2.0 / max(min(H, W) - 1, 1)

        # Single Adam instance that persists across outer iters so moment
        # estimates accumulate (important for geometry convergence).
        optimizer = torch.optim.Adam(
            [
                {"params": [self.centers], "lr": lr_centers},
                {"params": [self.covs], "lr": lr_covs},
            ]
        )

        for it in range(n_iters):
            do_reseed = reseed_every > 0 and (it + 1) % reseed_every == 0
            rendered, mse, _C_star = self._varpro_step(
                img,
                xx_grid,
                yy_grid,
                optimizer,
                sigma_min,
                sigma_max,
                ridge,
                do_reseed,
            )
            if verbose:
                psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))
                with torch.no_grad():
                    # covs[:, 1:] are log-scale during VarPro phase
                    smaller = torch.minimum(
                        self.covs[:, 1].exp(), self.covs[:, 2].exp()
                    )
                    n_subpixel = int((smaller < pixel).sum().item())
                sys.stdout.write(
                    f"\r[VarPro] iter {it + 1}/{n_iters}, "
                    f"PSNR {psnr.item():.2f} dB, "
                    f"sub-pixel {n_subpixel}/{self.n_gaussians}"
                )
                sys.stdout.flush()
            yield rendered, mse.to("cpu")

        if verbose and n_iters > 0:
            sys.stdout.write("\n")

        if polish_iters > 0:
            # Convert log-scale covs → abs-scale so base-class rasterizer
            # (.abs() convention) and ellipse_contours() work correctly.
            with torch.no_grad():
                self.covs.data[:, 1:] = self.covs.data[:, 1:].exp()
            if verbose:
                print(
                    f"[VarPro] polishing with Adam for {polish_iters} iters @ lr={polish_lr}"
                )
            yield from super()._fit_iterable(
                img, n_iters=polish_iters, lr=polish_lr, verbose=verbose
            )

    def fit(  # type: ignore[override]
        self,
        img: Tensor,
        n_iters: int = 10,
        sigma_min: float = 0.005,
        sigma_max: float = 0.5,
        lam: float = 1e-4,   # accepted for API compatibility; unused in VarPro
        ridge: float = 1e-6,
        lr_centers: float = 1e-2,
        lr_covs: float = 1e-2,
        reseed_every: int = 5,
        polish_iters: int = 100,
        polish_lr: float = 0.01,
        saliency_init: bool = True,
        verbose: bool = True,
    ) -> None:
        """Fit with true VarPro (closed-form colors + Adam on geometry)."""
        for _ in self._fit_iterable(
            img,
            n_iters=n_iters,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            lam=lam,
            ridge=ridge,
            lr_centers=lr_centers,
            lr_covs=lr_covs,
            reseed_every=reseed_every,
            polish_iters=polish_iters,
            polish_lr=polish_lr,
            saliency_init=saliency_init,
            verbose=verbose,
        ):
            pass
