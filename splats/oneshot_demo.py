"""Thin demo adapter around the pure-NumPy :class:`oneshot.GaussianImage2D`.

``oneshot.py`` is a self-contained, deterministic NumPy implementation with its
own API (``fit(image_hwc)`` / ``render()`` / ``covariances()``).  This wrapper
leaves that file completely untouched and only bridges it to the interface the
demo drives on the torch models:

    model = OneShotGaussianImage(n_gaussians=..., n_channels=...).to(device)
    model.fit(img_chw, n_iters=...)
    rendered_chw = model()                # torch.Tensor (C, H, W)
    contours = model.ellipse_contours()   # torch.Tensor (G, P, 2) in [-1, 1]

The underlying fit runs on the CPU in NumPy.  A GPU/MPS-accelerated variant
would be a *separate* torch port (a new file), not an edit to ``oneshot.py``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .oneshot import GaussianImage2D


class OneShotGaussianImage:
    """Adapter exposing the demo's model interface over ``GaussianImage2D``.

    Constructor mirrors the torch models so the demo can build it the same way.
    ``learn_opacity`` / ``fixed_opacity`` / ``local_render`` are accepted for
    signature compatibility but have no effect (the NumPy model has no opacity
    parameter and always renders the full image).
    """

    def __init__(
        self,
        n_gaussians: int,
        n_channels: int = 3,
        learn_opacity: bool = False,
        fixed_opacity: float = 1.0,
        local_render: bool = False,
        **kwargs,
    ) -> None:
        self.n_gaussians = int(n_gaussians)
        self.n_channels = int(n_channels)
        self.learn_opacity = bool(learn_opacity)
        self.fixed_opacity = float(fixed_opacity)
        self.local_render = bool(local_render)
        self._device = torch.device("cpu")
        self._extra = kwargs  # forwarded to GaussianImage2D.__init__
        self.gi: GaussianImage2D | None = None
        self.img_size: tuple[int, int] | None = None
        self._tp: dict | None = None  # refined torch params after refine_adam()

    # -- device handling: NumPy core stays on CPU; we only track the output
    #    tensor's device so renders/contours land where the demo expects ------
    def to(self, device) -> "OneShotGaussianImage":
        self._device = torch.device(device)
        return self

    # ------------------------------------------------------------------ #
    #  fitting
    # ------------------------------------------------------------------ #
    def fit(self, img: Tensor, n_iters: int = 10, **kwargs) -> None:
        """Fit to ``img`` (C, H, W).  ``n_iters`` maps to the Adam-polish steps
        of the one-shot construction (``refine_iters``)."""
        self.img_size = tuple(img.shape[1:])
        self._target = img.detach().to(self._device).clamp(0, 1).float()  # (C,H,W)
        img_hwc = img.detach().to("cpu").clamp(0, 1).permute(1, 2, 0).numpy()
        self.gi = GaussianImage2D(
            n_splats=self.n_gaussians,
            refine_iters=int(n_iters),
            verbose=True,
            **self._extra,
        ).fit(img_hwc)

    # ------------------------------------------------------------------ #
    #  GPU/MPS Adam refinement (exact one-shot forward model, torch)
    # ------------------------------------------------------------------ #
    def _params_to_torch(self) -> dict:
        """Lift the one-shot result into torch tensors on ``self._device``.

        Matches oneshot.py exactly: Î(p) = bg + Σ_i c_i·exp(-½ q_i), with
        q = (la·dy + lb·dx)² + (ld·dx)²,  dy = row-μ_row, dx = col-μ_col  (px).
        """
        assert self.gi is not None, "call fit() first"
        dev = self._device
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
        return {
            "mu": t(self.gi.mu_).clone(),        # (G,2) (row,col) px
            "la": t(self.gi._la).clone(),        # (G,)
            "lb": t(self.gi._lb).clone(),
            "ld": t(self.gi._ld).clone(),
            "col": t(self.gi.colors_).clone(),   # (G,C) signed
            "bg": t(self.gi.background_).clone(),  # (C,)
        }

    @staticmethod
    def _tile_pred(p: dict, rows: Tensor, cols: Tensor) -> Tensor:
        """Render a horizontal pixel band: returns (C, len(rows), W)."""
        dy = rows[None, :, None] - p["mu"][:, 0, None, None]   # (G, th, W)
        dx = cols[None, None, :] - p["mu"][:, 1, None, None]
        t1 = p["la"][:, None, None] * dy + p["lb"][:, None, None] * dx
        t2 = p["ld"][:, None, None] * dx
        g = torch.exp(-0.5 * (t1 * t1 + t2 * t2))              # (G, th, W)
        th, W = g.shape[1], g.shape[2]
        contrib = p["col"].t() @ g.reshape(g.shape[0], -1)     # (C, th*W)
        return contrib.reshape(-1, th, W) + p["bg"][:, None, None]

    def refine_adam(self, steps: int, lr: float = 0.2, budget: int = 8_000_000) -> None:
        """Refine the one-shot fit with ``steps`` Adam iterations on GPU/MPS.

        Optimises positions, full-Cholesky shape, signed colors, and background
        against the exact one-shot objective.  Memory is bounded by tiling over
        pixel rows (``budget`` ≈ max G·tileRows·W elements live at once) and
        running one backward per tile, so it scales to full-resolution images.
        """
        assert self.gi is not None, "call fit() first"
        if steps <= 0:
            return
        from torch.optim import Adam

        dev = self._device
        H, W = self.gi.orig_shape_
        target = self._target[:, :H, :W]                      # (C,H,W)
        C = target.shape[0]
        p = self._params_to_torch()
        for k in p:
            p[k].requires_grad_(True)
        opt = Adam(
            [
                {"params": [p["mu"]], "lr": lr},
                {"params": [p["la"], p["lb"], p["ld"]], "lr": lr * 0.05},
                {"params": [p["col"]], "lr": lr * 0.1},
                {"params": [p["bg"]], "lr": lr * 0.02},
            ]
        )
        tile = max(1, min(H, int(budget // max(1, self.n_gaussians * W))))
        cols = torch.arange(W, dtype=torch.float32, device=dev)
        norm = float(C * H * W)

        for it in range(steps):
            opt.zero_grad()
            total_se = 0.0
            for r0 in range(0, H, tile):
                r1 = min(H, r0 + tile)
                rows = torch.arange(r0, r1, dtype=torch.float32, device=dev)
                pred = self._tile_pred(p, rows, cols)         # (C, th, W)
                se = ((pred - target[:, r0:r1, :]) ** 2).sum()
                (se / norm).backward()
                total_se += float(se.detach())
            opt.step()
            mse = total_se / norm
            psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))
            print(f"\r[oneshot-adam] {it + 1}/{steps}  PSNR {psnr:.2f} dB", end="", flush=True)
        print()

        # keep the optimised torch params so forward() renders the *same* model
        # that was just trained (no truncation gap between reported & displayed).
        self._tp = {k: v.detach() for k, v in p.items()}
        self._tile = tile
        # also write back to the NumPy model so ellipse_contours() reflects the
        # refined geometry (it reads gi.covariances() / gi.mu_).
        with torch.no_grad():
            self.gi.mu_ = self._tp["mu"].cpu().numpy().astype(np.float32)
            self.gi._la = self._tp["la"].cpu().numpy().astype(np.float32)
            self.gi._lb = self._tp["lb"].cpu().numpy().astype(np.float32)
            self.gi._ld = self._tp["ld"].cpu().numpy().astype(np.float32)
            self.gi.colors_ = self._tp["col"].cpu().numpy().astype(np.float32)
            self.gi.background_ = self._tp["bg"].cpu().numpy().astype(np.float32)
        self.gi._build_cache()

    @torch.no_grad()
    def _render_full_torch(self) -> Tensor:
        """Assemble the full (C,H,W) image from the refined torch params,
        tiled over rows to stay within the same memory budget as training."""
        H, W = self.gi.orig_shape_
        dev = self._device
        cols = torch.arange(W, dtype=torch.float32, device=dev)
        out = torch.empty(self._tp["col"].shape[1], H, W, device=dev)
        for r0 in range(0, H, self._tile):
            r1 = min(H, r0 + self._tile)
            rows = torch.arange(r0, r1, dtype=torch.float32, device=dev)
            out[:, r0:r1, :] = self._tile_pred(self._tp, rows, cols)
        return out

    # ------------------------------------------------------------------ #
    #  rendering
    # ------------------------------------------------------------------ #
    def __call__(self, img_size=None) -> Tensor:
        return self.forward(img_size)

    def forward(self, img_size=None) -> Tensor:
        assert self.gi is not None, "call fit() first"
        if self._tp is not None:  # refined: render the exact trained torch model
            return self._render_full_torch()
        rec = self.gi.render(clip=False)  # (H, W) or (H, W, C)
        rec = np.atleast_3d(rec).astype(np.float32)  # (H, W, C)
        t = torch.from_numpy(np.ascontiguousarray(rec)).permute(2, 0, 1)  # (C,H,W)
        return t.to(self._device)

    # ------------------------------------------------------------------ #
    #  ellipses for the plot — (G, P, 2) in normalized [-1, 1] (row, col)
    # ------------------------------------------------------------------ #
    def ellipse_contours(self, n_sigma: float = 2.0, n_points: int = 64) -> Tensor:
        assert self.gi is not None, "call fit() first"
        covs = self.gi.covariances().astype(np.float64)  # (G, 2, 2), (row, col)
        centers = self.gi.mu_.astype(np.float64)          # (G, 2), (row, col) px
        H, W = self.gi.orig_shape_

        # symmetric 2x2 eigendecomposition (ascending eigenvalues)
        evals, evecs = np.linalg.eigh(covs)               # (G,2), (G,2,2)
        axes = n_sigma * np.sqrt(np.clip(evals, 0.0, None))  # (G, 2)

        t = np.linspace(0.0, 2.0 * np.pi, n_points)
        unit = np.stack([np.cos(t), np.sin(t)], axis=0)   # (2, P)
        scaled = axes[:, :, None] * unit[None, :, :]      # (G, 2, P)
        pts = np.einsum("gij,gjp->gip", evecs, scaled)    # (G, 2, P) (row, col)
        pts = pts + centers[:, :, None]                   # shift to splat center
        pts = np.transpose(pts, (0, 2, 1))                # (G, P, 2)

        # pixel coords -> normalized [-1, 1]
        pts[..., 0] = pts[..., 0] / max(H - 1, 1) * 2.0 - 1.0  # row
        pts[..., 1] = pts[..., 1] / max(W - 1, 1) * 2.0 - 1.0  # col
        return torch.from_numpy(pts.astype(np.float32)).to(self._device)
