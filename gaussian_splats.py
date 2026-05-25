"""2D Gaussian Splatting for image rendering.

Parameterization (per Gaussian):
    centers[i] = (cx, cy) in [-1, 1]^2  (cx along dim 0 / rows, cy along dim 1 / cols)
    covs[i]    = (theta, b, a)
        theta : rotation angle (radians)
        a     : std-dev along the rotated "x" (row) axis
        b     : std-dev along the rotated "y" (col) axis
    colors[i]  in [0, 1]^C
    alphas[i]  in [0, 1]   (learnable or fixed)
"""
from __future__ import annotations

import sys
from collections.abc import Generator

import torch
from torch import Size, Tensor, nn
from torch.optim import Adam


class GaussianSplats(nn.Module):
    """2D Gaussian Splatting renderer.

    Args:
        n_gaussians: number of Gaussian components.
        n_channels: number of color channels (default 3).
        learn_opacity: if False, opacities are frozen at ``fixed_opacity`` for
            every Gaussian. Equal-opacity training is faster and has been
            reported to remain competitive (see arxiv.org/abs/2512.12774).
        fixed_opacity: opacity value used when ``learn_opacity`` is False.
    """

    def __init__(
        self,
        n_gaussians: int,
        n_channels: int = 3,
        learn_opacity: bool = True,
        fixed_opacity: float = 1.0,
        local_render: bool = False,
    ) -> None:
        super().__init__()
        self.n_gaussians = n_gaussians
        self.n_channels = n_channels
        self.learn_opacity = learn_opacity
        # When True, each Gaussian is evaluated only on its 3-sigma AABB and
        # the contributions are scatter-added into image-sized accumulators.
        # Theoretical cost goes from O(G*H*W) to O(G*bbox_area), but on MPS the
        # scatter (index_add) cost dominates for G >= ~1000, so the dense path
        # is faster there in practice. Expect this to win on CUDA, or at small G.
        self.local_render = local_render

        self.centers = nn.Parameter(2 * torch.rand(n_gaussians, 2) - 1)
        # (theta, b, a) -- abs() so it starts positive
        self.covs = nn.Parameter(0.1 * torch.randn(n_gaussians, 3).abs())
        self._colors = nn.Parameter(torch.rand(n_gaussians, n_channels))
        if learn_opacity:
            self._alphas = nn.Parameter(torch.rand(n_gaussians))
        else:
            self.register_buffer(
                "_alphas", torch.full((n_gaussians,), float(fixed_opacity))
            )
        self.img_size: Size | tuple[int, int] | None = None

    @property
    def colors(self) -> Tensor:
        self._colors.data.clamp_(0.0, 1.0)
        return self._colors

    @property
    def alphas(self) -> Tensor:
        if self.learn_opacity:
            self._alphas.data.clamp_(0.0, 1.0)
        return self._alphas

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = [self.centers, self.covs, self._colors]
        if self.learn_opacity:
            params.append(self._alphas)
        return params

    def forward(self, img_size: Size | tuple[int, int] | None = None) -> Tensor:
        """Rasterize the Gaussians to an image of shape ``(C, H, W)``."""
        img_size = self.img_size if img_size is None else img_size
        assert img_size is not None, "Provide an image size or fit before rendering"
        size = (int(img_size[0]), int(img_size[1]))
        if self.local_render:
            return self._forward_local(size)
        return self._forward_dense(size)

    def _forward_dense(self, img_size: tuple[int, int]) -> Tensor:
        device = self.centers.device
        dtype = self.centers.dtype
        x = torch.linspace(-1, 1, img_size[0], device=device, dtype=dtype)
        y = torch.linspace(-1, 1, img_size[1], device=device, dtype=dtype)
        xx, yy = torch.meshgrid(x, y, indexing="ij")

        xx = xx.unsqueeze(0) - self.centers[:, 0, None, None]
        yy = yy.unsqueeze(0) - self.centers[:, 1, None, None]

        theta = self.covs[:, 0, None, None]
        cos_t, sin_t = theta.cos(), theta.sin()
        rot_xx = xx * cos_t - yy * sin_t
        rot_yy = xx * sin_t + yy * cos_t

        inv_a = 1.0 / (self.covs[:, 2].abs() + 1e-7)[:, None, None]
        inv_b = 1.0 / (self.covs[:, 1].abs() + 1e-7)[:, None, None]
        zz = (rot_xx * inv_a) ** 2 + (rot_yy * inv_b) ** 2
        gs = torch.exp(-0.5 * zz).unsqueeze(1)

        alphas = self.alphas[:, None, None, None]
        colors = self.colors[:, :, None, None]
        weights = gs * alphas
        return (colors * weights).sum(0) / (weights.sum(0) + 1e-7)

    def _forward_local(self, img_size: tuple[int, int]) -> Tensor:
        """Evaluate each Gaussian only on its 3-sigma AABB, scatter-add to image."""
        H, W = img_size
        device = self.centers.device
        dtype = self.centers.dtype
        G = self.n_gaussians
        C = self.n_channels

        theta = self.covs[:, 0]
        a = self.covs[:, 2].abs() + 1e-7
        b = self.covs[:, 1].abs() + 1e-7
        cos_t, sin_t = theta.cos(), theta.sin()

        # 3-sigma AABB half-extents (rotated ellipse -> AABB)
        extent_x = 3.0 * torch.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2)
        extent_y = 3.0 * torch.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2)
        half_h_px = (extent_x.detach() * (H - 1) * 0.5).clamp(min=1.0)
        half_w_px = (extent_y.detach() * (W - 1) * 0.5).clamp(min=1.0)
        P_h = int(half_h_px.max().ceil().item()) * 2 + 1
        P_w = int(half_w_px.max().ceil().item()) * 2 + 1

        # Pixel-space integer center for patch placement (no gradient through .long)
        ci = (self.centers[:, 0].detach() + 1.0) * (H - 1) * 0.5
        cj = (self.centers[:, 1].detach() + 1.0) * (W - 1) * 0.5
        ci_int = ci.round().long()
        cj_int = cj.round().long()

        rel_h = torch.arange(-(P_h // 2), P_h // 2 + 1, device=device)
        rel_w = torch.arange(-(P_w // 2), P_w // 2 + 1, device=device)
        abs_i = ci_int[:, None] + rel_h[None, :]  # (G, P_h)
        abs_j = cj_int[:, None] + rel_w[None, :]  # (G, P_w)

        mask_i = (abs_i >= 0) & (abs_i < H)
        mask_j = (abs_j >= 0) & (abs_j < W)

        # Normalized-coord offsets from center (gradient flows back to centers here)
        px_norm = abs_i.to(dtype) * 2.0 / (H - 1) - 1.0
        py_norm = abs_j.to(dtype) * 2.0 / (W - 1) - 1.0
        xx = px_norm - self.centers[:, 0:1]
        yy = py_norm - self.centers[:, 1:2]

        cos_b3 = cos_t[:, None, None]
        sin_b3 = sin_t[:, None, None]
        xx_b = xx[:, :, None]
        yy_b = yy[:, None, :]
        rot_xx = xx_b * cos_b3 - yy_b * sin_b3
        rot_yy = xx_b * sin_b3 + yy_b * cos_b3
        inv_a = (1.0 / a)[:, None, None]
        inv_b = (1.0 / b)[:, None, None]
        zz = (rot_xx * inv_a) ** 2 + (rot_yy * inv_b) ** 2
        gs = torch.exp(-0.5 * zz)  # (G, P_h, P_w)

        mask = (mask_i[:, :, None] & mask_j[:, None, :]).to(dtype)
        w = gs * mask * self.alphas[:, None, None]  # (G, P_h, P_w)

        # OOB indices clamped; their contribution is already zero via mask.
        lin_idx = abs_i.clamp(0, H - 1)[:, :, None] * W + abs_j.clamp(0, W - 1)[:, None, :]
        lin_flat = lin_idx.reshape(-1)
        w_flat = w.reshape(-1)

        colors = self.colors  # (G, C)
        contrib = (w.unsqueeze(-1) * colors[:, None, None, :]).reshape(-1, C)

        den = torch.zeros(H * W, device=device, dtype=dtype).index_add(0, lin_flat, w_flat)
        num = torch.zeros(H * W, C, device=device, dtype=dtype).index_add(0, lin_flat, contrib)
        img = num / (den.unsqueeze(-1) + 1e-7)  # (H*W, C)
        return img.t().reshape(C, H, W)

    def _fit_iterable(
        self,
        img: Tensor,
        n_iters: int = 100,
        lr: float = 0.05,
        verbose: bool = True,
    ) -> Generator[tuple[Tensor, Tensor], None, None]:
        """Fit to ``img`` (C, H, W); yield ``(rendered, loss)`` each step."""
        self.img_size = tuple(img.shape[1:])
        assert self.colors.shape[-1] == img.shape[0], "channel count mismatch"
        optim = Adam(self.trainable_parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_iters)
        loss_fn = nn.MSELoss()
        for it in range(n_iters):
            rendered = self.forward()
            err = loss_fn(rendered, img)
            if verbose:
                psnr = 10.0 * torch.log10(1.0 / err.detach().clamp(min=1e-12))
                sys.stdout.write(
                    f"\rIteration {it + 1}/{n_iters}, PSNR {psnr.item():.2f} dB"
                )
                sys.stdout.flush()
            yield rendered, err.detach().to("cpu")
            err.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()
        if verbose:
            sys.stdout.write("\n")

    def fit(
        self,
        img: Tensor,
        n_iters: int = 100,
        lr: float = 0.05,
        verbose: bool = True,
    ) -> None:
        """Fit the Gaussian parameters to ``img``."""
        for _ in self._fit_iterable(img, n_iters, lr, verbose=verbose):
            pass

    @torch.no_grad()
    def ellipse_contours(
        self,
        n_sigma: float = 2.0,
        n_points: int = 64,
    ) -> Tensor:
        """Return points on the n-sigma iso-contour of each Gaussian.

        Returns a tensor of shape ``(n_gaussians, n_points, 2)`` in the model's
        normalized ``[-1, 1]`` coordinate space. The last dim is ``(row, col)``
        (i.e. dim-0, dim-1 of the rasterized image).
        """
        device = self.centers.device
        dtype = self.centers.dtype
        t = torch.linspace(0, 2 * torch.pi, n_points, device=device, dtype=dtype)
        # In the rotated frame, the n-sigma contour is (n*a cos t, n*b sin t)
        a = self.covs[:, 2].abs() * n_sigma  # (G,)
        b = self.covs[:, 1].abs() * n_sigma
        theta = self.covs[:, 0]
        rx = a[:, None] * t.cos()[None, :]   # (G, P)
        ry = b[:, None] * t.sin()[None, :]
        cos_t = theta.cos()[:, None]
        sin_t = theta.sin()[:, None]
        # Inverse of the rotation used in forward(): R^T applied to (rx, ry)
        x = cos_t * rx + sin_t * ry
        y = -sin_t * rx + cos_t * ry
        x = x + self.centers[:, 0, None]
        y = y + self.centers[:, 1, None]
        return torch.stack([x, y], dim=-1)  # (G, P, 2)
