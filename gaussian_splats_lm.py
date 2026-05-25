"""2D Gaussian Splatting fitted with Levenberg-Marquardt.

Same renderer as ``gaussian_splats.py``; only the optimizer changes.
LM is a damped Gauss-Newton method that operates on the per-pixel residual
vector ``r = rendered - target`` and at each iteration solves

    (JᵀJ + λ I) δ = -Jᵀ r

J is never materialized — the normal-equation matvec ``v ↦ JᵀJ v + λ v`` is
computed matrix-free via one JVP (``J v``) + one VJP (``Jᵀ ·``) per step, and
the system is solved with conjugate gradients. Peak memory is therefore
dominated by a few forward activations, not by an M×N Jacobian.

On accepted steps λ is divided by ``lam_down``; on rejected steps it is
multiplied by ``lam_up`` (classic Marquardt strategy).
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Generator

import torch
from torch import Tensor
from torch.func import functional_call, jvp, vjp

from gaussian_splats import GaussianSplats


def _cg(matvec: Callable[[Tensor], Tensor], b: Tensor, max_iter: int, tol: float) -> Tensor:
    """Conjugate-gradient solve of ``A x = b`` for symmetric PSD ``matvec``."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = (r * r).sum()
    tol_sq = (tol * b.norm()) ** 2 + 1e-30
    for _ in range(max_iter):
        Ap = matvec(p)
        alpha = rs / ((p * Ap).sum() + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if rs_new < tol_sq:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


class GaussianSplatsLM(GaussianSplats):
    """LM-fit variant of :class:`GaussianSplats`."""

    # The parent's ``colors`` / ``alphas`` properties do an in-place
    # ``.data.clamp_(0, 1)``. Under ``functional_call`` the parameters are
    # views into our flat LM vector, so that mutation would silently corrupt
    # the optimization state and break the autograd graph. Bounds are enforced
    # by an explicit projection after each accepted step instead.
    @property
    def colors(self) -> Tensor:  # type: ignore[override]
        return self._colors

    @property
    def alphas(self) -> Tensor:  # type: ignore[override]
        return self._alphas

    def _fit_iterable(  # type: ignore[override]
        self,
        img: Tensor,
        n_iters: int = 50,
        lm_lambda: float = 1e-3,
        lam_up: float = 10.0,
        lam_down: float = 10.0,
        max_inner: int = 4,
        cg_iters: int = 30,
        cg_tol: float = 1e-3,
        verbose: bool = True,
    ) -> Generator[tuple[Tensor, Tensor], None, None]:
        """Fit to ``img`` (C, H, W); yield ``(rendered, mse)`` each step."""
        self.img_size = tuple(img.shape[1:])
        assert self.colors.shape[-1] == img.shape[0], "channel count mismatch"
        target = img.detach()
        device = target.device
        dtype = target.dtype

        named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        names = [n for n, _ in named]
        shapes = [p.shape for _, p in named]
        numels = [p.numel() for _, p in named]
        total = sum(numels)

        # Indices in the flat vector that must stay in [0, 1] (colors, learnable alphas).
        bound_mask = torch.zeros(total, dtype=torch.bool, device=device)
        offset = 0
        for n, k in zip(names, numels):
            if n in ("_colors", "_alphas"):
                bound_mask[offset:offset + k] = True
            offset += k

        def vec_to_param_dict(vec: Tensor) -> dict[str, Tensor]:
            d: dict[str, Tensor] = {}
            o = 0
            for n, shape, k in zip(names, shapes, numels):
                d[n] = vec[o:o + k].view(shape)
                o += k
            return d

        def residuals(vec: Tensor) -> Tensor:
            params_dict = vec_to_param_dict(vec)
            rendered = functional_call(self, params_dict, args=())
            return (rendered - target).reshape(-1)

        p_vec = torch.cat([p.detach().flatten() for _, p in named]).clone()
        p_vec[bound_mask] = p_vec[bound_mask].clamp(0.0, 1.0)

        lam = float(lm_lambda)
        _ = dtype  # silence unused

        with torch.no_grad():
            r = residuals(p_vec)
            loss = 0.5 * (r * r).sum()

        for it in range(n_iters):
            # One VJP function built per LM iter; reused inside the inner λ
            # search and for the RHS. ``vjp_func`` closes over the forward
            # activations at p_vec, so reusing it is O(backward) per call.
            r_val, vjp_func = vjp(residuals, p_vec)
            rhs = -vjp_func(r_val)[0]

            def matvec(v: Tensor, lam_local: float = lam) -> Tensor:
                # (JᵀJ + λI) v  =  Jᵀ (J v) + λ v
                _, jv = jvp(residuals, (p_vec,), (v,))
                jtjv = vjp_func(jv)[0]
                return jtjv + lam_local * v

            accepted = False
            for _ in range(max_inner):
                delta = _cg(lambda v, _l=lam: matvec(v, _l), rhs, cg_iters, cg_tol)
                trial = p_vec + delta
                trial[bound_mask] = trial[bound_mask].clamp(0.0, 1.0)
                with torch.no_grad():
                    r_new = residuals(trial)
                    loss_new = 0.5 * (r_new * r_new).sum()
                if loss_new < loss:
                    p_vec, r, loss = trial, r_new, loss_new
                    lam = max(lam / lam_down, 1e-12)
                    accepted = True
                    break
                lam = min(lam * lam_up, 1e12)

            with torch.no_grad():
                rendered = functional_call(self, vec_to_param_dict(p_vec), args=()).detach()
            mse = (2.0 * loss / r.numel()).clamp(min=1e-12)
            if verbose:
                psnr = 10.0 * torch.log10(1.0 / mse)
                flag = "" if accepted else " (rejected)"
                sys.stdout.write(
                    f"\rIter {it + 1}/{n_iters}, λ={lam:.2e}, "
                    f"PSNR {psnr.item():.2f} dB{flag}"
                )
                sys.stdout.flush()
            yield rendered, mse.detach().to("cpu")

        with torch.no_grad():
            o = 0
            for (_, p), k in zip(named, numels):
                p.data.copy_(p_vec[o:o + k].view(p.shape))
                o += k

        if verbose:
            sys.stdout.write("\n")

    def fit(  # type: ignore[override]
        self,
        img: Tensor,
        n_iters: int = 50,
        lm_lambda: float = 1e-3,
        cg_iters: int = 30,
        cg_tol: float = 1e-3,
        verbose: bool = True,
    ) -> None:
        """Fit the Gaussian parameters to ``img`` with Levenberg-Marquardt."""
        for _ in self._fit_iterable(
            img,
            n_iters=n_iters,
            lm_lambda=lm_lambda,
            cg_iters=cg_iters,
            cg_tol=cg_tol,
            verbose=verbose,
        ):
            pass
