"""Adam fit + sparsifying pruning for the pixel-anchored sparse splat model.

Kept deliberately **separate** from the in-package fitters (EM / VarPro /
additive VarPro).  The model :class:`splats.sparse_splat.SparseSplatImage`
renders with the original **normalised** blend (no opacity):

    Î(x) = Σ_{k active} g_k(x) c_k / Σ_{k active} g_k(x)

so each pixel is a convex combination of the colors of the Gaussians reaching
it.  Only the 6 per-splat variables (3 colors + 3 covariances) are optimised.

Why sparsity is a *prune*, not a loss penalty
---------------------------------------------
In a normalised blend with no opacity there is no per-splat coefficient that can
be driven to zero to switch a splat off:

* a zero color is not "off" — it contributes black to the weighted average;
* any weight that appears in *both* numerator and denominator is scale-invariant
  (Î is unchanged if all weights are scaled together), so an L1 on it just
  shrinks everything uniformly instead of selecting which splats to drop.

The honest mechanism is therefore **pruning**: deactivate the splats that
contribute least (smallest assignment mass), and stop when reconstruction
quality would drop past a budget.  ``prune_and_refit`` does exactly this and its
knob ``max_psnr_drop`` (λ) is the quality you are willing to trade for fewer
splats — to be tuned later.

Run directly to fit + sparsify an image and save a comparison figure::

    python3 sparse_optim.py data/gauss.jpeg --size 128 --stride 3 \
        --n-iters 400 --prune --max-psnr-drop 1.0
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Generator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from splats.sparse_splat import SparseSplatImage


def _psnr(mse: float) -> float:
    return 10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()


# ---------------------------------------------------------------------------
# Adam fit (data term only — the 6 per-splat variables)
# ---------------------------------------------------------------------------

def fit_sparse(
    model: SparseSplatImage,
    img: Tensor,
    n_iters: int = 400,
    lr: float = 0.05,
    sigma_min: float = 1e-3,
    verbose: bool = True,
    tag: str = "fit",
) -> Generator[tuple[Tensor, Tensor], None, None]:
    """Plain Adam fit of ``model`` to ``img`` (C, H, W) on the normalised render.

    Optimises only the active splats' colors + covariances (centers are fixed
    buffers; the active mask is fixed during a fit — pruning happens between
    fits).  Yields ``(rendered, mse)`` each iteration.
    """
    model.img_size = tuple(img.shape[1:])
    assert model.n_channels == img.shape[0], "channel count mismatch"

    optim = Adam([model._colors, model.covs], lr=lr)
    for i in range(n_iters):
        rendered = model.forward()
        mse = F.mse_loss(rendered, img)

        mse.backward()
        optim.step()
        optim.zero_grad()

        # Keep scales strictly positive / non-degenerate (abs()-param kinks at 0).
        with torch.no_grad():
            model.covs[:, 1:].abs_().clamp_(min=sigma_min)

        if verbose:
            psnr = 10.0 * torch.log10(1.0 / mse.detach().clamp(min=1e-12))
            sys.stdout.write(
                f"\r[{tag}] {i + 1}/{n_iters}  PSNR {psnr.item():5.2f} dB  "
                f"active {model.n_active()}/{model.n_candidates}"
            )
            sys.stdout.flush()
        yield rendered.detach(), mse.detach().cpu()

    if verbose:
        sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Alternative sparsity: opacity + lasso (proximal) + background
# ---------------------------------------------------------------------------

def fit_opacity_lasso(
    model: SparseSplatImage,
    img: Tensor,
    n_iters: int = 400,
    lr: float = 0.05,
    l1: float = 5e-2,
    energy_weighted: bool = False,
    sigma_min: float = 1e-3,
    active_tol: float = 1e-2,
    verbose: bool = True,
) -> Generator[tuple[Tensor, Tensor, int], None, None]:
    """Adam fit + L1 lasso on the per-splat opacity (model built with opacity=True).

    The data term (MSE on the normalised+background render) is optimised by Adam;
    the lasso is applied as a **proximal** soft-threshold ``α ← max(0, α − τ)``
    after each step, which drives unneeded opacities to exactly zero.  The
    model's background term breaks the scale-invariance that would otherwise make
    this lasso shrink all opacities uniformly.

    energy_weighted : optional.  If False, plain lasso λ·Σ_k α_k → uniform
        threshold τ = λ·lr (the convex surrogate for splat *count*).  If True,
        weight the penalty by each splat's energy E_k ∝ a_k·b_k (∫g_k), i.e.
        λ·Σ_k E_k·α_k (an opacity×area / "ink" penalty) → per-splat threshold
        τ_k = λ·lr·E_k/mean(E).  This desynchronises the collapse (softer cliff)
        and removes large diffuse splats first.

    Requires ``model.opacity`` is True (so ``_alphas`` is a learnable parameter).
    """
    assert model.opacity, "build the model with opacity=True for fit_opacity_lasso"
    model.img_size = tuple(img.shape[1:])

    optim = Adam([model._colors, model.covs, model._alphas], lr=lr)
    for i in range(n_iters):
        rendered = model.forward()
        mse = F.mse_loss(rendered, img)

        mse.backward()
        optim.step()
        optim.zero_grad()

        with torch.no_grad():
            # proximal soft-threshold of the lasso over α ≥ 0  →  exact zeros
            tau = l1 * lr
            if energy_weighted:
                energy = model.covs[:, 1].abs() * model.covs[:, 2].abs()  # ∝ ∫g_k
                tau = tau * (energy / (energy.mean() + 1e-12))            # (N,) per-splat
            model._alphas.copy_((model._alphas - tau).clamp_(min=0.0))
            model.covs[:, 1:].abs_().clamp_(min=sigma_min)
            n_active = model.n_active(active_tol)

        if verbose:
            psnr = 10.0 * torch.log10(1.0 / mse.detach().clamp(min=1e-12))
            sys.stdout.write(
                f"\r[lasso] {i + 1}/{n_iters}  PSNR {psnr.item():5.2f} dB  "
                f"active {n_active}/{model.n_candidates}"
            )
            sys.stdout.flush()
        yield rendered.detach(), mse.detach().cpu(), n_active

    if verbose:
        sys.stdout.write("\n")


# ---------------------------------------------------------------------------
# Sparsity: greedy contribution-based pruning under a quality budget
# ---------------------------------------------------------------------------

@torch.no_grad()
def _mse(model: SparseSplatImage, img: Tensor) -> float:
    return F.mse_loss(model.forward(), img).item()


def prune_and_refit(
    model: SparseSplatImage,
    img: Tensor,
    max_psnr_drop: float = 1.0,
    prune_frac: float = 0.1,
    refit_iters: int = 40,
    lr: float = 0.03,
    min_active: int = 1,
    verbose: bool = True,
) -> int:
    """Iteratively prune the lowest-contribution splats, refit, stop on budget.

    Each round: deactivate the ``prune_frac`` fraction of *active* splats with
    the smallest assignment mass, then refit the survivors with Adam.  Pruning
    stops once the PSNR has dropped ``max_psnr_drop`` dB below the pre-pruning
    baseline (the last harmful round is rolled back).  ``max_psnr_drop`` (λ) is
    the quality-vs-count knob.  Returns the final number of active splats.
    """
    model.img_size = tuple(img.shape[1:])
    base_psnr = _psnr(_mse(model, img))
    if verbose:
        print(f"[prune] baseline PSNR {base_psnr:.2f} dB  "
              f"active {model.n_active()}/{model.n_candidates}")

    while model.n_active() > min_active:
        prev_alphas = model._alphas.clone()  # snapshot for rollback
        prev_colors = model._colors.detach().clone()
        prev_covs = model.covs.detach().clone()

        mass = model.contribution()                 # (N,)
        active = model.active_mask()
        n_act = int(active.sum().item())
        k = max(1, min(int(prune_frac * n_act), n_act - min_active))

        # lowest-mass *among active* splats
        mass_active = mass.masked_fill(~active, float("inf"))
        victims = torch.topk(mass_active, k, largest=False).indices
        model.deactivate(victims)

        for _ in fit_sparse(model, img, n_iters=refit_iters, lr=lr, verbose=False):
            pass

        cur_psnr = _psnr(_mse(model, img))
        if base_psnr - cur_psnr > max_psnr_drop:
            # Rolled back: this prune overspent the budget — restore and stop.
            model._alphas.copy_(prev_alphas)
            model._colors.data.copy_(prev_colors)
            model.covs.data.copy_(prev_covs)
            if verbose:
                print(f"[prune] stop: dropping {k} more would exceed "
                      f"{max_psnr_drop:.2f} dB budget")
            break

        if verbose:
            print(f"[prune] -{k} → active {model.n_active()}/{model.n_candidates}  "
                  f"PSNR {cur_psnr:.2f} dB")

    return model.n_active()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import numpy as np

    from demo import load_image, pick_device, tensor_to_np

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=str, nargs="?", default=None,
                   help="path to input image (omit for the synthetic test pattern)")
    p.add_argument("--size", type=int, default=128,
                   help="resize input to size×size (default 128 keeps memory bounded)")
    p.add_argument("--stride", type=int, default=3,
                   help="candidate every `stride` pixels; 1 = every pixel (max splats)")
    p.add_argument("--n-iters", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--sigma-init", type=float, default=None,
                   help="initial isotropic std in [-1,1] units (default: half spacing)")
    p.add_argument("--sigma-min", type=float, default=1e-3)
    p.add_argument("--no-warm-colors", action="store_true",
                   help="start from gray colors instead of the image pixels")
    p.add_argument("--opacity", action="store_true",
                   help="sparsify via learnable opacity + L1 lasso (+background), not pruning")
    p.add_argument("--l1", type=float, default=5e-2,
                   help="lasso weight λ on the opacity (opacity mode); higher ⇒ fewer splats")
    p.add_argument("--energy-weighted", action="store_true",
                   help="weight the opacity lasso by splat energy a·b (softer cliff; opacity mode)")
    p.add_argument("--bg-weight", type=float, default=0.1,
                   help="background weight b that breaks scale-invariance (opacity mode)")
    p.add_argument("--prune", action="store_true",
                   help="sparsify via contribution-based pruning after the fit")
    p.add_argument("--max-psnr-drop", type=float, default=1.0,
                   help="λ: max PSNR (dB) you accept to lose for fewer splats")
    p.add_argument("--prune-frac", type=float, default=0.1,
                   help="fraction of active splats pruned per round")
    p.add_argument("--refit-iters", type=int, default=40,
                   help="Adam steps refitting survivors after each prune round")
    p.add_argument("--local-render", action="store_true",
                   help="use the per-Gaussian bbox PyTorch rasteriser (memory-light)")
    p.add_argument("--out", type=str, default="sparse_out.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()

    img = load_image(args.image, args.size).to(device)
    H, W = img.shape[1:]

    model = SparseSplatImage(
        img_size=(H, W),
        n_channels=img.shape[0],
        stride=args.stride,
        sigma_init=args.sigma_init,
        local_render=args.local_render,
        opacity=args.opacity,
        bg_weight=args.bg_weight,
    ).to(device)
    if not args.no_warm_colors:
        model.init_colors_from_image(img)
    if args.opacity:
        model.init_background(img)

    mode = "opacity+lasso" if args.opacity else "prune" if args.prune else "fit-only"
    print(f"[sparse] device={device}  image={H}×{W}  "
          f"candidates={model.n_candidates} (stride={args.stride})  mode={mode}")

    if args.opacity:
        for _ in fit_opacity_lasso(model, img, n_iters=args.n_iters, lr=args.lr,
                                   l1=args.l1, energy_weighted=args.energy_weighted,
                                   sigma_min=args.sigma_min):
            pass
    else:
        for _ in fit_sparse(model, img, n_iters=args.n_iters, lr=args.lr,
                            sigma_min=args.sigma_min):
            pass
        if args.prune:
            prune_and_refit(
                model, img,
                max_psnr_drop=args.max_psnr_drop,
                prune_frac=args.prune_frac,
                refit_iters=args.refit_iters,
                lr=max(args.lr * 0.5, 1e-3),
            )

    with torch.no_grad():
        rendered = model().clamp(0, 1)
    mse = torch.mean((rendered - img) ** 2).item()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    kept = model.n_active()
    frac = 100.0 * kept / model.n_candidates
    print(f"[sparse] final PSNR={psnr:.2f} dB  active splats={kept}/{model.n_candidates} "
          f"({frac:.1f}% of candidates)")

    # --- Figure: clean | rendered | active ellipses --------------------------
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(tensor_to_np(img))
    axes[0].set_title("clean")
    axes[1].imshow(tensor_to_np(rendered))
    axes[1].set_title(f"sparse render (PSNR {psnr:.1f} dB)")

    mask = model.active_mask().cpu().numpy()
    contours = model.ellipse_contours(n_sigma=2.0, n_points=48).detach().cpu().numpy()
    contours = contours[mask]
    px = (contours[..., 1] + 1.0) * 0.5 * (W - 1)
    py = (contours[..., 0] + 1.0) * 0.5 * (H - 1)
    segs = np.stack([px, py], axis=-1)
    axes[2].add_collection(LineCollection(segs, colors="black", linewidths=0.4, alpha=0.7))
    axes[2].set_xlim(-0.5, W - 0.5)
    axes[2].set_ylim(H - 0.5, -0.5)
    axes[2].set_aspect("equal")
    axes[2].set_facecolor("white")
    axes[2].set_title(f"{kept} active splats")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[sparse] saved figure → {args.out}")


if __name__ == "__main__":
    _main()
