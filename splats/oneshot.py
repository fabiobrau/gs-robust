"""
gaussian_image.py -- fast, deterministic 2D Gaussian-splat image representation.

Fits   I(p)  ~=  bg + sum_i  c_i * exp( -0.5 * (p-mu_i)^T Q_i (p-mu_i) )

with signed colors c_i (additive blending: order-free, no sorting) and
truncated footprints (each splat touches O(1) pixels), so every stage is
linear in the number of splats N.

Pipeline (no "wait for convergence" anywhere):
  1. CONSTRUCT  - coarse-to-fine residual pyramid. Per level, splats have a
                  fixed window in level pixels, so peak-picking + closed-form
                  moment fits + subtraction are fully batched tensor ops.
                  One deterministic pass produces all geometry.
  2. SOLVE      - colors are linear least squares (sparse). Jacobi-
                  preconditioned CG, warm-started; each iteration is one
                  scatter + one gather over cached footprints (no exp calls).
  3. POLISH     - optional few Adam steps on geometry with analytic
                  gradients, interleaved with short CG color re-solves.

Everything is scatter/gather over per-level batches -> ports 1:1 to GPU
(torch index_add) for another 10-100x if needed.

Author: written from scratch (no reference implementation consulted).
"""

from __future__ import annotations

import time

import numpy as np

# --------------------------------------------------------------------------- #
#  small vectorized helpers
# --------------------------------------------------------------------------- #


def _pool2(a: np.ndarray) -> np.ndarray:
    """2x average pooling of an (H, W, C) array. H, W must be even."""
    h, w, c = a.shape
    return a.reshape(h // 2, 2, w // 2, 2, c).mean(axis=(1, 3))


def _grids(muy, mux, wsz, h, w):
    """Integer pixel grids of a (B, wsz, wsz) window per splat, clipped
    so every window lies fully inside the image (windows near the border
    are shifted, not shrunk -> all batches stay rectangular).
    """
    half = (wsz - 1) * 0.5
    ty = np.clip(np.rint(muy - half), 0, h - wsz).astype(np.int64)
    tx = np.clip(np.rint(mux - half), 0, w - wsz).astype(np.int64)
    r = np.arange(wsz, dtype=np.int64)
    b = muy.shape[0]
    yy = np.broadcast_to(ty[:, None, None] + r[None, :, None], (b, wsz, wsz))
    xx = np.broadcast_to(tx[:, None, None] + r[None, None, :], (b, wsz, wsz))
    return yy, xx


def _eval(mu, la, lb, ld, wsz, h, w):
    """Evaluate B Gaussians on their windows.

    Q = L L^T with lower-triangular L = [[la, 0], [lb, ld]], so the
    Mahalanobis form needs two multiplies:  q = (la*dy + lb*dx)^2 + (ld*dx)^2.

    Returns g, flat pixel index, dy, dx  -- each (B, wsz*wsz), float32/int64.
    """
    yy, xx = _grids(mu[:, 0], mu[:, 1], wsz, h, w)
    dy = (yy - mu[:, 0, None, None]).astype(np.float32)
    dx = (xx - mu[:, 1, None, None]).astype(np.float32)
    t1 = la[:, None, None] * dy + lb[:, None, None] * dx
    t2 = ld[:, None, None] * dx
    g = np.exp(-0.5 * (t1 * t1 + t2 * t2)).astype(np.float32)
    b, k2 = mu.shape[0], wsz * wsz
    idx = (yy * w + xx).reshape(b, k2)
    return g.reshape(b, k2), idx, dy.reshape(b, k2), dx.reshape(b, k2)


def _scatter_add(canvas, idx, g, colors):
    """canvas[(Npix, C)] += sum of splat contributions, via bincount."""
    npx, c = canvas.shape
    contrib = (g[:, :, None] * colors[:, None, :]).reshape(-1, c)
    fi = idx.reshape(-1)
    for ch in range(c):
        canvas[:, ch] += np.bincount(fi, weights=contrib[:, ch], minlength=npx).astype(
            np.float32
        )


def _gather(flat_img, idx, g):
    """Per-splat weighted sums  sum_p g(p) * img(p)  ->  (B, C)."""
    vals = flat_img[idx]  # (B, K2, C)
    return np.einsum("bk,bkc->bc", g, vals, optimize=True)


def _pick_peaks(energy, k, cell):
    """Up to k peaks with *guaranteed* spatial separation >= cell+1 px.

    Partitions the energy map into (cell x cell) blocks, takes each block's
    argmax, then keeps only blocks that are strict local maxima among their
    8 neighbor blocks (deterministic tiny jitter breaks plateau ties). Kept
    blocks are pairwise non-adjacent, so simultaneously fitted splats are
    nearly orthogonal -- which is what makes batched least-squares
    subtraction stable. Deterministic, O(P).
    """
    h, w = energy.shape
    hc, wc = h // cell, w // cell
    if hc == 0 or wc == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    e = energy[: hc * cell, : wc * cell]
    e = e.reshape(hc, cell, wc, cell).transpose(0, 2, 1, 3).reshape(hc, wc, -1)
    sub = e.argmax(-1)
    val = np.take_along_axis(e, sub[..., None], -1)[..., 0]
    # deterministic tie-break so flat plateaus still yield sparse maxima
    cy0, cx0 = np.meshgrid(np.arange(hc), np.arange(wc), indexing="ij")
    jit = ((cy0 * 61 + cx0 * 127) % 251).astype(np.float64) / 251.0
    v = val + jit * 1e-6 * (val.mean() + 1e-12)
    p = np.pad(v, 1, constant_values=-np.inf)
    nbmax = np.full_like(v, -np.inf)
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue
            np.maximum(nbmax, p[i : i + hc, j : j + wc], out=nbmax)
    cand = np.flatnonzero((v > nbmax) & (val > 1e-12))
    if cand.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    if cand.size > k:
        cand = cand[np.argpartition(val.reshape(-1)[cand], -k)[-k:]]
    cy, cx = cand // wc, cand % wc
    s = sub.reshape(-1)[cand]
    return cy * cell + s // cell, cx * cell + s % cell


def _clamp_cov(syy, sxy, sxx, smin, smax):
    """Clamp eigenvalues of batched 2x2 covariances into [smin^2, smax^2]
    (closed-form symmetric eigendecomposition, fully vectorized).
    """
    half_tr = 0.5 * (syy + sxx)
    df = 0.5 * (syy - sxx)
    disc = np.sqrt(df * df + sxy * sxy)
    l1 = np.clip(half_tr + disc, smin**2, smax**2)
    l2 = np.clip(half_tr - disc, smin**2, smax**2)
    # eigenvector of the larger eigenvalue
    v1y, v1x = sxy.copy(), (half_tr + disc) - syy
    deg = (np.abs(v1y) + np.abs(v1x)) < 1e-12  # already diagonal
    v1y = np.where(deg, np.where(syy >= sxx, 1.0, 0.0), v1y)
    v1x = np.where(deg, np.where(syy >= sxx, 0.0, 1.0), v1x)
    n = np.sqrt(v1y * v1y + v1x * v1x)
    v1y, v1x = v1y / n, v1x / n
    v2y, v2x = -v1x, v1y
    return (
        l1 * v1y * v1y + l2 * v2y * v2y,
        l1 * v1y * v1x + l2 * v2y * v2x,
        l1 * v1x * v1x + l2 * v2x * v2x,
    )


def _cov_to_chol_of_Q(syy, sxy, sxx):
    """Sigma (2x2, batched) -> (la, lb, ld), the Cholesky of Q = Sigma^-1."""
    det = np.maximum(syy * sxx - sxy * sxy, 1e-12)
    q00, q01, q11 = sxx / det, -sxy / det, syy / det
    la = np.sqrt(np.maximum(q00, 1e-12))
    lb = q01 / la
    ld = np.sqrt(np.maximum(q11 - lb * lb, 1e-12))
    return (la.astype(np.float32), lb.astype(np.float32), ld.astype(np.float32))


# --------------------------------------------------------------------------- #
#  main class
# --------------------------------------------------------------------------- #


class GaussianImage2D:
    """Fit an image with N additive 2D Gaussian splats, in linear time and
    (essentially) a single deterministic pass.

    Parameters
    ----------
    n_splats     : total splat budget (None -> ~1 splat per 48 pixels).
    window       : footprint size in *level* pixels (odd; 9 => sigma in
                   [0.45, 1.5] level px, truncation at tau sigma).
    tau          : truncation radius in standard deviations.
    waves        : max placement passes per pyramid level (each pass places
                   only well-separated splats and sees the residual left by
                   earlier passes; passes stop early once the budget is spent).
    cg_iters     : preconditioned-CG iterations of the exact color solve.
    refine_iters : optional Adam polish steps on geometry (0 = skip). This is
                   the main speed/quality dial: 0 is fastest, ~25 is smoother.
    cell         : peak-picking cell size in level px (None = window // 4).
                   Splats placed in one pass are >= cell+1 px apart.
    amp_damp     : under-relaxation of constructive amplitudes (stability at
                   dense placement; the exact CG solve removes the bias).
    sigma_min    : smallest splat std-dev in level px.
    verbose      : print stage timings and PSNR.

    After fit():  mu_ (N,2 in y,x), cholesky_ (N,3 = la,lb,ld of Q),
                  colors_ (N,C), level_ (N,), background_ (C,).
    All deterministic: same image -> identical representation.
    """

    def __init__(
        self,
        n_splats=None,
        window=11,
        tau=3.0,
        waves=8,
        cg_iters=12,
        refine_iters=10,
        lr_mu=0.2,
        lr_shape=0.02,
        cell=None,
        amp_damp=0.6,
        sigma_min=0.6,
        verbose=False,
    ):
        assert window % 2 == 1 and window >= 5
        self.n_splats = n_splats
        self.window = int(window)
        self.tau = float(tau)
        self.waves = int(waves)
        self.cg_iters = int(cg_iters)
        self.refine_iters = int(refine_iters)
        self.lr_mu = float(lr_mu)
        self.lr_shape = float(lr_shape)
        self.cell = cell
        self.amp_damp = float(amp_damp)
        self.sigma_min = float(sigma_min)
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------ #
    #  fitting
    # ------------------------------------------------------------------ #

    def fit(self, image):
        t0 = time.perf_counter()
        img = np.asarray(image)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = img[..., None]
        h0, w0, c = img.shape
        self.orig_shape_, self.channels_ = (h0, w0), c
        self._orig = img

        # ---- pyramid setup: pad bottom/right so 2x pooling is exact ------
        n_lev = max(1, int(np.floor(np.log2(min(h0, w0) / self.window))) + 1)
        n_lev = min(n_lev, 7)
        f = 1 << (n_lev - 1)
        hp, wp = -(-h0 // f) * f, -(-w0 // f) * f
        pad = np.pad(img, ((0, hp - h0), (0, wp - w0), (0, 0)), mode="edge")
        self.shape_, self.n_levels_ = (hp, wp), n_lev
        self.background_ = img.mean(axis=(0, 1)).astype(np.float32)
        target = pad.reshape(-1, c)
        self._target = target
        pyr = [pad]
        for _ in range(n_lev - 1):
            pyr.append(_pool2(pyr[-1]))

        # ---- splat budget per level (proportional to level pixel count) --
        n_total = self.n_splats or max(256, (h0 * w0) // 48)
        cell = self.cell or max(2, self.window // 4)
        caps = [self.waves * ((p.shape[0] // cell) * (p.shape[1] // cell)) for p in pyr]
        raw = np.array([4.0**-l for l in range(n_lev)])
        alloc = np.minimum((n_total * raw / raw.sum()).astype(int), caps)
        for l in range(n_lev):  # hand leftovers to the
            spare = n_total - alloc.sum()  # finest levels first
            if spare <= 0:
                break
            alloc[l] = min(alloc[l] + spare, caps[l])

        # ---- 1. CONSTRUCT: coarse -> fine, batched moment fits ----------
        smin, smax = self.sigma_min, self.window / (2.0 * self.tau)
        recon = np.tile(self.background_, (hp * wp, 1)).astype(np.float32)
        P = {k: [] for k in ("mu", "la", "lb", "ld", "col", "lvl")}

        for l in range(n_lev - 1, -1, -1):
            if alloc[l] == 0:
                continue
            scale = 1 << l
            hl, wl = pyr[l].shape[:2]
            rec_l = recon.reshape(hp, wp, c)
            for _ in range(l):
                rec_l = _pool2(rec_l)
            res = (pyr[l] - rec_l).copy()  # level residual
            res_flat = res.reshape(-1, c)
            got = {k: [] for k in ("mu", "la", "lb", "ld", "col")}
            placed = 0

            for _ in range(self.waves):
                if placed >= alloc[l]:
                    break
                e = (res * res).sum(-1)
                py, px = _pick_peaks(e, int(alloc[l] - placed), cell)
                if py.size == 0:
                    break
                # windows around integer peaks
                yy, xx = _grids(
                    py.astype(np.float32), px.astype(np.float32), self.window, hl, wl
                )
                widx = (yy * wl + xx).reshape(py.size, -1)
                win = res_flat[widx]  # (B, K2, C)
                peak = res_flat[py * wl + px]  # (B, C)
                u = peak / (np.linalg.norm(peak, axis=1, keepdims=True) + 1e-12)
                wgt = np.einsum("bkc,bc->bk", win, u)
                wgt = np.clip(wgt, 0, None) + 1e-8  # moment weights
                m = wgt.sum(1)
                yf = yy.reshape(py.size, -1).astype(np.float32)
                xf = xx.reshape(py.size, -1).astype(np.float32)
                muy = (wgt * yf).sum(1) / m
                mux = (wgt * xf).sum(1) / m
                dy, dx = yf - muy[:, None], xf - mux[:, None]
                syy = (wgt * dy * dy).sum(1) / m
                sxy = (wgt * dy * dx).sum(1) / m
                sxx = (wgt * dx * dx).sum(1) / m
                syy, sxy, sxx = _clamp_cov(syy, sxy, sxx, smin, smax)
                la, lb, ld = _cov_to_chol_of_Q(syy, sxy, sxx)
                mu = np.stack([muy, mux], 1).astype(np.float32)
                # amplitude: per-splat 1D least squares on its window
                g, idx, _, _ = _eval(mu, la, lb, ld, self.window, hl, wl)
                num = _gather(res_flat, idx, g)
                den = (g * g).sum(1) + 1e-12
                col = (self.amp_damp * num / den[:, None]).astype(np.float32)
                _scatter_add(res_flat, idx, g, -col)  # subtract
                for k, v in zip(("mu", "la", "lb", "ld", "col"), (mu, la, lb, ld, col)):
                    got[k].append(v)
                placed += py.size

            if l > 0 and placed < alloc[l]:
                alloc[l - 1] += alloc[l] - placed  # unspent budget -> finer
            if not got["mu"]:
                continue
            mu = np.concatenate(got["mu"])
            la = np.concatenate(got["la"])
            lb = np.concatenate(got["lb"])
            ld = np.concatenate(got["ld"])
            col = np.concatenate(got["col"])
            # level coords -> full-res coords (pixel centers align)
            mu_f = (mu + 0.5) * scale - 0.5
            la_f, lb_f, ld_f = la / scale, lb / scale, ld / scale
            wf = min(self.window * scale, min(hp, wp))
            gf, idxf, _, _ = _eval(mu_f, la_f, lb_f, ld_f, wf, hp, wp)
            _scatter_add(recon, idxf, gf, col)  # exact render
            P["mu"].append(mu_f)
            P["la"].append(la_f)
            P["lb"].append(lb_f)
            P["ld"].append(ld_f)
            P["col"].append(col)
            P["lvl"].append(np.full(len(mu_f), l, np.int32))
            if self.verbose:
                print(
                    f"  level {l}: +{len(mu_f)} splats "
                    f"({hl}x{wl}, window {wf} px at full res)"
                )

        self.mu_ = np.concatenate(P["mu"])
        self._la = np.concatenate(P["la"])
        self._lb = np.concatenate(P["lb"])
        self._ld = np.concatenate(P["ld"])
        self.colors_ = np.concatenate(P["col"])
        self.level_ = np.concatenate(P["lvl"])
        order = np.argsort(self.level_, kind="stable")  # group by level
        for a in ("mu_", "_la", "_lb", "_ld", "colors_", "level_"):
            setattr(self, a, getattr(self, a)[order])
        t1 = time.perf_counter()
        if self.verbose:
            print(
                f"construct: {len(self.mu_)} splats   "
                f"{t1 - t0:.2f}s   PSNR {self._psnr_from(recon):.2f} dB"
            )

        # ---- 2. SOLVE: exact colors by preconditioned CG ------------------
        self._build_cache()
        self._cg_colors(self.cg_iters)
        t2 = time.perf_counter()
        if self.verbose:
            print(
                f"color CG : {t2 - t1:.2f}s   "
                f"PSNR {self._psnr_from(self._render_cached()):.2f} dB"
            )

        # ---- 3. POLISH: optional Adam steps on geometry -------------------
        if self.refine_iters > 0:
            self._refine(self.refine_iters)
            self._build_cache()
            self._cg_colors(max(4, self.cg_iters // 2))
        t3 = time.perf_counter()
        if self.verbose and self.refine_iters > 0:
            print(
                f"polish   : {t3 - t2:.2f}s   "
                f"PSNR {self._psnr_from(self._render_cached()):.2f} dB"
            )
        if self.verbose:
            n = len(self.mu_)
            floats = n * (5 + self.channels_) + self.channels_
            print(
                f"total    : {t3 - t0:.2f}s   {n} splats = {floats} floats "
                f"({h0 * w0 * c / floats:.1f}x fewer than pixels)"
            )
        return self

    # ------------------------------------------------------------------ #
    #  footprint cache  (geometry-dependent; colors are not baked in)
    # ------------------------------------------------------------------ #

    def _level_slices(self):
        hp, wp = self.shape_
        out = []
        for l in np.unique(self.level_):
            i = np.flatnonzero(self.level_ == l)
            sl = slice(i[0], i[-1] + 1)
            wf = min(self.window << int(l), min(hp, wp))
            out.append((sl, wf))
        return out

    def _build_cache(self):
        hp, wp = self.shape_
        self._cache = []
        diag = np.empty(len(self.mu_), np.float32)
        for sl, wf in self._level_slices():
            g, idx, dy, dx = _eval(
                self.mu_[sl], self._la[sl], self._lb[sl], self._ld[sl], wf, hp, wp
            )
            self._cache.append((sl, g, idx, dy, dx))
            diag[sl] = (g * g).sum(1) + 1e-8
        self._diag = diag

    def _render_cached(self, colors=None):
        colors = self.colors_ if colors is None else colors
        canvas = np.tile(self.background_, (self.shape_[0] * self.shape_[1], 1)).astype(
            np.float32
        )
        for sl, g, idx, _, _ in self._cache:
            _scatter_add(canvas, idx, g, colors[sl])
        return canvas

    def _gather_cached(self, flat_img):
        out = np.empty((len(self.mu_), self.channels_), np.float32)
        for sl, g, idx, _, _ in self._cache:
            out[sl] = _gather(flat_img, idx, g)
        return out

    # ------------------------------------------------------------------ #
    #  exact color solve:   min_c || Phi c - (target - bg) ||^2
    #  A = Phi^T Phi applied as gather(render(.)); Jacobi preconditioner.
    # ------------------------------------------------------------------ #

    def _cg_colors(self, iters):
        bgless = self._target - self.background_[None, :]

        def A(v):
            canvas = np.zeros_like(self._target)
            for sl, g, idx, _, _ in self._cache:
                _scatter_add(canvas, idx, g, v[sl])
            return self._gather_cached(canvas)

        x = self.colors_
        b = self._gather_cached(bgless)
        r = b - A(x)
        z = r / self._diag[:, None]
        p = z.copy()
        rz = np.einsum("nc,nc->c", r, z, dtype=np.float64)
        for _ in range(iters):
            ap = A(p)
            denom = np.einsum("nc,nc->c", p, ap, dtype=np.float64)
            alpha = (rz / np.maximum(denom, 1e-30)).astype(np.float32)
            x += alpha * p
            r -= alpha * ap
            z = r / self._diag[:, None]
            rz_new = np.einsum("nc,nc->c", r, z, dtype=np.float64)
            beta = (rz_new / np.maximum(rz, 1e-30)).astype(np.float32)
            p = z + beta * p
            rz = rz_new
        self.colors_ = x

    # ------------------------------------------------------------------ #
    #  geometry polish: analytic gradients + Adam (colors fixed,
    #  re-solved by short CG every few steps)
    # ------------------------------------------------------------------ #

    def _geometry_grads(self, res_flat):
        n = len(self.mu_)
        gmu = np.zeros((n, 2), np.float32)
        ga = np.zeros(n, np.float32)
        gb = np.zeros(n, np.float32)
        gd = np.zeros(n, np.float32)
        for sl, g, idx, dy, dx in self._cache:
            rwin = res_flat[idx]  # (B,K2,C)
            rho = np.einsum("bkc,bc->bk", rwin, self.colors_[sl])
            la, lb, ld = self._la[sl], self._lb[sl], self._ld[sl]
            q00, q01 = la * la, la * lb
            q11 = lb * lb + ld * ld
            common = 2.0 * rho * g
            gmu[sl, 0] = (common * (q00[:, None] * dy + q01[:, None] * dx)).sum(1)
            gmu[sl, 1] = (common * (q01[:, None] * dy + q11[:, None] * dx)).sum(1)
            rg = rho * g
            s00 = -(rg * dy * dy).sum(1)
            s01 = -(rg * dy * dx).sum(1)
            s11 = -(rg * dx * dx).sum(1)
            # Q = L L^T  =>  dL = 2 S L ; chain a = log la, d = log ld
            ga[sl] = 2.0 * (s00 * la + s01 * lb) * la
            gb[sl] = 2.0 * (s01 * la + s11 * lb)
            gd[sl] = 2.0 * s11 * ld * ld
        return gmu, ga, gb, gd

    def _refine(self, iters):
        hp, wp = self.shape_
        a = np.log(self._la).astype(np.float32)
        d = np.log(self._ld).astype(np.float32)
        b = self._lb.astype(np.float32)
        scl = (2.0**self.level_).astype(np.float32)  # per splat
        a_min = np.log(
            2.0 * self.tau / np.minimum(self.window * scl, min(hp, wp))
        ).astype(np.float32)
        a_max = np.float32(np.log(1.0 / max(0.3, 0.7 * self.sigma_min)))
        params = {"mu": self.mu_, "a": a, "b": b, "d": d}
        mom = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in params.items()}
        b1, b2, eps = 0.9, 0.99, 1e-8
        for t in range(1, iters + 1):
            self._la, self._ld, self._lb = np.exp(a), np.exp(d), b
            self._build_cache()
            res = self._render_cached() - self._target
            grads = dict(zip(("mu", "a", "b", "d"), self._geometry_grads(res)))
            for k in params:
                m, v = mom[k]
                gk = grads[k]
                m[:] = b1 * m + (1 - b1) * gk
                v[:] = b2 * v + (1 - b2) * gk * gk
                step = (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)
                if k == "mu":
                    params[k] -= (self.lr_mu * scl)[:, None] * step
                else:
                    params[k] -= self.lr_shape * step
            np.clip(a, a_min, a_max, out=a)
            np.clip(d, a_min, a_max, out=d)
            np.clip(b, -3.3, 3.3, out=b)
            np.clip(self.mu_[:, 0], 0, hp - 1, out=self.mu_[:, 0])
            np.clip(self.mu_[:, 1], 0, wp - 1, out=self.mu_[:, 1])
            if t % 3 == 0:  # keep colors matched
                self._build_cache()
                self._cg_colors(2)
        self._la, self._ld, self._lb = np.exp(a), np.exp(d), b

    # ------------------------------------------------------------------ #
    #  outputs
    # ------------------------------------------------------------------ #

    def render(self, scale=1.0, clip=True):
        """Render the representation; `scale` > 1 exploits the continuous
        nature of the splats (e.g. scale=2 renders at twice the resolution).
        """
        h0, w0 = self.orig_shape_
        if scale == 1.0:
            out = self._render_cached().reshape(*self.shape_, self.channels_)
            out = out[:h0, :w0]
        else:
            hs, ws = int(round(h0 * scale)), int(round(w0 * scale))
            mu = (self.mu_ + 0.5) * scale - 0.5
            la, lb, ld = (x / scale for x in (self._la, self._lb, self._ld))
            canvas = np.tile(self.background_, (hs * ws, 1)).astype(np.float32)
            for sl, wf in self._level_slices():
                wsz = min(int(round(wf * scale)), min(hs, ws))
                g, idx, _, _ = _eval(mu[sl], la[sl], lb[sl], ld[sl], wsz, hs, ws)
                _scatter_add(canvas, idx, g, self.colors_[sl])
            out = canvas.reshape(hs, ws, self.channels_)
        if clip:
            out = np.clip(out, 0.0, 1.0)
        return out[..., 0] if self.channels_ == 1 else out

    def psnr(self, image=None):
        ref = self._orig if image is None else np.asarray(image, np.float32)
        if ref.dtype == np.uint8:
            ref = ref.astype(np.float32) / 255.0
        if ref.ndim == 2:
            ref = ref[..., None]
        rec = self.render(clip=True)
        rec = rec[..., None] if rec.ndim == 2 else rec
        mse = float(np.mean((rec - ref) ** 2))
        return -10.0 * np.log10(max(mse, 1e-12))

    def _psnr_from(self, recon_flat):
        h0, w0 = self.orig_shape_
        rec = recon_flat.reshape(*self.shape_, self.channels_)[:h0, :w0]
        mse = float(np.mean((np.clip(rec, 0, 1) - self._orig) ** 2))
        return -10.0 * np.log10(max(mse, 1e-12))

    def state_dict(self):
        """Everything needed to re-render: ~(5+C) floats per splat."""
        return {
            "mu": self.mu_.copy(),
            "cholesky_Q": np.stack([self._la, self._lb, self._ld], 1),
            "colors": self.colors_.copy(),
            "level": self.level_.copy(),
            "background": self.background_.copy(),
            "shape": self.orig_shape_,
            "window": self.window,
        }

    @property
    def cholesky_(self):
        return np.stack([self._la, self._lb, self._ld], 1)

    def covariances(self):
        """Per-splat 2x2 covariance matrices Sigma = (L L^T)^-1, (N,2,2)."""
        la, lb, ld = self._la, self._lb, self._ld
        q00, q01, q11 = la * la, la * lb, lb * lb + ld * ld
        det = q00 * q11 - q01 * q01
        cov = np.empty((len(la), 2, 2), np.float32)
        cov[:, 0, 0] = q11 / det
        cov[:, 0, 1] = cov[:, 1, 0] = -q01 / det
        cov[:, 1, 1] = q00 / det
        return cov


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # tiny self-test on a synthetic image
    yy, xx = np.mgrid[0:200, 0:300].astype(np.float32)
    img = np.stack([xx / 300, yy / 200, 0.5 + 0.3 * np.sin(xx / 12)], -1)
    img[60:140, 90:210] = [0.9, 0.2, 0.1]
    gs = GaussianImage2D(n_splats=1200, verbose=True).fit(img)
    print(f"final PSNR: {gs.psnr():.2f} dB, {len(gs.mu_)} splats")
