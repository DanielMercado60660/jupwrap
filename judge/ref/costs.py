"""Reference statistical cost model, DESIGN.md section 5 (eqs. C1-C8).

Clean-room reconstruction of the MAP cost idea of Chen and Zebker (2001):
cost = -log pdf of the unwrapped phase gradient given the wrapped data and
coherence. The exact SNAPHU cost shapes are NOT reproduced (papers paywalled;
see DESIGN.md "Unverified reconstruction").
"""
from __future__ import annotations

import numpy as np

from .conventions import (COST_CAP, COST_SCALE, KT, RHO_MAX, SIGMA2_MAX, TAU,
                          Dims, wrapped_gradients)


def effective_coherence(corr, nlooks):
    """C1: NaN->0, clip [0,1]; if L > 1: rho^2 = max(0, (L r^2 - 1)/(L - 1)); cap RHO_MAX."""
    r = np.asarray(corr, dtype=np.float32).astype(np.float64)
    r = np.where(np.isnan(r), 0.0, r)
    r = np.clip(r, 0.0, 1.0)
    L = float(nlooks)
    if L > 1.0:
        r = np.sqrt(np.maximum(0.0, (L * r * r - 1.0) / (L - 1.0)))
    return np.minimum(r, RHO_MAX)


def pixel_sigma2(corr, nlooks):
    """C2: Cramer-Rao phase variance (1-rho^2)/(2 L rho^2), capped at pi^2/3; rho==0 -> cap."""
    rho = effective_coherence(corr, nlooks)
    L = float(nlooks)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (1.0 - rho * rho) / (2.0 * L * rho * rho)
    s = np.where(rho > 0.0, np.minimum(s, SIGMA2_MAX), SIGMA2_MAX)
    return s


def _box_sum(a, hr, hc):
    """Sum of a over the clipped window [r-hr, r+hr] x [c-hc, c+hc]."""
    R, C = a.shape
    if R == 0 or C == 0:
        return a.copy()
    s = np.zeros((R + 1, C + 1))
    s[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    r0 = np.clip(np.arange(R) - hr, 0, R)
    r1 = np.clip(np.arange(R) + hr + 1, 0, R)
    c0 = np.clip(np.arange(C) - hc, 0, C)
    c1 = np.clip(np.arange(C) + hc + 1, 0, C)
    return (s[r1][:, c1] - s[r0][:, c1] - s[r1][:, c0] + s[r0][:, c0])


def _box_sum_direct(a, hr, hc):
    R, C = a.shape
    out = np.zeros_like(a)
    for r in range(R):
        for c in range(C):
            out[r, c] = a[max(r - hr, 0): r + hr + 1, max(c - hc, 0): c + hc + 1].sum()
    return out


def mean_gradients(dims: Dims, psi, mask, window, direct=False, return_resultant=False):
    """C3: per-edge circular mean of wrapped gradients of the same direction over
    the clipped window of half-widths (wr//2, wc//2) centred on the edge; only
    edges with both endpoints valid contribute; mu = atan2(S, C), and mu = 0 when
    no valid edge lies in the window. With return_resultant, also returns
    |(S, C)| / count (0 when count == 0) for conditioning checks."""
    R, Cn = dims.nrow, dims.ncol
    D = wrapped_gradients(dims, psi)
    p, q = dims.endpoints()
    m = np.ones(dims.npix, bool) if mask is None else (np.asarray(mask).ravel() != 0)
    valid = (m[p] & m[q]).astype(np.float64)
    hr, hc = window[0] // 2, window[1] // 2
    box = _box_sum_direct if direct else _box_sum
    out, res = [], []
    for sl, shape in ((slice(0, dims.nv), (R - 1, Cn)), (slice(dims.nv, dims.n_edges), (R, Cn - 1))):
        if shape[0] <= 0 or shape[1] <= 0:
            continue
        d = D[sl].reshape(shape)
        v = valid[sl].reshape(shape)
        S = box(np.sin(d) * v, hr, hc)
        Cc = box(np.cos(d) * v, hr, hc)
        cnt = np.rint(box(v, hr, hc))
        mu = np.where(cnt > 0, np.arctan2(S, Cc), 0.0)
        out.append(mu.ravel())
        res.append(np.where(cnt > 0, np.hypot(S, Cc) / np.maximum(cnt, 1), 0.0).ravel())
    mu = np.concatenate(out) if out else np.zeros(0)
    if return_resultant:
        return mu, (np.concatenate(res) if res else np.zeros(0))
    return mu


def float_costs(dims: Dims, psi, corr, mask, mode, nlooks, window, sigma2_signal, p_disc, kt=KT, mu=None):
    """C4-C6: float cost c_e(k) for k in [-kt, kt], shape [E, 2kt+1]; plus validity."""
    p, q = dims.endpoints()
    s2pix = pixel_sigma2(np.asarray(corr).ravel(), nlooks)
    s2 = s2pix[p] + s2pix[q] + sigma2_signal
    D = wrapped_gradients(dims, psi)
    if mu is None:
        mu = mean_gradients(dims, psi, mask, window)
    ks = np.arange(-kt, kt + 1, dtype=np.float64)
    d = D[:, None] + TAU * ks[None, :] - mu[:, None]
    if mode == "smooth":
        c = d * d / (2.0 * s2[:, None])
    elif mode == "defo":
        a = np.log1p(-p_disc) - d * d / (2.0 * s2[:, None]) - 0.5 * np.log(TAU * s2[:, None])
        b = np.log(p_disc) - np.log(TAU * (2 * kt + 1))
        hi = np.maximum(a, b)
        lo = np.minimum(a, b)
        c = -(hi + np.log1p(np.exp(lo - hi)))
    else:
        raise ValueError(mode)
    m = np.ones(dims.npix, bool) if mask is None else (np.asarray(mask).ravel() != 0)
    valid = m[p] & m[q]
    return c, valid


def quantize(c, valid):
    """C7: floor(COST_SCALE*(c - min_k c) + 0.5), capped at COST_CAP; invalid edges -> 0."""
    t = np.floor(COST_SCALE * (c - c.min(axis=1, keepdims=True)) + 0.5)
    t = np.minimum(t, COST_CAP)
    t[~valid] = 0
    return t.astype(np.int64)


def build_costs(dims, psi, corr, mask, mode, nlooks, window=(7, 7), sigma2_signal=None, p_disc=None, kt=KT):
    from .conventions import P_DISCONTINUITY_DEFO, SIGMA2_SIGNAL_DEFO, SIGMA2_SIGNAL_SMOOTH
    if sigma2_signal is None:
        sigma2_signal = SIGMA2_SIGNAL_SMOOTH if mode == "smooth" else SIGMA2_SIGNAL_DEFO
    if p_disc is None:
        p_disc = P_DISCONTINUITY_DEFO
    c, valid = float_costs(dims, psi, corr, mask, mode, nlooks, window, sigma2_signal, p_disc, kt)
    return quantize(c, valid)


def _floor_div(a, b):
    return a // b  # python ints: floor division


def envelope_row(v):
    """C8: integer convex surrogate of one table row (list of ints)."""
    v = [int(x) for x in v]
    m = len(v)
    # lower hull (monotone chain), strictly convex vertices
    hull = []
    for i in range(m):
        while len(hull) >= 2:
            (i1, i2) = hull[-2], hull[-1]
            # cross of (i2-i1, v2-v1) x (i-i1, vi-v1) <= 0 -> pop (keeps lower hull)
            cross = (i2 - i1) * (v[i] - v[i1]) - (v[i2] - v[i1]) * (i - i1)
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    inc = [0] * (m - 1)  # increment for unit segment [i, i+1]
    for a, b in zip(hull[:-1], hull[1:]):
        s = _floor_div(v[b] - v[a], b - a)
        for i in range(a, b):
            inc[i] = s
    istar = min(range(m), key=lambda i: (v[i], i))
    e = [0] * m
    e[istar] = v[istar]
    for i in range(istar, m - 1):
        e[i + 1] = e[i] + inc[i]
    for i in range(istar, 0, -1):
        e[i - 1] = e[i] - inc[i - 1]
    return e


def convex_envelope(table):
    table = np.asarray(table, dtype=np.int64)
    return np.array([envelope_row(r) for r in table], dtype=np.int64).reshape(table.shape)


def is_convex_rows(table):
    t = np.asarray(table, dtype=np.int64)
    d = np.diff(t, axis=1)
    return bool((np.diff(d, axis=1) >= 0).all())
