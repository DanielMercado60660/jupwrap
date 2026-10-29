"""Synthetic interferograms with known unwrapped phase (DESIGN.md section 14).

Noise model: L independent looks of a correlated circular complex Gaussian
pair with true coherence rho: s1 = a, s2 = rho*a + sqrt(1-rho^2)*b, with a, b
~ CN(0, 1). igram = exp(i*phi) * sum_L s1*conj(s2); sample coherence
corr = |sum s1 s2*| / sqrt(sum|s1|^2 sum|s2|^2). Deterministic from the seed
(numpy default_rng, pinned by judge/uv.lock).
"""
from __future__ import annotations

import numpy as np


def _smooth_field(rng, R, C, beta, scale):
    """Gaussian random field with power-law spectrum k^-beta, unit std, times scale."""
    ky = np.fft.fftfreq(R)[:, None]
    kx = np.fft.rfftfreq(C)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    k[0, 0] = np.inf
    amp = k ** (-beta / 2.0)
    z = rng.normal(size=(R, C // 2 + 1)) + 1j * rng.normal(size=(R, C // 2 + 1))
    f = np.fft.irfft2(z * amp, s=(R, C))
    f -= f.mean()
    f /= f.std() + 1e-300
    return f * scale


def topo_phase(rng, R, C, max_grad=1.2):
    """Topography-like surface: power-law field plus a few ridges; scaled so the
    95th percentile of the true gradient magnitude is ~max_grad/2 rad/pixel and
    the maximum is about max_grad."""
    f = _smooth_field(rng, R, C, beta=3.6, scale=1.0)
    yy, xx = np.mgrid[0:R, 0:C].astype(np.float64)
    for _ in range(3):
        cy, cx = rng.uniform(0, R), rng.uniform(0, C)
        s = rng.uniform(0.08, 0.2) * min(R, C)
        f += rng.uniform(1, 3) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    gy, gx = np.gradient(f)
    g = np.sqrt(gx * gx + gy * gy)
    f *= max_grad / max(np.max(g), 1e-12)
    return f


def defo_phase(rng, R, C, peak_cycles=12.0, fault=True, fault_cycles=3.0):
    """Deformation bowl (Mogi-like), plus an optional straight fault segment with
    a slip step of fault_cycles that tapers to zero at both ends."""
    yy, xx = np.mgrid[0:R, 0:C].astype(np.float64)
    cy, cx = R * rng.uniform(0.4, 0.6), C * rng.uniform(0.4, 0.6)
    d = 0.18 * min(R, C)
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    phi = -2 * np.pi * peak_cycles * d ** 3 / (r2 + d * d) ** 1.5
    if fault:
        # fault along a line through (cy, cx) at angle th, length 0.5*min(R,C)
        th = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)       # along strike
        v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)      # across strike
        half = 0.25 * min(R, C)
        taper = np.clip(1.0 - (u / half) ** 2, 0.0, 1.0)
        phi += np.where(v > 0, 0.5, -0.5) * 2 * np.pi * fault_cycles * taper
    return phi


def coherence_map(rng, R, C, base=(0.55, 0.95), holes=3, hole_rho=0.08, layover=True):
    """True coherence: smooth field in `base`, low-coherence holes (blobs) and a
    near-zero 'layover' band. Returns (rho, hole_mask) with hole_mask True in holes."""
    f = _smooth_field(rng, R, C, beta=3.0, scale=1.0)
    f = (f - f.min()) / max(f.max() - f.min(), 1e-12)
    rho = base[0] + (base[1] - base[0]) * f
    yy, xx = np.mgrid[0:R, 0:C].astype(np.float64)
    hole = np.zeros((R, C), bool)
    for _ in range(holes):
        cy, cx = rng.uniform(0.1, 0.9) * R, rng.uniform(0.1, 0.9) * C
        ry, rx = rng.uniform(0.03, 0.08) * R, rng.uniform(0.03, 0.08) * C
        hole |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1.0
    if layover:
        c0 = rng.uniform(0.2, 0.8) * C
        w = max(2.0, 0.015 * C)
        band = (np.abs(xx - c0 - 0.15 * (yy - R / 2)) < w) & (yy > 0.2 * R) & (yy < 0.6 * R)
        hole |= band
    rho = np.where(hole, hole_rho, rho)
    return rho, hole


def noisy_igram(rng, phi, rho, nlooks):
    """L-look circular-Gaussian interferogram. Returns (igram complex64, corr float32)."""
    R, C = phi.shape
    L = int(nlooks)
    acc = np.zeros((R, C), np.complex128)
    p1 = np.zeros((R, C))
    p2 = np.zeros((R, C))
    s = np.sqrt(np.clip(1.0 - rho * rho, 0.0, 1.0))
    for _ in range(L):
        a = (rng.normal(size=(R, C)) + 1j * rng.normal(size=(R, C))) / np.sqrt(2)
        b = (rng.normal(size=(R, C)) + 1j * rng.normal(size=(R, C))) / np.sqrt(2)
        s1 = a
        s2 = rho * a + s * b
        acc += s1 * np.conj(s2)
        p1 += np.abs(s1) ** 2
        p2 += np.abs(s2) ** 2
    corr = np.abs(acc) / np.sqrt(p1 * p2)
    igram = np.exp(1j * phi) * acc
    return igram.astype(np.complex64), corr.astype(np.float32)


def make_case(kind, R, C, seed, nlooks=9, **kw):
    """Named synthetic cases. Returns dict(phi, igram, corr, rho, hole, mask)."""
    rng = np.random.default_rng(seed)
    mask = None
    if kind == "clean_topo":           # no noise, no residues -> exact recovery
        phi = topo_phase(rng, R, C, max_grad=kw.get("max_grad", 1.0))
        rho = np.full((R, C), 0.999)
        hole = np.zeros((R, C), bool)
        igram = np.exp(1j * phi).astype(np.complex64)
        corr = np.full((R, C), 0.999, np.float32)
    elif kind == "topo":
        phi = topo_phase(rng, R, C, max_grad=kw.get("max_grad", 1.2))
        rho, hole = coherence_map(rng, R, C, base=kw.get("base", (0.55, 0.95)), holes=kw.get("holes", 3))
        igram, corr = noisy_igram(rng, phi, rho, nlooks)
    elif kind == "defo":
        phi = defo_phase(rng, R, C, peak_cycles=kw.get("peak_cycles", 12.0), fault=kw.get("fault", True),
                         fault_cycles=kw.get("fault_cycles", 3.0))
        rho, hole = coherence_map(rng, R, C, base=kw.get("base", (0.6, 0.95)), holes=kw.get("holes", 2),
                                  layover=False)
        igram, corr = noisy_igram(rng, phi, rho, nlooks)
    elif kind == "masked_topo":         # water body masked out
        phi = topo_phase(rng, R, C, max_grad=kw.get("max_grad", 1.0))
        rho, hole = coherence_map(rng, R, C, holes=1)
        yy, xx = np.mgrid[0:R, 0:C]
        water = ((yy - 0.7 * R) / (0.18 * R)) ** 2 + ((xx - 0.3 * C) / (0.25 * C)) ** 2 < 1.0
        rho = np.where(water, 0.02, rho)
        hole |= water
        igram, corr = noisy_igram(rng, phi, rho, nlooks)
        mask = (~water).astype(np.uint8)
    else:
        raise ValueError(kind)
    return dict(phi=phi, igram=igram, corr=corr, rho=rho, hole=hole, mask=mask, nlooks=nlooks)
