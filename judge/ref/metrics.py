"""End-to-end metrics (DESIGN.md section 14). Regions are defined on the TRUTH
(true coherence >= rho_min, not a hole, not masked, 4-connected, size >=
min_frac * npix), independent of the unwrapper's connected components."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

TAU = 2.0 * np.pi


def truth_regions(rho, hole, mask, rho_min=0.5, min_frac=0.002):
    valid = (rho >= rho_min) & (~hole)
    if mask is not None:
        valid &= np.asarray(mask).reshape(rho.shape) != 0
    lab, n = ndimage.label(valid)  # 4-connectivity
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= max(1, int(np.ceil(min_frac * rho.size)))
    keep[0] = False
    lab = np.where(keep[lab], lab, 0)
    return lab


def fraction_correct(unw, phi, regions):
    """Per truth region, offset m = mode(round((unw-phi)/TAU)); pixel correct iff
    |unw - phi - TAU m| < pi. Returns (fraction over all region pixels, per-region list)."""
    unw = np.asarray(unw, np.float64).reshape(phi.shape)
    err = unw - phi
    cyc = np.round(err / TAU).astype(np.int64)
    tot = 0
    good = 0
    per = []
    for r in np.unique(regions[regions > 0]):
        sel = regions == r
        c = cyc[sel]
        vals, cnt = np.unique(c, return_counts=True)
        m = vals[np.argmax(cnt)]
        ok = np.abs(err[sel] - TAU * m) < np.pi
        tot += sel.sum()
        good += ok.sum()
        per.append((int(sel.sum()), float(ok.mean())))
    return (good / tot if tot else 1.0), per


def congruence_error(unw, psi, mask=None):
    """max |wrap(unw - psi)| over unmasked pixels (radians)."""
    d = np.asarray(unw, np.float64).ravel() - np.asarray(psi, np.float32).ravel().astype(np.float64)
    w = d - TAU * np.floor(d / TAU + 0.5)
    if mask is not None:
        w = w[np.asarray(mask).ravel() != 0]
    return float(np.max(np.abs(w))) if w.size else 0.0


def psi_of(igram):
    ig = np.asarray(igram)
    return np.arctan2(ig.imag.astype(np.float64), ig.real.astype(np.float64)).astype(np.float32)
