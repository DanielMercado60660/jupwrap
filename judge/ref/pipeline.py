"""Reference single-tile pipeline used only to calibrate end-to-end thresholds.
Exact convex optimum (OR-tools) on smooth costs or on the defo convex
envelope, then greedy single-pixel nonconvex moves (defo)."""
from __future__ import annotations

import numpy as np

from .conventions import CONNCOMP_COST_THRESH, KT, TAU, Dims, integrate, min_component_size, residues
from .costs import build_costs, convex_envelope
from .flows import grid_mcf, label_components, local_search_pixels, mst_init
from .metrics import psi_of


def ref_unwrap(igram, corr, nlooks, mode="smooth", mask=None, window=(7, 7), init="mcf", min_conncomp_frac=0.01):
    R, C = igram.shape
    d = Dims(R, C)
    psi = psi_of(igram)
    corr = np.nan_to_num(np.asarray(corr, np.float32), nan=0.0).clip(0, 1)
    table = build_costs(d, psi, corr.ravel(), None if mask is None else mask.ravel(), mode, nlooks, window)
    res = residues(d, psi)
    tin = table if mode == "smooth" else convex_envelope(table)
    if init == "mcf":
        k, _ = grid_mcf(d, tin, res, B=KT)
    else:
        k, _, _ = mst_init(d, table, res)
    if mode == "defo" or init == "mst":
        k = local_search_pixels(d, table, k)
    n = integrate(d, psi, k)
    unw = (psi.ravel().astype(np.float64) + TAU * n).astype(np.float32)
    lab, nc = label_components(d, table, k, None if mask is None else mask.ravel(), CONNCOMP_COST_THRESH,
                               min_component_size(min_conncomp_frac, d.npix))
    if mask is not None:
        unw[mask.ravel() == 0] = 0
    return unw.reshape(R, C), lab.reshape(R, C), psi, table, k
