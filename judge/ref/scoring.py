"""Scoring helpers shared by the e2e judges and gen_e2e.py."""
from __future__ import annotations

import numpy as np

from .conventions import Dims, k_from_n, table_cost
from .costs import build_costs
from .metrics import psi_of

TAU = 2.0 * np.pi


def ref_table(cs, window=(7, 7)):
    R, C = cs["igram"].shape
    d = Dims(R, C)
    psi = psi_of(cs["igram"])
    corr = np.nan_to_num(np.asarray(cs["corr"], np.float32), nan=0.0).clip(0, 1)
    mask = None if cs["mask"] is None else cs["mask"].ravel()
    return d, psi, build_costs(d, psi, corr.ravel(), mask, cs["cost"], cs["nlooks"], window)


def cost_of_unw(d, psi, table, unw, mask=None):
    """Cost under the reference table of the cycle field implied by unw,
    summed over edges whose endpoints are both valid."""
    n = np.round((np.asarray(unw, np.float64).ravel() - psi.ravel().astype(np.float64)) / TAU).astype(np.int64)
    k = k_from_n(d, psi, n)
    c = table_cost(table, k)
    if mask is not None:
        p, q = d.endpoints()
        m = np.asarray(mask).ravel() != 0
        c = c[m[p] & m[q]]
    return int(c.sum())
