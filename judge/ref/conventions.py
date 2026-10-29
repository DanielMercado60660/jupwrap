"""Reference conventions (mirror of crates/core/src/lib.rs). DESIGN.md section 3.

Independent NumPy implementation used to generate goldens and to score
end-to-end runs. All arithmetic on phases is float64 on float32-cast inputs.
"""
from __future__ import annotations

import numpy as np

TAU = 2.0 * np.pi
COST_SCALE = 100.0
COST_CAP = 1_000_000_000
KT = 4
MIN_TAIL_SLOPE = 100
K_BIG = 1 << 20
RHO_MAX = 0.999
SIGMA2_SIGNAL_SMOOTH = 0.02
SIGMA2_SIGNAL_DEFO = 0.02
P_DISCONTINUITY_DEFO = 1e-4
CONNCOMP_COST_THRESH = 300
SIGMA2_MAX = np.pi * np.pi / 3.0


def wrap_count(d):
    return np.floor(np.asarray(d, dtype=np.float64) / TAU + 0.5)


def wrap(d):
    d = np.asarray(d, dtype=np.float64)
    return d - TAU * np.floor(d / TAU + 0.5)


class Dims:
    def __init__(self, nrow: int, ncol: int):
        self.nrow, self.ncol = int(nrow), int(ncol)

    @property
    def npix(self):
        return self.nrow * self.ncol

    @property
    def nv(self):
        return max(self.nrow - 1, 0) * self.ncol

    @property
    def nh(self):
        return self.nrow * max(self.ncol - 1, 0)

    @property
    def n_edges(self):
        return self.nv + self.nh

    @property
    def n_loops(self):
        return max(self.nrow - 1, 0) * max(self.ncol - 1, 0)

    @property
    def ground(self):
        return self.n_loops

    @property
    def n_dual(self):
        return self.n_loops + 1

    def endpoints(self):
        """(p, q) pixel index arrays for all edges, edge order."""
        R, C = self.nrow, self.ncol
        idx = np.arange(R * C).reshape(R, C)
        pv, qv = idx[:-1, :].ravel(), idx[1:, :].ravel()
        ph, qh = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        return np.concatenate([pv, ph]), np.concatenate([qv, qh])

    def dual_arcs(self):
        """(from, to) dual node arrays for all edges, edge order."""
        R, C = self.nrow, self.ncol
        g = self.ground
        W = C - 1
        # vertical edges (r, c), r < R-1: from loop(r, c-1) | ground, to loop(r, c) | ground
        rr, cc = np.meshgrid(np.arange(R - 1), np.arange(C), indexing="ij")
        fv = np.where(cc == 0, g, rr * W + (cc - 1))
        tv = np.where(cc == C - 1, g, rr * W + cc)
        # horizontal edges (r, c), c < C-1: from loop(r, c) | ground(r==R-1), to loop(r-1, c) | ground(r==0)
        rr, cc = np.meshgrid(np.arange(R), np.arange(C - 1), indexing="ij")
        fh = np.where(rr == R - 1, g, rr * W + cc)
        th = np.where(rr == 0, g, (rr - 1) * W + cc)
        return (np.concatenate([fv.ravel(), fh.ravel()]).astype(np.int64),
                np.concatenate([tv.ravel(), th.ravel()]).astype(np.int64))


def diffs(dims: Dims, psi):
    p, q = dims.endpoints()
    psi = np.asarray(psi, dtype=np.float32).ravel().astype(np.float64)
    return psi[q] - psi[p]


def wrapped_gradients(dims, psi):
    return wrap(diffs(dims, psi))


def wrap_counts(dims, psi):
    return wrap_count(diffs(dims, psi)).astype(np.int64)


def divergence(dims: Dims, k):
    """out - in of flow k at every dual node."""
    f, t = dims.dual_arcs()
    k = np.asarray(k, dtype=np.int64)
    d = np.zeros(dims.n_dual, dtype=np.int64)
    np.add.at(d, f, k)
    np.add.at(d, t, -k)
    return d


def residues(dims: Dims, psi):
    """residue(loop) = -(signed sum of wrap counts) = -divergence(c) on loops."""
    c = wrap_counts(dims, psi)
    return (-divergence(dims, c)[: dims.n_loops]).astype(np.int8)


def supply(dims: Dims, res):
    res = np.asarray(res, dtype=np.int64)
    return np.concatenate([-res, [res.sum()]]).astype(np.int64)


def k_from_n(dims: Dims, psi, n):
    p, q = dims.endpoints()
    n = np.asarray(n, dtype=np.int64).ravel()
    return (wrap_counts(dims, psi) + n[q] - n[p]).astype(np.int64)


def integrate(dims: Dims, psi, k):
    """n with n[0] = 0 and n_q - n_p = k_e - c_e. Raises ValueError if not curl-free."""
    k = np.asarray(k, dtype=np.int64)
    c = wrap_counts(dims, psi)
    res = residues(dims, psi)
    if not np.array_equal(divergence(dims, k), supply(dims, res)):
        raise ValueError("not curl-free")
    R, C = dims.nrow, dims.ncol
    g = k - c  # n_q - n_p
    gv = g[: dims.nv].reshape(R - 1, C) if R > 1 else np.zeros((0, C), np.int64)
    gh = g[dims.nv:].reshape(R, C - 1) if C > 1 else np.zeros((R, 0), np.int64)
    n = np.zeros((R, C), dtype=np.int64)
    # first column via vertical edges, then each row via horizontal edges
    if R > 1:
        n[1:, 0] = np.cumsum(gv[:, 0])
    if C > 1:
        n[:, 1:] = n[:, :1] + np.cumsum(gh, axis=1)
    return n.ravel()


def table_cost(table, k):
    """GridCosts::cost for every edge (table shape [E, 2kt+1]) at flows k, with the linear extension."""
    table = np.asarray(table, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    kt = (table.shape[1] - 1) // 2
    E = table.shape[0]
    rows = np.arange(E)
    kc = np.clip(k, -kt, kt)
    base = table[rows, kc + kt]
    out = base.copy()
    hi = k > kt
    if hi.any():
        slope = np.maximum(table[rows, 2 * kt] - table[rows, 2 * kt - 1], MIN_TAIL_SLOPE) if kt > 0 else np.full(E, MIN_TAIL_SLOPE)
        out[hi] = table[hi, 2 * kt] + slope[hi] * (k[hi] - kt)
    lo = k < -kt
    if lo.any():
        slope = np.maximum(table[rows, 0] - table[rows, 1], MIN_TAIL_SLOPE) if kt > 0 else np.full(E, MIN_TAIL_SLOPE)
        out[lo] = table[lo, 0] + slope[lo] * (-kt - k[lo])
    return out


def total_cost(table, k):
    return int(table_cost(table, k).sum())


def min_component_size(frac, npix):
    s = np.ceil(frac * npix)
    return int(s) if np.isfinite(s) and s >= 1 else 1
