"""Deterministic instance recipes shared bit-for-bit with crates/testkit
(splitmix64). Used for large judge instances that are too big to store.

Recipe "quadgrid"(nrow, ncol, seed, B, resid_permille):
  draws are taken in this order from splitmix64 seeded with `seed`:
  1. per loop (loop order): r; residue = 0 if r % 1000 >= resid_permille,
     else (+1 if (r >> 32) & 1 else -1)
  2. per edge (edge order): r1, r2; a = 1 + r1 % 2000;
     u = -1 if r2 % 10 == 0 else (1 if r2 % 10 == 1 else 0)
  table row for k in [-B, B]: a * (k - u)^2  (int64)
"""
from __future__ import annotations

import numpy as np

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
M1 = np.uint64(0xBF58476D1CE4E5B9)
M2 = np.uint64(0x94D049BB133111EB)


def splitmix(seed: int, start: int, count: int):
    """Outputs number start .. start+count-1 of splitmix64(seed) as uint64."""
    with np.errstate(over="ignore"):
        i = np.arange(start + 1, start + count + 1, dtype=np.uint64)
        z = np.uint64(seed) + i * GOLDEN
        z = (z ^ (z >> np.uint64(30))) * M1
        z = (z ^ (z >> np.uint64(27))) * M2
        z = z ^ (z >> np.uint64(31))
    return z


def quadgrid(nrow, ncol, seed, B, resid_permille):
    from .conventions import Dims
    d = Dims(nrow, ncol)
    L = d.n_loops
    E = d.n_edges
    r = splitmix(seed, 0, L)
    res = np.where(r % np.uint64(1000) >= np.uint64(resid_permille), 0,
                   np.where((r >> np.uint64(32)) & np.uint64(1), 1, -1)).astype(np.int64)
    rr = splitmix(seed, L, 2 * E)
    r1, r2 = rr[0::2], rr[1::2]
    a = (np.uint64(1) + r1 % np.uint64(2000)).astype(np.int64)
    m = (r2 % np.uint64(10)).astype(np.int64)
    u = np.where(m == 0, -1, np.where(m == 1, 1, 0))
    ks = np.arange(-B, B + 1)
    table = a[:, None] * (ks[None, :] - u[:, None]) ** 2
    return d, res, table
