"""Goldens for jupwrap-grid (residues, dual network, integration, k_from_n).
Reference: judge/ref/conventions.py. Run: .venv/bin/python -m gen.gen_grid"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ref.conventions import TAU, Dims, divergence, integrate, k_from_n, residues, supply, wrap

OUT = Path(__file__).resolve().parent.parent / "golden" / "grid"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rng = np.random.default_rng(7)
    shapes = [(1, 1), (1, 7), (6, 1), (2, 2), (3, 4), (7, 9), (64, 80), (200, 150), (513, 257)]
    for i, (R, C) in enumerate(shapes):
        d = Dims(R, C)
        phi = np.cumsum(rng.normal(0, 1.8, (R, C)), axis=1) + np.cumsum(rng.normal(0, 1.8, (R, C)), axis=0)
        if i == 4:  # include exact +-pi wrapped values and repeated values
            phi[0, :] = np.pi
        psi = wrap(phi).astype(np.float32)
        n_true = np.round((phi - psi.astype(np.float64)) / TAU).astype(np.int64).ravel()
        k = k_from_n(d, psi, n_true)
        res = residues(d, psi)
        assert np.array_equal(divergence(d, k), supply(d, res))
        n = integrate(d, psi, k)
        assert n[0] == 0
        f, t = d.dual_arcs()
        nrand = rng.integers(-50, 50, d.npix)
        kn = k_from_n(d, psi, nrand)
        cd = OUT / f"case_{i:02d}"
        cd.mkdir()
        np.save(cd / "psi.npy", psi)
        np.save(cd / "residues.npy", res.astype(np.int8))
        np.save(cd / "tail.npy", f.astype(np.int64))
        np.save(cd / "head.npy", t.astype(np.int64))
        np.save(cd / "supply.npy", supply(d, res))
        np.save(cd / "k.npy", k.astype(np.int32))
        np.save(cd / "n.npy", n.astype(np.int32))
        np.save(cd / "nrand.npy", nrand.astype(np.int32))
        np.save(cd / "kn.npy", kn.astype(np.int32))
        has_bad = d.n_loops > 0
        if has_bad:
            kbad = k.copy()
            kbad[int(rng.integers(0, d.n_edges))] += 1
            np.save(cd / "kbad.npy", kbad.astype(np.int32))
        meta = dict(nrow=R, ncol=C, n_residues=int(np.count_nonzero(res)), has_bad=has_bad)
        (cd / "meta.json").write_text(json.dumps(meta, indent=1))
        print(i, R, C, meta)
    (OUT / "summary.json").write_text(json.dumps(dict(n_cases=len(shapes)), indent=1))


if __name__ == "__main__":
    main()
