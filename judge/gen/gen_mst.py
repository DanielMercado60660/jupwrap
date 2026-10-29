"""Goldens for jupwrap-init, MST initialization (DESIGN.md section 8). Reference: judge/ref/flows.py.
Run: .venv/bin/python -m gen.gen_mst"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ref.conventions import Dims, divergence, residues, supply, wrap
from ref.costs import build_costs, convex_envelope
from ref.flows import kruskal, mst_init, mst_weights, table_argmin
from ref.recipe import quadgrid

OUT = Path(__file__).resolve().parent.parent / "golden" / "mst"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rng = np.random.default_rng(41)
    specs = [(1, 7, "smooth", False), (6, 1, "defo", False), (2, 2, "smooth", False), (7, 9, "defo", False),
             (50, 60, "smooth", False), (50, 60, "defo", True), (120, 90, "envelope", False), (300, 300, "smooth", True)]
    for i, (R, C, mode, masked) in enumerate(specs):
        d = Dims(R, C)
        yy, xx = np.mgrid[0:R, 0:C]
        phi = 0.02 * (xx - C / 2) ** 2 / 3 + 0.7 * yy + 2 * np.sin(xx / 6.0)
        psi = wrap(phi + rng.normal(0, 1.0, (R, C))).astype(np.float32)
        corr = rng.uniform(0.05, 0.95, R * C).astype(np.float32)
        mask = (rng.random(R * C) > 0.2).astype(np.uint8) if masked else None
        m = "defo" if mode == "envelope" else mode
        table = build_costs(d, psi, corr, mask, m, 5)
        if mode == "envelope":
            table = convex_envelope(table)
        res = residues(d, psi)
        k, tree, W = mst_init(d, table, res)
        assert np.array_equal(divergence(d, k), supply(d, res))
        cd = OUT / f"case_{i:02d}"
        cd.mkdir()
        np.save(cd / "table.npy", table.astype(np.int32))
        np.save(cd / "residues.npy", res.astype(np.int8))
        np.save(cd / "k0.npy", table_argmin(table).astype(np.int32))
        meta = dict(nrow=R, ncol=C, mode=mode, masked=masked, mst_weight=W, n_residues=int(np.count_nonzero(res)))
        (cd / "meta.json").write_text(json.dumps(meta, indent=1))
        print(i, meta)
    d, res, table = quadgrid(1024, 1024, 21, 4, 30)
    f, t = d.dual_arcs()
    k0 = table_argmin(table)
    w = mst_weights(table, k0)
    tree = kruskal(d.n_dual, f, t, w)
    rec = dict(nrow=1024, ncol=1024, seed=21, B=4, resid_permille=30, mst_weight=int(w[tree].sum()), time_limit_s=20.0)
    (OUT / "recipe.json").write_text(json.dumps(rec, indent=1))
    print(rec)
    (OUT / "summary.json").write_text(json.dumps(dict(n_cases=len(specs)), indent=1))


if __name__ == "__main__":
    main()
