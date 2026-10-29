"""Goldens for jupwrap-conncomp (DESIGN.md section 9). Reference: judge/ref/flows.py.
Run: .venv/bin/python -m gen.gen_conncomp"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ref.conventions import CONNCOMP_COST_THRESH, Dims, min_component_size
from ref.flows import label_components
from ref.pipeline import ref_unwrap
from ref.synth import make_case

OUT = Path(__file__).resolve().parent.parent / "golden" / "conncomp"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    specs = [
        # kind, R, C, seed, L, thresh, frac
        ("topo", 128, 128, 1, 3, CONNCOMP_COST_THRESH, 0.01),
        ("topo", 128, 160, 2, 3, 500, 0.002),
        ("masked_topo", 120, 120, 3, 9, CONNCOMP_COST_THRESH, 0.01),
        ("defo", 96, 128, 4, 4, 800, 0.0),
        ("topo", 64, 64, 5, 2, 200, 0.5),
        ("topo", 1, 50, 6, 9, CONNCOMP_COST_THRESH, 0.01),
        ("topo", 40, 1, 7, 9, CONNCOMP_COST_THRESH, 0.01),
        ("masked_topo", 64, 64, 8, 9, 10**9, 0.01),   # threshold above every weight -> nothing labelled
    ]
    for i, (kind, R, C, seed, L, th, frac) in enumerate(specs):
        if R < 8 or C < 8:
            cs = make_case("topo", 64, 64, seed, nlooks=L, base=(0.2, 0.9))
            sl = (slice(0, R), slice(0, C))
            igram, corr = cs["igram"][sl], cs["corr"][sl]
            mask = None
        else:
            cs = make_case(kind, R, C, seed, nlooks=L)
            igram, corr, mask = cs["igram"], cs["corr"], cs["mask"]
        unw, _, psi, table, k = ref_unwrap(igram, corr, L, "defo" if kind == "defo" else "smooth", mask)
        d = Dims(R, C)
        ms = min_component_size(frac, d.npix)
        lab, nc = label_components(d, table, k, None if mask is None else mask.ravel(), th, ms)
        cd = OUT / f"case_{i:02d}"
        cd.mkdir()
        np.save(cd / "table.npy", table.astype(np.int32))
        np.save(cd / "k.npy", k.astype(np.int32))
        if mask is not None:
            np.save(cd / "mask.npy", mask.ravel().astype(np.uint8))
        np.save(cd / "labels.npy", lab.astype(np.uint32))
        meta = dict(nrow=R, ncol=C, thresh=int(th), min_size=ms, n_components=int(nc), masked=mask is not None)
        (cd / "meta.json").write_text(json.dumps(meta, indent=1))
        print(i, meta)
    (OUT / "summary.json").write_text(json.dumps(dict(n_cases=len(specs)), indent=1))


if __name__ == "__main__":
    main()
