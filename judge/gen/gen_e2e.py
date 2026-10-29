"""Reference numbers for the end-to-end judges (pipeline, tiling, python).
Reference = judge/ref/pipeline.py (exact OR-tools optimum for smooth costs or
the defo convex surrogate, then greedy pixel moves for defo). Thresholds are
written next to the numbers with the reasoning in DESIGN.md section 14.
Run: .venv/bin/python -m gen.gen_e2e"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ref import e2e_cases as E
from ref.metrics import fraction_correct, truth_regions
from ref.pipeline import ref_unwrap
from ref.scoring import cost_of_unw, ref_table

OUT = Path(__file__).resolve().parent.parent / "golden" / "e2e"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exp = {}
    for name in E.CASES:
        cs = E.get(name)
        rec = dict(input_sha256=E.input_hash(cs), shape=list(cs["igram"].shape))
        if not name.startswith("strip_33000"):
            t0 = time.time()
            unw, lab, psi, table, k = ref_unwrap(cs["igram"], cs["corr"], cs["nlooks"], cs["cost"], cs["mask"])
            reg = truth_regions(cs["rho"], cs["hole"], cs["mask"])
            fc, per = fraction_correct(unw, cs["phi"], reg)
            d, psi2, t2 = ref_table(cs)
            rec.update(ref_frac=float(fc), ref_cost=cost_of_unw(d, psi2, t2, unw, cs["mask"]),
                       ref_seconds=round(time.time() - t0, 1), n_regions=len(per),
                       region_pixels=int((reg > 0).sum()))
        exp[name] = rec
        print(name, rec)
    (OUT / "expected.json").write_text(json.dumps(exp, indent=1))


if __name__ == "__main__":
    main()
