"""Goldens for jupwrap-costs (DESIGN.md section 5, eqs. C1-C8).
Reference: judge/ref/costs.py. Run: .venv/bin/python -m gen.gen_costs"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ref.conventions import KT, Dims, wrap
from ref.costs import (convex_envelope, effective_coherence, float_costs, mean_gradients, pixel_sigma2,
                       quantize)

OUT = Path(__file__).resolve().parent.parent / "golden" / "costs"

CASES = [
    # R, C, mode, nlooks, window, sigma2_signal, p_disc, masked
    (40, 50, "smooth", 9.0, (7, 7), 0.02, 1e-4, False),
    (33, 47, "defo", 4.0, (5, 9), 0.02, 1e-4, True),
    (20, 20, "smooth", 1.0, (1, 1), 0.02, 1e-4, False),
    (25, 31, "defo", 2.5, (6, 4), 0.02, 1e-4, True),
    (30, 30, "smooth", 50.0, (3, 3), 0.02, 1e-4, False),
    (1, 9, "smooth", 5.0, (7, 7), 0.02, 1e-4, False),
    (9, 1, "defo", 5.0, (7, 7), 0.02, 1e-4, False),
    (28, 36, "defo", 6.0, (7, 7), 0.1, 1e-2, False),
    (64, 64, "smooth", 3.0, (15, 11), 0.005, 1e-4, True),
]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rng = np.random.default_rng(31)
    for i, (R, C, mode, L, win, s2s, pd, masked) in enumerate(CASES):
        d = Dims(R, C)
        yy, xx = np.mgrid[0:R, 0:C]
        phi = 0.9 * xx + 0.4 * yy + 2.0 * np.sin(xx / 5.0 + yy / 7.0)
        psi = wrap(phi + rng.normal(0, 0.7, (R, C))).astype(np.float32)
        corr = rng.uniform(0, 1, (R, C)).astype(np.float32)
        corr[rng.random((R, C)) < 0.03] = np.nan
        corr[rng.random((R, C)) < 0.02] = 1.3
        corr[rng.random((R, C)) < 0.02] = -0.2
        corr[rng.random((R, C)) < 0.05] = 1.0
        corr[rng.random((R, C)) < 0.05] = 0.0
        mask = None
        if masked:
            mask = (rng.random((R, C)) > 0.15).astype(np.uint8)
            mask[R // 3: R // 2, C // 4: C // 2] = 0
        mu, resultant = mean_gradients(d, psi, mask, win, return_resultant=True)
        mu_direct = mean_gradients(d, psi, mask, win, direct=True)
        # edges whose window mean is well conditioned (or empty) are checked exactly
        check = (resultant == 0.0) | (resultant >= 1e-6)
        assert np.abs(wrap(mu - mu_direct))[check].max() < 1e-9
        c, valid = float_costs(d, psi, corr.ravel(), None if mask is None else mask.ravel(), mode, L, win, s2s, pd, KT)
        table = quantize(c, valid)
        env = convex_envelope(table)
        cd = OUT / f"case_{i:02d}"
        cd.mkdir()
        np.save(cd / "psi.npy", psi)
        np.save(cd / "corr.npy", corr)
        if mask is not None:
            np.save(cd / "mask.npy", mask)
        np.save(cd / "effcoh.npy", effective_coherence(corr.ravel(), L))
        np.save(cd / "sigma2.npy", pixel_sigma2(corr.ravel(), L))
        np.save(cd / "mu.npy", mu)
        np.save(cd / "check.npy", check.astype(np.uint8))
        np.save(cd / "table.npy", table.astype(np.int32))
        np.save(cd / "envelope.npy", env.astype(np.int32))
        meta = dict(nrow=R, ncol=C, mode=mode, nlooks=L, window=list(win), sigma2_signal=s2s, p_discontinuity=pd,
                    masked=masked)
        (cd / "meta.json").write_text(json.dumps(meta, indent=1))
        print(i, meta, "max", int(table.max()), "unchecked", int((~check).sum()))
    # random integer rows for the envelope (nonconvex, ties, plateaus)
    rows = []
    for _ in range(3000):
        kind = rng.integers(0, 4)
        if kind == 0:
            r = rng.integers(0, 1000, 2 * KT + 1)
        elif kind == 1:
            r = rng.integers(0, 3, 2 * KT + 1) * 7
        elif kind == 2:
            ks = np.arange(-KT, KT + 1)
            r = np.minimum(rng.integers(1, 400) * (ks - rng.integers(-2, 3)) ** 2, rng.integers(50, 2000))
        else:
            r = np.sort(rng.integers(0, 500, 2 * KT + 1))[::-1] if rng.random() < 0.5 else np.zeros(2 * KT + 1, np.int64)
        r = r - r.min()
        rows.append(r)
    rows = np.array(rows, np.int64)
    rd = OUT / "envelope_random"
    rd.mkdir()
    np.save(rd / "rows.npy", rows.astype(np.int32))
    np.save(rd / "envelope.npy", convex_envelope(rows).astype(np.int32))
    (OUT / "summary.json").write_text(json.dumps(dict(n_cases=len(CASES)), indent=1))


if __name__ == "__main__":
    main()
