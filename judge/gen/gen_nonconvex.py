"""Goldens for jupwrap-nonconvex (DESIGN.md section 7).

* convex: smooth-mode tables, start = MST init; optimum = OR-tools exact
  (bounds wide enough that the optimum is interior, so it is the unbounded optimum).
* defo / plateau: small grids, start = exact optimum of the integer convex
  surrogate (C8); optimum = CP-SAT over pixel fields (judge/ref/flows.py
  cpsat_opt); reference heuristic = cycle canceling + pixel moves (ref gap).
* medium: invariants only (cost not above start, divergence preserved,
  single-pixel local optimality, time).
Run: .venv/bin/python -m gen.gen_nonconvex
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from ref.conventions import KT, Dims, divergence, residues, supply, table_cost, total_cost, wrap
from ref.costs import build_costs, convex_envelope
from ref.flows import cpsat_opt, cycle_cancel, grid_mcf, mcf_ortools, mst_init, pixel_move_deltas
from ref.synth import make_case

OUT = Path(__file__).resolve().parent.parent / "golden" / "nonconvex"


def save(cd, table, k_start, meta):
    cd.mkdir(parents=True)
    np.save(cd / "table.npy", np.asarray(table, np.int32))
    np.save(cd / "k_start.npy", np.asarray(k_start, np.int32))
    (cd / "meta.json").write_text(json.dumps(meta, indent=1))
    print(cd.name, meta)


def plateau_tables(rng, E):
    ks = np.arange(-KT, KT + 1)
    a = rng.integers(20, 800, E)[:, None]
    u = rng.choice([-1, 0, 0, 0, 1], E)[:, None]
    plat = rng.integers(150, 1500, E)[:, None]
    t = np.minimum(a * (ks[None, :] - u) ** 2, plat)
    return t - t.min(axis=1, keepdims=True)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rng = np.random.default_rng(51)
    idx = 0
    # convex
    for R, C, seed in [(24, 24, 1), (40, 48, 2), (64, 64, 3)]:
        cs = make_case("topo", R, C, seed, nlooks=3, base=(0.2, 0.8), max_grad=1.5)
        d = Dims(R, C)
        psi = np.angle(cs["igram"]).astype(np.float32)
        table = build_costs(d, psi, cs["corr"].ravel(), None, "smooth", 3, (5, 5))
        res = residues(d, psi)
        k0, _, _ = mst_init(d, table, res)
        B = KT + 3
        f, t = d.dual_arcs()
        E = d.n_edges
        ks = np.arange(-B, B + 1)
        tabs = table_cost(np.repeat(table, len(ks), 0), np.tile(ks, E)).reshape(E, -1)
        st, opt, kopt = mcf_ortools(d.n_dual, f, t, np.full(E, -B), np.full(E, B), supply(d, res), tabs, True)
        assert st == "optimal" and np.abs(kopt).max() < B
        save(OUT / f"case_{idx:02d}", table, k0, dict(nrow=R, ncol=C, kind="convex", opt_cost=opt,
                                                       start_cost=total_cost(table, k0)))
        idx += 1
    # small nonconvex with exact optimum
    gaps = {"defo": [], "plateau": []}
    for i in range(24):
        kind = "defo" if i < 16 else "plateau"
        R = C = (10 + (i % 3)) if kind == "defo" else (7 + (i % 3))
        d = Dims(R, C)
        if kind == "defo":
            cs = make_case("defo", R, C, 100 + i, nlooks=3, base=(0.2, 0.8), peak_cycles=4, fault_cycles=2.0)
            psi = np.angle(cs["igram"]).astype(np.float32)
            table = build_costs(d, psi, cs["corr"].ravel(), None, "defo", 3, (3, 3))
        else:
            psi = wrap(rng.normal(0, 2.0, (R, C)).cumsum(axis=1)).astype(np.float32)
            table = plateau_tables(rng, d.n_edges)
        res = residues(d, psi)
        ks, _ = grid_mcf(d, convex_envelope(table), res, B=KT)
        ok, opt, _ = cpsat_opt(d, table, psi, time_limit=300.0)
        assert ok, (i, kind)
        t0 = time.time()
        kref = cycle_cancel(d, table, ks)
        cref = total_cost(table, kref)
        gaps[kind].append((cref - opt) / opt)
        save(OUT / f"case_{idx:02d}", table, ks, dict(nrow=R, ncol=C, kind=kind, opt_cost=opt,
                                                       start_cost=total_cost(table, ks), ref_cost=cref,
                                                       ref_seconds=round(time.time() - t0, 2)))
        idx += 1
    # medium: invariants and time
    for R, C, seed in [(128, 128, 7), (200, 240, 8)]:
        cs = make_case("defo", R, C, seed, nlooks=4, base=(0.3, 0.9), peak_cycles=10, fault_cycles=3.0)
        d = Dims(R, C)
        psi = np.angle(cs["igram"]).astype(np.float32)
        table = build_costs(d, psi, cs["corr"].ravel(), None, "defo", 4)
        res = residues(d, psi)
        ks, _ = grid_mcf(d, convex_envelope(table), res, B=KT)
        save(OUT / f"case_{idx:02d}", table, ks, dict(nrow=R, ncol=C, kind="medium",
                                                       start_cost=total_cost(table, ks), time_limit_s=20.0))
        idx += 1
    summ = dict(n_cases=idx)
    for kind, g in gaps.items():
        summ[f"{kind}_ref_mean_gap"] = float(np.mean(g))
        summ[f"{kind}_ref_max_gap"] = float(np.max(g))
        summ[f"{kind}_mean_gap_limit"] = round(max(0.03, 1.5 * float(np.mean(g))), 4)
        summ[f"{kind}_max_gap_limit"] = round(max(0.15, float(np.max(g)) + 0.05), 4)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(summ)


if __name__ == "__main__":
    main()
